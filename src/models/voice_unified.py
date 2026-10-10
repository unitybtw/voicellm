"""One locally initialized speech/text model. No pretrained weights or remote loaders."""
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.models.micro_voice_slm import MicroVoiceSLM
from src.codec.nano_codec import ResidualVectorQuantizer
from src.checkpoint import save_atomic


class ByteTokenizer:
    pad, bos, eos, speech = 0, 1, 2, 3
    vocab_size = 260

    def encode(self, text):
        if not text.strip():
            raise ValueError("Text must not be empty")
        return [self.bos] + [b + 4 for b in text.encode('utf-8')] + [self.eos]

    def decode(self, ids):
        return bytes(i - 4 for i in ids if 4 <= i < 260).decode('utf-8', errors='replace')


@dataclass
class UnifiedConfig:
    sample_rate: int = 16000
    hop_length: int = 80
    codec_dim: int = 96
    codebook_size: int = 256
    codebooks: int = 24
    hidden_size: int = 384
    layers: int = 8
    heads: int = 6
    kv_heads: int = 2
    intermediate_size: int = 1024
    max_seq_len: int = 2048
    gain_bins: int = 256
    depth_hidden: int = 128

    def __post_init__(self):
        if any(value <= 0 for key, value in asdict(self).items() if key != 'depth_hidden') or self.depth_hidden < 0:
            raise ValueError("Configuration values must be positive")
        if self.gain_bins != 1 and (self.gain_bins != self.codebook_size or self.codebooks < 2):
            raise ValueError('Gain coding requires one vocabulary-sized gain table and acoustic tables')
        if self.hidden_size % self.heads or self.heads % self.kv_heads or (self.hidden_size // self.heads) % 2:
            raise ValueError("Invalid attention head dimensions")


class CausalFrameStack(nn.Module):
    history_frames = 6

    def __init__(self, dim):
        super().__init__()
        self.layers = nn.ModuleList(nn.Conv1d(dim, dim, 3) for _ in range(3))

    def forward(self, x):
        for layer in self.layers:
            x = x + F.silu(layer(F.pad(x, (2, 0))))
        return x


class CausalFrameCodec(nn.Module):
    """Native configurable frame codec; decoding uses past frames only."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_in = nn.Linear(config.hop_length, config.codec_dim)
        self.encoder = CausalFrameStack(config.codec_dim)
        acoustic_books = config.codebooks - int(config.gain_bins > 1)
        self.rvq = ResidualVectorQuantizer(acoustic_books, config.codebook_size, config.codec_dim)
        levels = (torch.cat([torch.zeros(1), torch.logspace(-5, 0, config.gain_bins-1)])
                  if config.gain_bins > 1 else torch.ones(1))
        self.register_buffer('gain_levels', levels)
        self.decoder = CausalFrameStack(config.codec_dim)
        self.decoder_out = nn.Linear(config.codec_dim, config.hop_length)

    def input_frames(self, wav):
        if wav.ndim != 2 or wav.shape[-1] == 0 or not torch.isfinite(wav).all():
            raise ValueError("Expected finite audio [batch, samples]")
        wav = wav.to(device=self.encoder_in.weight.device, dtype=self.encoder_in.weight.dtype)
        wav = F.pad(wav, (0, (-wav.shape[-1]) % self.config.hop_length))
        frames = wav.unfold(-1, self.config.hop_length, self.config.hop_length)
        return frames

    def frame_gain_tokens(self, wav):
        peaks = self.input_frames(wav).abs().amax(-1).float()
        if self.config.gain_bins == 1:
            raise ValueError('Gain coding is disabled')
        levels = self.gain_levels[1:].float().log()
        indices = (peaks.clamp_min(1e-5).log()[..., None]-levels).abs().argmin(-1)+1
        return torch.where(peaks == 0, 0, indices)

    def encode_latents(self, wav):
        frames = self.input_frames(wav)
        if self.config.gain_bins > 1:
            # Preserve a gain symbol rather than making quiet speech compete with loud frames.
            frames = frames / frames.abs().amax(-1, keepdim=True).clamp_min(1e-5) * .1
        z = self.encoder(self.encoder_in(frames).transpose(1, 2))
        return z

    def encode(self, wav):
        z, tokens, loss = self.rvq(self.encode_latents(wav))
        if self.config.gain_bins > 1:
            tokens = torch.cat([self.frame_gain_tokens(wav).unsqueeze(1), tokens], dim=1)
        return z, tokens, loss

    def decode_latents(self, z, gain_tokens=None):
        frames = self.decoder_out(self.decoder(z).transpose(1, 2)).tanh()
        if self.config.gain_bins > 1:
            if gain_tokens is None:
                raise ValueError('Normalized latents require gain tokens')
            frames = (frames * (self.gain_levels[gain_tokens].unsqueeze(-1)/.1)).clamp(-1, 1)
        return frames.flatten(1)

    def decode(self, tokens):
        if self.config.gain_bins > 1:
            return self.decode_latents(self.rvq.decode_indices(tokens[:, 1:]), tokens[:, 0])
        return self.decode_latents(self.rvq.decode_indices(tokens))

    def decode_frame(self, tokens, history=None):
        if history is not None:
            tokens = torch.cat([history, tokens], -1)
        wav = self.decode(tokens)[..., -self.config.hop_length:]
        return wav, tokens[..., -self.decoder.history_frames:]


class UnifiedVoiceModel(nn.Module):
    format_version = 1

    def __init__(self, config=None):
        super().__init__()
        self.config = config or UnifiedConfig()
        c = self.config
        self.tokenizer = ByteTokenizer()
        self.codec = CausalFrameCodec(c)
        self.backbone = MicroVoiceSLM(
            vocab_size=self.tokenizer.vocab_size, audio_vocab_size=c.codebook_size,
            num_audio_codebooks=c.codebooks, hidden_size=c.hidden_size, num_layers=c.layers,
            num_heads=c.heads, num_kv_heads=c.kv_heads, intermediate_size=c.intermediate_size,
            max_seq_len=c.max_seq_len,
        )
        # Tie input/output weights: one vocabulary matrix for each modality.
        self.backbone.text_head.weight = self.backbone.text_embedding.weight
        for head, embedding in zip(self.backbone.audio_heads, self.backbone.audio_embeddings):
            head.weight = embedding.weight
        nn.init.normal_(self.backbone.text_embedding.weight, std=.02)
        for embedding in self.backbone.audio_embeddings:
            nn.init.normal_(embedding.weight, std=.02)
        self.stop_head = nn.Linear(c.hidden_size, 1)
        if c.depth_hidden:
            self.depth_in = nn.Linear(c.hidden_size, c.depth_hidden, bias=False)
            self.depth_out = nn.Linear(c.depth_hidden, c.hidden_size, bias=False)
            # Preserve the previous native model's logits at the upgrade boundary.
            nn.init.zeros_(self.depth_out.weight)
        self.register_buffer('codec_steps', torch.tensor(0, dtype=torch.long))
        self.register_buffer('speech_steps', torch.tensor(0, dtype=torch.long))
        self.register_buffer('text_steps', torch.tensor(0, dtype=torch.long))
        self.register_buffer('audio_steps', torch.tensor(0, dtype=torch.long))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def conditioned_audio_logits(self, hidden, targets, pad_to_multiple=1):
        """Joint frame distribution: codebook k sees only target books < k."""
        if not self.config.depth_hidden:
            return torch.stack([head(hidden) for head in self.backbone.audio_heads], 1)
        length = hidden.shape[1]
        extra = (-length) % pad_to_multiple
        if extra:
            hidden = F.pad(hidden, (0, 0, 0, extra))
            targets = F.pad(targets, (0, extra))
        prefix = torch.zeros_like(hidden)
        logits = []
        for index, head in enumerate(self.backbone.audio_heads):
            state = hidden
            if self.config.depth_hidden:
                state = hidden + self.depth_out(F.silu(self.depth_in(hidden + prefix / math.sqrt(max(index, 1)))))
            logits.append(head(state))
            prefix = prefix + self.backbone.audio_embeddings[index](targets[:, index])
        return torch.stack(logits, 1)[:, :, :length]

    def sample_audio_tokens(self, hidden, temperature=0., pad_to_multiple=1):
        """Roll out each frame's codebooks using predictions, never gold prefixes."""
        if temperature < 0 or not math.isfinite(temperature):
            raise ValueError('Invalid sampling temperature')
        if pad_to_multiple < 1:
            raise ValueError('Padding multiple must be positive')
        length = hidden.shape[1]
        extra = (-length) % pad_to_multiple
        if extra:
            hidden = F.pad(hidden, (0, 0, 0, extra))
        prefix = torch.zeros_like(hidden)
        frame = []
        for book, head in enumerate(self.backbone.audio_heads):
            state = hidden
            if self.config.depth_hidden:
                state = hidden + self.depth_out(F.silu(self.depth_in(hidden + prefix / math.sqrt(max(book, 1)))))
            logits = head(state)
            if temperature == 0:
                token = logits.argmax(-1)
            else:
                values, indices = (logits / temperature).topk(min(8, self.config.codebook_size), -1)
                picks = torch.multinomial(values.softmax(-1).reshape(-1, values.shape[-1]), 1)
                token = indices.gather(-1, picks.reshape(*indices.shape[:-1], 1)).squeeze(-1)
            frame.append(token)
            prefix = prefix + self.backbone.audio_embeddings[book](token)
        return torch.stack(frame, 1)[:, :, :length]

    def codec_loss(self, wav, quantize=True):
        gains = None
        if quantize:
            z, tokens, vq = self.codec.encode(wav)
            if self.config.gain_bins > 1:
                gains = tokens[:, 0]
        else:
            z = self.codec.encode_latents(wav)
            vq = wav.new_zeros(())
            if self.config.gain_bins > 1:
                gains = self.codec.frame_gain_tokens(wav)
        reconstructed = self.codec.decode_latents(z, gains)[..., :wav.shape[-1]]
        spectral = wav.new_zeros(())
        convergence = wav.new_zeros(())
        for size in (256, 512, 1024):
            window = torch.hann_window(size, device=wav.device)
            kwargs = dict(n_fft=size, hop_length=size // 4, window=window,
                          return_complex=True, pad_mode='constant')
            actual = torch.stft(wav.float(), **kwargs).abs()
            predicted = torch.stft(reconstructed.float(), **kwargs).abs()
            spectral = spectral + (predicted.log1p() - actual.log1p()).abs().mean()
            convergence = convergence + (predicted-actual).flatten(1).norm(dim=1).div(actual.flatten(1).norm(dim=1).clamp_min(.1)).mean()
        scale = wav.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(.01)
        waveform = ((reconstructed-wav).abs()/scale).mean()
        derivative = ((torch.diff(reconstructed)-torch.diff(wav)).abs()/scale).mean() if wav.shape[-1] > 1 else wav.new_zeros(())
        hop = self.config.hop_length
        if wav.shape[-1] > hop:
            pred_jumps = reconstructed[:, hop::hop]-reconstructed[:, hop-1:-1:hop]
            true_jumps = wav[:, hop::hop]-wav[:, hop-1:-1:hop]
            boundary = ((pred_jumps-true_jumps).abs()/scale).mean()
        else:
            boundary = wav.new_zeros(())
        reconstruction = waveform + .25 * convergence / 3 + spectral / 3 + .25 * derivative + .5 * boundary
        return {'loss': reconstruction + .25 * vq, 'reconstruction': reconstruction, 'waveform': waveform, 'boundary': boundary, 'vq': vq}

    def audio_loss(self, wav=None, tokens=None):
        """Audio-only next-frame pretraining; teaches no text or end-of-utterance labels."""
        if tokens is None:
            with torch.no_grad():
                _, tokens, _ = self.codec.encode(wav)
        if tokens.ndim != 3 or tokens.shape[1] != self.config.codebooks or tokens.shape[-1] < 1:
            raise ValueError('Expected nonempty acoustic tokens [batch, codebooks, frames]')
        if tokens.shape[-1] + 1 > self.config.max_seq_len:
            raise ValueError('Audio clip exceeds context')
        marker = torch.full((tokens.shape[0],1), self.tokenizer.speech, device=tokens.device)
        inputs = torch.cat([self.backbone.text_embedding(marker), self.backbone.embed_audio_tokens(tokens)], dim=1)
        outputs, _ = self.backbone(inputs_embeds=inputs, compute_audio_logits=False)
        logits = self.conditioned_audio_logits(outputs['hidden_states'][:, :tokens.shape[-1]], tokens)
        loss = F.cross_entropy(logits.reshape(-1, self.config.codebook_size), tokens.reshape(-1))
        accuracy = (logits.argmax(-1) == tokens).float().mean().detach()
        return {'loss': loss, 'token_accuracy': accuracy}

    @torch.no_grad()
    def predicted_training_tokens(self, text, tokens, pad_to_multiple=1):
        """Causal predictions for exposure training; frame t never sees target t."""
        ids = torch.tensor([self.tokenizer.encode(text) + [self.tokenizer.speech]], device=tokens.device)
        embeddings = torch.cat([self.backbone.text_embedding(ids), self.backbone.embed_audio_tokens(tokens)], 1)
        outputs, _ = self.backbone(inputs_embeds=embeddings, pad_to_multiple=pad_to_multiple,
                                   compute_audio_logits=False)
        hidden = outputs['hidden_states'][:, ids.shape[1]-1:ids.shape[1]-1+tokens.shape[-1]]
        return self.sample_audio_tokens(hidden, pad_to_multiple=pad_to_multiple)

    def speech_loss(self, text, wav=None, tokens=None, pad_to_multiple=1, input_tokens=None, coarse_weight=1.):
        """Teacher-forced text-to-audio objective using the trained, frozen codec."""
        if tokens is None:
            with torch.no_grad():
                _, tokens, _ = self.codec.encode(wav)
        if tokens.ndim != 3 or tokens.shape[0] != 1 or tokens.shape[1] != self.config.codebooks or tokens.shape[-1] < 1:
            raise ValueError('Speech training requires one nonempty acoustic sequence')
        if input_tokens is not None and input_tokens.shape != tokens.shape:
            raise ValueError('Exposure inputs must match the aligned acoustic target shape')
        ids = torch.tensor([self.tokenizer.encode(text) + [self.tokenizer.speech]], device=tokens.device)
        text_emb = self.backbone.text_embedding(ids)
        audio_emb = self.backbone.embed_audio_tokens(tokens if input_tokens is None else input_tokens)
        outputs, _ = self.backbone(inputs_embeds=torch.cat([text_emb, audio_emb], 1),
                                   pad_to_multiple=pad_to_multiple, compute_audio_logits=False)
        start = ids.shape[1] - 1
        logits = self.conditioned_audio_logits(outputs['hidden_states'][:, start:start + tokens.shape[-1]],
                                               tokens, pad_to_multiple=pad_to_multiple)
        if coarse_weight < 1 or not math.isfinite(coarse_weight):
            raise ValueError('Coarse audio weight must be finite and >= 1')
        per_book = F.cross_entropy(logits.reshape(-1, self.config.codebook_size), tokens.reshape(-1),
                                  reduction='none').reshape(tokens.shape).mean((0, 2))
        audio_loss = per_book.mean()
        weights = torch.ones_like(per_book)
        weights[:min(4, self.config.codebooks)] = coarse_weight
        weighted_audio = (per_book * weights).sum() / weights.sum()
        text_logits = outputs['text_logits'][:, :ids.shape[1] - 1]
        text_loss = F.cross_entropy(text_logits.reshape(-1, 260), ids[:, 1:].reshape(-1))
        stop_logits = self.stop_head(outputs['hidden_states'][:, start:]).squeeze(-1)
        stops = torch.zeros_like(stop_logits)
        stops[:, -1] = 1
        # Clip-end positives are rare; balance them instead of learning "never stop".
        stop_loss = F.binary_cross_entropy_with_logits(stop_logits, stops,
                          pos_weight=stop_logits.new_tensor(float(tokens.shape[-1])))
        accuracy = (logits.argmax(-1) == tokens).float().mean().detach()
        return {'loss': weighted_audio + .1 * text_loss + .1 * stop_loss,
                'audio': audio_loss, 'coarse_audio':per_book[:min(4,self.config.codebooks)].mean(),
                'text': text_loss, 'stop': stop_loss, 'token_accuracy': accuracy}

    def text_loss(self, text):
        device = next(self.parameters()).device
        ids = torch.tensor([self.tokenizer.encode(text)], device=device)
        outputs, _ = self.backbone(text_tokens=ids[:, :-1])
        return {'loss': F.cross_entropy(outputs['text_logits'].reshape(-1, 260), ids[:, 1:].reshape(-1))}

    @torch.inference_mode()
    def generate_text(self, prompt, max_tokens=64, allow_untrained=False):
        if not allow_untrained and self.text_steps.item() + self.speech_steps.item() == 0:
            raise RuntimeError("Language training is required before generating text")
        if self.training:
            raise RuntimeError("Call eval() before inference")
        ids = self.tokenizer.encode(prompt)[:-1]
        if max_tokens < 0 or len(ids) + max_tokens > self.config.max_seq_len:
            raise ValueError("Invalid text generation length")
        device = next(self.parameters()).device
        outputs, caches = self.backbone(text_tokens=torch.tensor([ids], device=device))
        result = []
        for index in range(max_tokens):
            logits = outputs['text_logits'][:, -1].clone()
            logits[:, [self.tokenizer.pad, self.tokenizer.bos, self.tokenizer.speech]] = float('-inf')
            token = logits.argmax(-1).item()
            if token == self.tokenizer.eos:
                break
            result.append(token)
            if index + 1 < max_tokens:
                outputs, caches = self.backbone(text_tokens=torch.tensor([[token]], device=device), past_kv_caches=caches)
        return self.tokenizer.decode(result)

    @torch.inference_mode()
    def stream(self, text, max_frames=150, temperature=0., reference_wav=None,
               interruption=None, allow_untrained=False):
        if not allow_untrained and self.speech_steps.item() == 0:
            raise RuntimeError("No speech training yet. Train on real audio/transcript pairs first.")
        if max_frames < 0 or temperature < 0 or not math.isfinite(temperature):
            raise ValueError("Invalid generation settings")
        if self.training:
            raise RuntimeError("Call eval() before inference")
        if max_frames == 0 or (interruption and interruption.get('interrupt')):
            return
        device = next(self.parameters()).device
        ids = torch.tensor([self.tokenizer.encode(text)], device=device)
        embeds = self.backbone.text_embedding(ids)
        if reference_wav is not None:
            _, reference_tokens, _ = self.codec.encode(reference_wav.to(device))
            embeds = torch.cat([self.backbone.embed_audio_tokens(reference_tokens), embeds], 1)
        marker = torch.tensor([[self.tokenizer.speech]], device=device)
        embeds = torch.cat([embeds, self.backbone.text_embedding(marker)], 1)
        if embeds.shape[1] + max_frames > self.config.max_seq_len:
            raise ValueError("Text/reference and output exceed the context window")
        outputs, caches = self.backbone(inputs_embeds=embeds, compute_audio_logits=False)
        history = None
        for index in range(max_frames):
            if interruption and interruption.get('interrupt'):
                return
            if not allow_untrained and index > 0 and self.stop_head(outputs['hidden_states'][:, -1]).sigmoid().item() > .5:
                return
            hidden = outputs['hidden_states'][:, -1:]
            tokens = self.sample_audio_tokens(hidden, temperature)
            audio, history = self.codec.decode_frame(tokens, history)
            if interruption and interruption.get('interrupt'):
                return
            yield audio
            if index + 1 < max_frames:
                outputs, caches = self.backbone(audio_tokens=tokens, past_kv_caches=caches, compute_audio_logits=False)

    def save(self, path, fp16=False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Remove duplicate storage keys for tied heads. Restore aliases on load.
        state = {key: (value.half() if fp16 and value.is_floating_point() else value)
                 for key, value in self.state_dict().items()
                 if key != 'backbone.text_head.weight' and not key.startswith('backbone.audio_heads.')}
        save_atomic({'format_version': self.format_version, 'config': asdict(self.config),
                    'state': state, 'external_weights': False}, path)
        return path.stat().st_size

    @classmethod
    def load(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint.get('model_kind') == 'phoneme_aligned_spectrum':
            from src.models.aligned_voice import AlignedVoiceModel
            return AlignedVoiceModel.from_checkpoint(checkpoint, device)
        if checkpoint.get('model_kind') == 'spectral_continuous':
            from src.models.spectral_voice import SpectralVoiceModel
            return SpectralVoiceModel.from_checkpoint(checkpoint, device)
        if checkpoint.get('format_version') != cls.format_version or checkpoint.get('external_weights') is not False:
            raise ValueError("Not a native unified checkpoint")
        config = dict(checkpoint['config'])
        config.setdefault('gain_bins', 1)  # Pre-gain checkpoints used every table for RVQ.
        config.setdefault('depth_hidden', 0)  # Older native models used independent heads.
        model = cls(UnifiedConfig(**config))
        state = checkpoint['state']
        state.setdefault('text_steps', torch.tensor(0, dtype=torch.long))
        state.setdefault('audio_steps', torch.tensor(0, dtype=torch.long))
        state.setdefault('codec.gain_levels', model.codec.gain_levels)
        state['backbone.text_head.weight'] = state['backbone.text_embedding.weight']
        for index in range(model.config.codebooks):
            state[f'backbone.audio_heads.{index}.weight'] = state[f'backbone.audio_embeddings.{index}.weight']
        stored_dtype = state['backbone.text_embedding.weight'].dtype
        model = model.to(dtype=stored_dtype)
        model.load_state_dict(state, strict=True)
        return model.to(device).eval()
