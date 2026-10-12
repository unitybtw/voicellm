"""Single native phoneme/duration/spectrum/pitch network with data-derived alignment."""
from dataclasses import dataclass,asdict
from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.models.micro_voice_slm import MicroVoiceSLM
from src.models.voice_unified import ByteTokenizer
from src.models.monotonic_alignment import monotonic_path
from src.text.phonemes import EnglishPhonemes
from src.codec.linear_spectrum import LinearSpectrum
from src.codec.pitch import pitch_labels
from src.checkpoint import save_atomic

@dataclass
class AlignedConfig:
    sample_rate:int=16000
    hop_length:int=256
    n_fft:int=1024
    hidden_size:int=384
    num_layers:int=8
    num_heads:int=6
    num_kv_heads:int=2
    intermediate_size:int=1024
    max_seq_len:int=2048
    griffin_lim_iterations:int=32
    clock_mode:str='hard'
    periodic_prior_mode:str='probability'
    def __post_init__(self):
        if any(type(v)!=int or v<1 for k,v in asdict(self).items() if k not in ('clock_mode','periodic_prior_mode')):raise ValueError('Positive integer configuration required')
        if self.hop_length>self.n_fft//2:raise ValueError('Invalid FFT hop')
        if self.clock_mode not in ('hard','soft'):raise ValueError('Invalid duration clock mode')
        if self.periodic_prior_mode not in ('probability','confidence'):raise ValueError('Invalid periodic prior mode')

class AlignedVoiceModel(nn.Module):
    model_kind='phoneme_aligned_spectrum';format_version=3
    def __init__(self,config=None):
        super().__init__();self.config=config or AlignedConfig();c=self.config
        self.tokenizer=ByteTokenizer();self.phonemizer=EnglishPhonemes()
        self.codec=LinearSpectrum(c.sample_rate,c.n_fft,c.hop_length,c.griffin_lim_iterations)
        # Coarse articulation features for alignment: harmonic-bin details should
        # not decide which phoneme gets a frame. Recomputed DSP, not model weights.
        edges=700*(10**(torch.linspace(0,2595*torch.log10(torch.tensor(1+c.sample_rate/1400)),42)/2595)-1)
        frequencies=torch.linspace(0,c.sample_rate/2,c.n_fft//2+1)
        filters=torch.minimum((frequencies[None]-edges[:-2,None])/(edges[1:-1,None]-edges[:-2,None]),
            (edges[2:,None]-frequencies[None])/(edges[2:,None]-edges[1:-1,None])).clamp_min(0)
        self.register_buffer('alignment_filters',filters/filters.sum(-1,keepdim=True).clamp_min(1e-8),persistent=False)
        self.backbone=MicroVoiceSLM(vocab_size=260,num_audio_codebooks=0,hidden_size=c.hidden_size,num_layers=c.num_layers,num_heads=c.num_heads,
            num_kv_heads=c.num_kv_heads,intermediate_size=c.intermediate_size,max_seq_len=c.max_seq_len)
        self.backbone.text_head.weight=self.backbone.text_embedding.weight
        self.phone_embedding=nn.Embedding(len(self.phonemizer.symbols),c.hidden_size)
        self.phone_context=nn.ModuleList([nn.Conv1d(c.hidden_size,c.hidden_size,5,padding=2) for _ in range(2)])
        self.phone_norm=nn.LayerNorm(c.hidden_size)
        bins=c.n_fft//2+1
        self.prior=nn.Linear(c.hidden_size,bins)
        self.duration=nn.Sequential(nn.Linear(c.hidden_size,c.hidden_size//2),nn.SiLU(),nn.Linear(c.hidden_size//2,1))
        nn.init.zeros_(self.duration[-1].weight);nn.init.constant_(self.duration[-1].bias,math.log(4.))
        self.position=nn.Linear(4,c.hidden_size)
        self.spectrum_out=nn.Linear(c.hidden_size,bins)
        self.pitch_out=nn.Linear(c.hidden_size,2)
        nn.init.zeros_(self.pitch_out.weight);nn.init.zeros_(self.pitch_out.bias)
        for name in ('speech_steps','text_steps','audio_steps','codec_steps'):self.register_buffer(name,torch.tensor(0))

    def encode_text(self,text):
        ids=self.phonemizer.encode(text)
        if len(ids)>self.config.max_seq_len:raise ValueError('Phoneme text exceeds context')
        encoded=self.phone_embedding(torch.tensor([ids],device=self.spectrum_out.weight.device))
        for conv in self.phone_context:encoded=encoded+F.silu(conv(encoded.transpose(1,2)).transpose(1,2))
        encoded=self.phone_norm(encoded)
        return encoded,self.duration(encoded).float().squeeze(-1).clamp(-2,5),self.prior(encoded)

    def alignment(self,prior,features,guidance=.1):
        target=features.to(prior).float()[0]@self.alignment_filters.float().T
        means=prior.float()[0]@self.alignment_filters.float().T
        cost=target.square().mean(-1)[:,None]+means.square().mean(-1)[None]-2*target@means.T/target.shape[-1]
        t=torch.linspace(0,1,target.shape[0],device=cost.device)[:,None]
        p=torch.linspace(0,1,means.shape[0],device=cost.device)[None]
        cost=cost+guidance*((t-p)/.2).square()
        minimum=2 if target.shape[0]>=2*means.shape[0] else 1
        return monotonic_path(cost.repeat_interleave(minimum,dim=1))//minimum

    def expand(self,encoded,path):
        units=encoded.shape[1];frames=path.numel()
        durations=torch.bincount(path,minlength=units)
        offsets=durations.cumsum(0)-durations
        local=(torch.arange(frames,device=path.device)-offsets[path]).float()/durations[path].clamp_min(1).float()
        globalpos=torch.linspace(0,1,frames,device=path.device)
        positions=torch.stack((local,local.square(),globalpos,torch.sin(globalpos*6.2831853)),-1)
        return encoded[:,path]+self.position(positions.to(encoded.dtype))[None],durations

    def decode_hidden(self,expanded,pad=1):
        outputs,_=self.backbone(inputs_embeds=expanded,pad_to_multiple=pad,compute_audio_logits=False)
        hidden=outputs['hidden_states'];raw=self.spectrum_out(hidden)
        pitch=self.pitch_out(hidden).float()
        # The periodic comb creates many spectral local minima around 200 Hz.
        # Train pitch/voicing from waveform labels, not from those comb gradients.
        acoustic_controls=pitch.detach()
        f0=(acoustic_controls[...,0].clamp(-2.1,1.4)*.5+math.log(200.)).exp()
        frequencies=torch.linspace(0,self.config.sample_rate/2,raw.shape[-1],device=raw.device)
        comb=(.08+.92*torch.exp(-8*torch.sin(math.pi*frequencies[None,None]/f0[...,None]).square())).log()
        # A learned pitch/voicing head constrains voiced spectra; all heads are inside this model.
        voicing=acoustic_controls[...,1:].sigmoid()
        if self.config.periodic_prior_mode=='confidence':
            # An uncertain/unvoiced classifier must not impose harmonic holes
            # on consonant/noise spectra before the DSP even receives them.
            voicing=(2*voicing-1).clamp_min(0)
        adjustment=comb*voicing/self.codec.std.to(raw.device).float().clamp_min(.3)
        return raw+adjustment.to(raw.dtype),pitch

    def expand_soft(self,encoded,logdur,frames):
        """Differentiable predicted timing, so acoustic error trains the duration head."""
        durations=logdur.exp().clamp_min(2.)
        total=durations.sum(-1,keepdim=True)
        offsets=durations.cumsum(-1)-durations
        centers=(offsets+durations*.5)/total
        positions=(torch.arange(frames,device=encoded.device).float()+.5)/frames
        widths=(durations/total*.35).clamp_min(.75/frames)
        weights=(-.5*((positions[None,:,None]-centers[:,None,:])/widths[:,None,:]).square()).softmax(-1)
        local=((positions[None,:,None]*total[:,None]-offsets[:,None])/durations[:,None]).clamp(0,1)
        local=(weights*local).sum(-1)
        globalpos=positions[None].expand(encoded.shape[0],-1)
        phases=torch.stack((local,local.square(),globalpos,torch.sin(globalpos*6.2831853)),-1)
        return weights.to(encoded.dtype)@encoded+self.position(phases.to(encoded.dtype))

    def predict_acoustics(self,text,frames=None,pad_to_multiple=1,clock_mode=None):
        encoded,logdur,_=self.encode_text(text)
        mode = self.config.clock_mode if clock_mode is None else clock_mode
        if frames is not None and (type(frames)!=int or frames<encoded.shape[1] or frames>self.config.max_seq_len):
            raise ValueError('Invalid output duration')
        if mode=='soft':
            if frames is None and 2*encoded.shape[1]>self.config.max_seq_len:raise ValueError('Phoneme durations exceed context')
            total=logdur.exp().clamp_min(2.).sum()
            length=min(self.config.max_seq_len,max(encoded.shape[1]*2,int(total.detach().round().item()))) if frames is None else frames
            if length<encoded.shape[1] or length>self.config.max_seq_len:raise ValueError('Invalid output duration')
            predicted,pitch=self.decode_hidden(self.expand_soft(encoded,logdur,length),pad_to_multiple)
            return predicted,pitch,total
        durations=logdur.exp()[0].round().long().clamp_min(1)
        total=logdur.exp().sum()
        limit=self.config.max_seq_len if frames is None else frames
        if limit<durations.numel():raise ValueError('Output limit is shorter than the phoneme sequence')
        if int(durations.sum())>limit or frames is not None:
            # Largest remainder allocation preserves every phone and the exact requested length.
            quotas=durations.float()/durations.sum()*(limit-durations.numel())
            durations=1+quotas.floor().long()
            remainder=limit-int(durations.sum())
            if remainder:durations[(quotas-quotas.floor()).argsort(descending=True)[:remainder]]+=1
        path=torch.repeat_interleave(torch.arange(len(durations),device=durations.device),durations)
        expanded,_=self.expand(encoded,path)
        predicted,pitch=self.decode_hidden(expanded,pad_to_multiple)
        return predicted,pitch,total

    def predict(self,text,frames=None,pad_to_multiple=1):
        features,_,total=self.predict_acoustics(text,frames,pad_to_multiple)
        return features,total

    def speech_loss(self,text,features,pitch_targets=None,guidance=.1,alignment_path=None):
        encoded,logdur,prior=self.encode_text(text)
        target=features.to(prior)
        path=self.alignment(prior,target,guidance) if alignment_path is None else alignment_path.to(prior.device)
        if (path.ndim!=1 or path.dtype!=torch.long or path.numel()!=target.shape[1] or path.numel()==0
            or path[0]!=0 or path[-1]!=encoded.shape[1]-1 or ((path.diff()<0)|(path.diff()>1)).any()):
            raise ValueError('Invalid cached target alignment')
        expanded,durations=self.expand(encoded,path)
        predicted,pitch=self.decode_hidden(expanded,32)
        spectral=F.l1_loss(predicted,target)
        energy=F.l1_loss(self.codec.log_rms(predicted),self.codec.log_rms(target))
        delta=F.l1_loss(predicted.diff(dim=1),target.diff(dim=1))
        prior_loss=F.mse_loss(prior[:,path],target)
        duration_loss=F.mse_loss(logdur,durations.float().clamp_min(1).log()[None])
        effective_duration=logdur.exp().clamp_min(2.) if self.config.clock_mode=='soft' else logdur.exp()
        duration_loss=duration_loss+.5*(effective_duration.sum(-1).log()-math.log(target.shape[1])).square().mean()
        pitch_loss=predicted.new_zeros(())
        if pitch_targets is not None:
            f0,voice=(x.to(pitch) for x in pitch_targets)
            pitch_loss=((pitch[...,0]-f0).abs()*voice).sum()/voice.sum().clamp_min(1)+F.binary_cross_entropy_with_logits(pitch[...,1],voice)
        loss=spectral+.4*delta+.4*prior_loss+.2*duration_loss+pitch_loss+.2*energy
        free_loss=predicted.new_zeros(())
        if self.config.clock_mode=='soft':
            free_features,free_pitch=self.decode_hidden(self.expand_soft(encoded,logdur,target.shape[1]),32)
            free_loss=F.l1_loss(free_features,target)+.3*F.l1_loss(free_features.diff(dim=1),target.diff(dim=1))
            free_loss=free_loss+.2*F.l1_loss(self.codec.log_rms(free_features),self.codec.log_rms(target))
            if pitch_targets is not None:
                free_loss=free_loss+.5*(((free_pitch[...,0]-f0).abs()*voice).sum()/voice.sum().clamp_min(1)+F.binary_cross_entropy_with_logits(free_pitch[...,1],voice))
            loss=loss+.7*free_loss
        return {'loss':loss,'spectral':spectral,'energy':energy,'delta':delta,'prior':prior_loss,'duration':duration_loss,'pitch':pitch_loss,'free_clock':free_loss,'path':path.detach()}

    @torch.inference_mode()
    def stream(self,text,max_frames=500,temperature=0.,reference_wav=None,allow_untrained=False,clock_mode='hard',sharpen_formants=0.0):
        if temperature!=0:raise ValueError('Aligned synthesis uses temperature 0')
        if int(self.speech_steps)==0 and not allow_untrained:raise ValueError('Model not trained')
        if type(max_frames)!=int or max_frames<2:raise ValueError('At least two acoustic frames required')
        ref_mean, ref_std, ref_median_f0 = None, None, None
        if reference_wav is not None:
            if not isinstance(reference_wav, torch.Tensor) or reference_wav.numel() < 1600 or not torch.isfinite(reference_wav).all():
                raise ValueError('Invalid reference audio')
            ref_wav = reference_wav.float().reshape(1, -1)
            ref_spec = self.codec.spectrum(ref_wav).abs().clamp_min(1e-5).log().T
            p_ref, v_ref = pitch_labels(ref_wav, self.config.n_fft, self.config.hop_length, self.config.sample_rate)
            min_frames = min(ref_spec.shape[0], v_ref.shape[1])
            ref_spec_sub = ref_spec[:min_frames]
            v_sub = v_ref[0, :min_frames]
            energy = ref_spec_sub.mean(-1)
            speech_mask = (v_sub > 0.4) | (energy > -5.0)
            speech_spec = ref_spec_sub[speech_mask] if speech_mask.sum() > 8 else ref_spec_sub
            ref_mean = speech_spec.mean(0)
            ref_std = speech_spec.std(0).clamp_min(0.25)
            f0_ref = (p_ref * 0.5 + math.log(200.)).exp()
            ref_median_f0 = float(f0_ref[v_ref > 0.5].median()) if (v_ref > 0.5).any() else 200.0
        features,pitch,_=self.predict_acoustics(text,clock_mode=clock_mode)
        if features.shape[1]>max_frames:features,pitch=features[:,:max_frames],pitch[:,:max_frames]
        if ref_median_f0 is not None:
            base_f0 = (pitch[..., 0].clamp(-2.1, 1.4) * 0.5 + math.log(200.)).exp()
            voicing = (2 * pitch[..., 1].sigmoid() - 1).clamp_min(0)
            voiced = (voicing[0] > 0.2)
            base_median_f0 = float(base_f0[0][voiced].median().detach()) if voiced.any() else 200.0
            cloned_f0 = base_f0 * (ref_median_f0 / base_median_f0)
            # Remove base comb filter so vocal tract formants are pure and match target pitch
            frequencies = torch.linspace(0, self.config.sample_rate / 2, features.shape[-1], device=features.device)
            comb_base = (.08 + .92 * torch.exp(-8 * torch.sin(math.pi * frequencies[None, None] / base_f0[..., None]).square())).log()
            adj_base = comb_base * voicing[..., None] / self.codec.std.to(features.device).clamp_min(0.3)
            features = features - adj_base
            # Natural cadence: declarative sentences have gentle fall, questions have gentle rising pitch
            if text.rstrip().endswith(('.', '!')) and features.shape[1] > 12:
                fall_len = min(15, features.shape[1] // 4)
                fall = torch.linspace(1.0, 0.88, fall_len, device=cloned_f0.device)
                cloned_f0[..., -fall_len:] *= fall
            elif text.rstrip().endswith(('?', '¿')) and features.shape[1] > 12:
                rise_len = min(15, features.shape[1] // 4)
                rise = torch.linspace(1.0, 1.14, rise_len, device=cloned_f0.device)
                cloned_f0[..., -rise_len:] *= rise
            pitch = pitch.clone()
            pitch[..., 0] = ((cloned_f0.log() - math.log(200.)) / 0.5).clamp(-2.1, 1.4)
            # Vocal Tract Length Normalization (VTLN): warp formants to match speaker vocal tract
            alpha = (ref_median_f0 / base_median_f0) ** (1/3)
            alpha = max(0.82, min(1.15, alpha))
            D = features.shape[-1]
            src_bins = (torch.arange(D, device=features.device).float() / alpha).clamp(0, D - 1)
            left = src_bins.floor().long().clamp(0, D - 2)
            frac = src_bins - left
            features = features[..., left] * (1 - frac) + features[..., left + 1] * frac
        if sharpen_formants>0:
            pad=7
            smooth=F.avg_pool1d(F.pad(features,(pad,pad),mode='replicate'),15,stride=1)
            features=features+sharpen_formants*(features-smooth)
        wave=self.codec.decode_source_filter(features.clamp(-6,6),pitch,mean=ref_mean,std=ref_std)
        yield from wave.split(self.config.hop_length,dim=-1)

    def count_parameters(self):return sum(p.numel() for p in self.parameters())

    def save(self,path,fp16=False):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        state={k:(v.detach().cpu().half() if fp16 and v.is_floating_point() and not k.startswith('codec.') else v.detach().cpu())
            for k,v in self.state_dict().items() if k!='backbone.text_head.weight'}
        save_atomic({'format_version':3,'model_kind':self.model_kind,'external_weights':False,'config':asdict(self.config),'state':state,
                    'phoneme_symbols':self.phonemizer.symbols,'dictionary_sha256':self.phonemizer.dictionary_sha256},path)
        return path.stat().st_size

    @classmethod
    def from_checkpoint(cls,checkpoint,device='cpu'):
        if checkpoint.get('format_version')!=3 or checkpoint.get('external_weights') is not False or checkpoint.get('model_kind')!=cls.model_kind:raise ValueError('Invalid native aligned checkpoint')
        model=cls(AlignedConfig(**checkpoint['config']))
        if checkpoint.get('phoneme_symbols')!=model.phonemizer.symbols or checkpoint.get('dictionary_sha256')!=model.phonemizer.dictionary_sha256:raise ValueError('Pronunciation data changed; phoneme IDs must remain stable')
        state=dict(checkpoint['state']);state['backbone.text_head.weight']=state['backbone.text_embedding.weight']
        model.to(dtype=state['spectrum_out.weight'].dtype);model.codec.float();model.load_state_dict(state,strict=True)
        return model.to(device).eval()
