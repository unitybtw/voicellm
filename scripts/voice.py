"""Canonical local-only CLI for the one VoiceLLM model."""
import argparse
from datetime import datetime, timezone
import copy
import json
import math
from pathlib import Path
import platform
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from src.models.voice_unified import UnifiedVoiceModel, UnifiedConfig
from src.data.video_audio import prepare_video, file_sha256


def read_audio(path, sr):
    audio, source_sr = sf.read(path, dtype='float32', always_2d=True)
    audio = audio.mean(1)
    if len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError(f'Invalid audio: {path}')
    divisor = math.gcd(sr, source_sr)
    if source_sr != sr:
        audio = resample_poly(audio, sr // divisor, source_sr // divisor)
    return torch.from_numpy(audio.copy()).unsqueeze(0)


def synchronize(device):
    if device == 'mps':
        torch.mps.synchronize()
    elif device.startswith('cuda'):
        torch.cuda.synchronize()


def benchmark(model, device, output, iterations):
    model.eval()
    text = 'Hello, I am VoiceLLM.'
    samples = []
    for iteration in range(iterations + 2):
        synchronize(device)
        start = time.perf_counter()
        latencies = []
        generated_samples = 0
        for index, chunk in enumerate(model.stream(text, max_frames=25, allow_untrained=True)):
            chunk.cpu()  # Include host transfer in end-to-end time.
            synchronize(device)
            now = time.perf_counter()
            latencies.append((now - start) * 1000)
            generated_samples += chunk.shape[-1]
        if iteration >= 2:
            samples.append({'ttfa_ms': latencies[0], 'total_ms': latencies[-1], 'audio_ms': generated_samples/model.config.sample_rate*1000})
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_bytes = model.save(Path(directory)/'voice.pt', fp16=True)
    c = model.config
    metrics = {
        'model': 'VoiceLLM Unified', 'device': device, 'platform': platform.platform(),
        'torch': torch.__version__, 'parameters': model.count_parameters(),
        'actual_fp16_checkpoint_mib': checkpoint_bytes / 1024**2,
        'loaded_parameter_mib': sum(p.numel()*p.element_size() for p in model.parameters())/1024**2,
        'config': c.__dict__, 'iterations': iterations, 'warmups': 2,
        'codec_steps': model.codec_steps.item(), 'speech_steps': model.speech_steps.item(),
        'audio_steps': model.audio_steps.item(),
        'ttfa_median_ms': float(np.median([v['ttfa_ms'] for v in samples])),
        'ttfa_p95_ms': float(np.percentile([v['ttfa_ms'] for v in samples], 95)),
        'rtf_median': float(np.median([v['total_ms'] / v['audio_ms'] for v in samples])),
        'quality_evaluated': False, 'allow_untrained': True,
        'notes': 'Compute timing only; no intelligibility evaluation. Spectral models use block phase reconstruction before the first chunk. RTF uses actual output length; lower is better.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def codec_validation(model, recordings):
    if not recordings:
        return None
    previous_mode = model.training
    model.eval()
    values = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for audio in recordings:
            values.append(float(model.codec_loss(audio.to(device))['reconstruction']))
    model.train(previous_mode)
    return sum(values)/len(values)


def training_crop(audio, length, generator):
    if audio.shape[-1] < length:
        return torch.nn.functional.pad(audio, (0, length-audio.shape[-1]))
    offset = int(torch.randint(audio.shape[-1]-length+1, (), generator=generator))
    return audio[:, offset:offset+length]


def train(model, args):
    if getattr(model, 'model_kind', None) in ('spectral_continuous','phoneme_aligned_spectrum'):
        script = 'train_aligned.py' if model.model_kind == 'phoneme_aligned_spectrum' else 'train_spectral.py'
        raise ValueError(f'Use scripts/{script} for this model training/resume')
    stage = args.stage
    starting_step = int(getattr(model, stage + '_steps'))
    input_checkpoint_sha256 = file_sha256(args.checkpoint) if args.checkpoint else None
    generator = torch.Generator().manual_seed(args.seed + starting_step)
    if stage == 'codec' and model.speech_steps.item()+model.audio_steps.item() > 0:
        raise ValueError('Codec updates would invalidate trained acoustic/speech tokens; use a fresh checkpoint for codec changes')
    if stage == 'speech' and model.codec_steps.item() == 0:
        raise ValueError('Train the codec on real recordings before speech training')
    records = []
    if stage == 'codec':
        if not args.audio_dir:
            raise ValueError('--audio-dir is required for codec training')
        paths = sorted(Path(args.audio_dir).glob('*.wav'))
        if not paths:
            raise ValueError('No WAV recordings found')
        records = [{'audio': str(path)} for path in paths]
    elif stage == 'text':
        if not args.text_file:
            raise ValueError('--text-file is required for language pretraining')
        records = [{'text': line, 'audio': args.text_file} for line in Path(args.text_file).read_text().splitlines() if line.strip()]
        if not records:
            raise ValueError('Text corpus is empty')
    else:
        if not args.manifest:
            raise ValueError('--manifest is required for speech training')
        manifest = Path(args.manifest).resolve()
        for line in manifest.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get('source') != 'human_recording' or not row.get('text', '').strip():
                    raise ValueError('Speech records require text and source=human_recording')
                row['audio'] = str(manifest.parent / row['audio'])
                records.append(row)
        if not records:
            raise ValueError('Manifest is empty')
    waves = [(row, None if stage == 'text' else read_audio(row['audio'], model.config.sample_rate)) for row in records]
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('codec.') if stage == 'codec' else not name.startswith('codec.'))
    model.float().train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    validation = []
    if args.validation_dir:
        if stage != 'codec':
            raise ValueError('Validation WAV directory currently supports codec stage only')
        validation_paths = sorted(Path(args.validation_dir).glob('*.wav'))
        if not validation_paths:
            raise ValueError('No validation WAV files')
        training_paths = {Path(row['audio']).resolve() for row in records}
        if any(path.resolve() in training_paths for path in validation_paths):
            raise ValueError('Training and validation files must be separate')
        validation = [read_audio(path, model.config.sample_rate)[:, :model.config.sample_rate] for path in validation_paths]
    before_validation = codec_validation(model, validation)
    best_validation = before_validation
    best_codec = copy.deepcopy(model.codec.state_dict()) if validation else None
    best_step = starting_step
    logs = []
    validation_history = []
    for step in range(args.steps):
        row, audio = waves[step % len(waves)]
        if stage == 'codec':
            audio = training_crop(audio, model.config.sample_rate, generator)
        elif stage == 'speech' and len(model.tokenizer.encode(row['text'])) + math.ceil(audio.shape[-1]/model.config.hop_length) + 1 > model.config.max_seq_len:
            raise ValueError(f"Pair too long: {row['audio']}. Segment audio and transcript together.")
        if audio is not None:
            audio = audio.to(args.device)
        optimizer.zero_grad(set_to_none=True)
        if stage == 'codec':
            losses = model.codec_loss(audio)
        elif stage == 'text':
            losses = model.text_loss(row['text'])
        else:
            losses = model.speech_loss(row['text'], audio)
        if not torch.isfinite(losses['loss']):
            raise FloatingPointError('Non-finite loss')
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        counter = getattr(model, stage + '_steps')
        counter.add_(1)
        logs.append({key: float(value.detach()) for key, value in losses.items()})
        if step == 0 or (step+1) % 100 == 0:
            print(f'{stage} step {step+1}: {logs[-1]}', flush=True)
        if validation and ((step+1) % 100 == 0 or step+1 == args.steps):
            value = codec_validation(model, validation)
            validation_history.append({'step': int(model.codec_steps), 'reconstruction': value})
            print(f'validation at step {int(model.codec_steps)}: {value:.6f}', flush=True)
            if value < best_validation:
                best_validation = value
                best_codec = copy.deepcopy(model.codec.state_dict())
                best_step = int(model.codec_steps)
    if validation:
        model.codec.load_state_dict(best_codec)
        model.codec_steps.fill_(best_step)
    model.save(args.output, fp16=True)
    report = {
        'stage': stage, 'steps_this_run': args.steps, 'recordings': [row['audio'] for row in records],
        'external_weights': False, 'seed': args.seed, 'losses': logs,
        'quality_validated': False, 'optimizer_saved': False,
        'starting_step': starting_step, 'selected_checkpoint_step': int(getattr(model, stage + '_steps')),
        'input_checkpoint_sha256': input_checkpoint_sha256,
        'validation_before': before_validation, 'validation_best': best_validation,
        'validation_history': validation_history,
        'validation_recordings': [str(path.resolve()) for path in validation_paths] if validation else [],
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False)+'\n'
    Path(str(args.output)+'.training.json').write_text(serialized)
    history_dir = Path('reports/training'); history_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    (history_dir/f'{stage}_{run_id}.json').write_text(serialized)
    print(f'Saved one native checkpoint: {args.output}')


def main():
    parser = argparse.ArgumentParser(description='Single native VoiceLLM; no remote model loaders')
    parser.add_argument('command', choices=['benchmark', 'train', 'speak', 'narrate', 'export', 'prepare-video', 'reconstruct'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp16')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--stage', choices=['codec', 'speech', 'text'], default='codec')
    parser.add_argument('--audio-dir')
    parser.add_argument('--validation-dir')
    parser.add_argument('--video', type=Path)
    parser.add_argument('--segment-seconds', type=float, default=4.)
    parser.add_argument('--audio', type=Path)
    parser.add_argument('--manifest')
    parser.add_argument('--text-file')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--text')
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=0., help='0 is deterministic; positive values sample from top 8 acoustic symbols')
    parser.add_argument('--max-seconds', type=float, default=8.)
    args = parser.parse_args()
    if args.steps < 1 or args.iterations < 1 or args.lr <= 0:
        parser.error('steps, iterations and learning rate must be positive')
    if args.command == 'prepare-video':
        if not args.video or not args.output:
            parser.error('prepare-video requires --video and --output directory')
        report = prepare_video(args.video, args.output, seconds=args.segment_seconds)
        print(f"Prepared {len(report['segments'])} segments from {report['duration_seconds']:.2f}s video")
        return
    torch.manual_seed(args.seed)
    if args.command in ['speak', 'narrate', 'export', 'reconstruct'] and not args.checkpoint:
        parser.error('speak requires --checkpoint with speech training')
    if args.config and args.checkpoint:
        parser.error('Checkpoint already contains its configuration; omit --config')
    config = UnifiedConfig(**json.loads(args.config.read_text())) if args.config else UnifiedConfig()
    model = UnifiedVoiceModel.load(args.checkpoint, args.device) if args.checkpoint else UnifiedVoiceModel(config).to(args.device)
    if args.command == 'reconstruct':
        if not args.audio:
            parser.error('reconstruct requires --audio')
        wav = read_audio(args.audio, model.config.sample_rate).to(args.device)
        with torch.inference_mode():
            _, tokens, _ = model.codec.encode(wav)
            result = model.codec.decode(tokens)[..., :wav.shape[-1]].float().cpu().flatten().numpy()
        output = args.output or Path('data/native_samples/reconstruction.wav')
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, result, model.config.sample_rate)
        print(f'Reconstructed input recording: {output}. This is not new-text synthesis.')
    elif args.command == 'export':
        output = args.output or Path('models/exported/voice_unified.pt')
        size = model.save(output, fp16=args.precision == 'fp16')
        print(f'Native single checkpoint: {output} ({size/1024**2:.2f} MiB)')
    elif args.command == 'benchmark':
        model = model.half() if args.precision == 'fp16' else model.float()
        if getattr(model, 'model_kind', None) in ('spectral_continuous','phoneme_aligned_spectrum'):
            model.codec.float()  # Preserve DSP filters/statistics in their native precision.
        benchmark(model, args.device, args.output or Path('reports/unified_benchmark.json'), args.iterations)
    elif args.command == 'train':
        args.output = args.output or Path('models/checkpoints/voice_unified.pt')
        train(model, args)
    elif args.command == 'narrate':
        if not args.text_file:
            parser.error('narrate requires --text-file with one sentence per line')
        if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
            parser.error('max-seconds must be finite and positive')
        lines = [line.strip() for line in Path(args.text_file).read_text().splitlines() if line.strip()]
        if not lines:
            raise ValueError('Story text is empty')
        narration = []
        for line in lines:
            frames = min(math.ceil(args.max_seconds*model.config.sample_rate/model.config.hop_length),
                         model.config.max_seq_len-len(model.tokenizer.encode(line))-1)
            if frames <= 0:
                raise ValueError('Narration sentence exceeds context; split the sentence')
            chunks = list(model.stream(line, max_frames=frames, temperature=args.temperature))
            if not chunks:
                raise RuntimeError('No narration generated')
            if narration:
                narration.append(chunks[0].new_zeros((*chunks[0].shape[:-1], round(model.config.sample_rate*.2))))
            narration.extend(chunks)
        result = torch.cat(narration, -1).flatten().float().cpu().numpy()
        output = args.output or Path('data/native_samples/story.wav')
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, result, model.config.sample_rate)
        print(f'Saved narration: {output}; intelligibility still requires listening evaluation')
    else:
        if not args.text:
            parser.error('speak requires --text')
        if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
            parser.error('max-seconds must be finite and positive')
        frames = args.max_frames if args.max_frames is not None else min(
            math.ceil(args.max_seconds*model.config.sample_rate/model.config.hop_length),
            model.config.max_seq_len-len(model.tokenizer.encode(args.text))-1)
        if frames <= 0:
            raise ValueError('Speech text or output duration exceeds context')
        chunks = list(model.stream(args.text, max_frames=frames, temperature=args.temperature))
        if not chunks:
            raise RuntimeError('No audio generated')
        output = args.output or Path('data/native_output.wav')
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, torch.cat(chunks, -1).flatten().float().cpu().numpy(), model.config.sample_rate)
        print(f'Saved: {output}; training counters do not certify intelligibility or quality')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        raise SystemExit(f'Error: {error}') from None
