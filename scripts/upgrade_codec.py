"""Train a higher bitrate native codec, and publish only a verified improvement."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.voice import read_audio, training_crop
from scripts.improve_codec import evaluate, initialize_codebooks
from src.models.voice_unified import UnifiedVoiceModel
from src.data.video_audio import file_sha256


def save_training_metadata(checkpoint, report_path, report):
    Path(str(checkpoint)+'.training.json').write_text(json.dumps({
        'stage': 'codec', 'architecture_changed': True,
        'codec_steps': report['selected_step'], 'speech_steps': 0,
        'training_report': str(report_path), 'checkpoint_sha256': file_sha256(checkpoint),
        'data_provenance': report.get('data_provenance', 'data/training/video_native_v1/provenance.json'),
        'external_weights': False,
        'note': 'Short-frame codec initialized locally; its counter starts at zero. Earlier codec histories are in reports/.',
    }, indent=2)+'\n')


def initialize_short_frames(model):
    """Identity linear projections for short frames; no pretrained weights."""
    c = model.config
    if c.codec_dim < c.hop_length:
        raise ValueError('Identity initialization needs codec_dim >= hop_length')
    with torch.no_grad():
        for p in model.codec.parameters():
            p.zero_()
        model.codec.encoder_in.weight[:c.hop_length].copy_(torch.eye(c.hop_length))
        model.codec.decoder_out.weight[:, :c.hop_length].copy_(torch.eye(c.hop_length))


def run(steps=1000, codebooks=24, gain_bins=256, data='data/training/video_native_v1', validation_data=None):
    torch.set_num_threads(4)
    torch.manual_seed(71)
    generator = torch.Generator().manual_seed(71)
    checkpoint = Path('models/checkpoints/voice_unified.pt')
    old = UnifiedVoiceModel.load(checkpoint).float()
    if int(old.speech_steps)+int(old.audio_steps):
        raise ValueError('Changing codec requires retraining acoustic/speech tokens')
    model = UnifiedVoiceModel(replace(old.config, hop_length=80, codebooks=codebooks, gain_bins=gain_bins)).eval()
    # Reuse only our own compatible language weights. New codec is initialized here.
    old_state = old.state_dict()
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if not name.startswith('codec.') and name in old_state and value.shape == old_state[name].shape:
                value.copy_(old_state[name])
        model.codec_steps.zero_()
    initialize_short_frames(model)
    root = Path(data)
    validation_root = Path(validation_data or data)
    train = [read_audio(p, model.config.sample_rate) for p in sorted((root/'train').glob('*.wav'))]
    valid = [read_audio(p, model.config.sample_rate)[:, :model.config.sample_rate]
             for p in sorted((validation_root/'validation').glob('*.wav'))]
    if not train or not valid:
        raise ValueError('Separate train and validation audio required')
    before = evaluate(old, valid)
    full_valid = [read_audio(p, model.config.sample_rate) for p in sorted((validation_root/'validation').glob('*.wav'))]
    full_before = evaluate(old, full_valid)
    print('Previous:', before, flush=True)
    initialize_codebooks(model, train, generator)
    # Zero is a reserved centroid at every stage: exact silence remains encodable.
    with torch.no_grad():
        for q in model.codec.rvq.quantizers:
            q.embedding.weight[0].zero_()
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith('codec.decoder'))
    initial = evaluate(model, valid)
    print('Short frames, before fitting:', initial, flush=True)
    best = copy.deepcopy(model.codec.state_dict())
    best_score = initial['normalized_mse'] + .1*initial['boundary_error']
    best_step = 0
    logs = []
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    for step in range(steps):
        index = int(torch.randint(len(train), (), generator=generator))
        wav = training_crop(train[index], model.config.sample_rate, generator)
        if step % 10 == 0:
            wav = torch.zeros_like(wav)
        optimizer.zero_grad(set_to_none=True)
        loss = model.codec_loss(wav)['loss']
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite codec loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.codec.parameters(), 1.)
        optimizer.step()
        model.codec_steps.add_(1)
        if (step+1) % 100 == 0:
            metrics = evaluate(model, valid)
            logs.append({'step': step+1, **metrics})
            print(step+1, metrics, flush=True)
            score = metrics['normalized_mse'] + .1*metrics['boundary_error']
            if score < best_score and metrics['silence_rms'] <= .001:
                best_score, best_step = score, step+1
                best = copy.deepcopy(model.codec.state_dict())
    model.codec.load_state_dict(best)
    model.codec_steps.fill_(best_step)
    # Compare the actually stored FP16 candidate, not only FP32 training weights.
    candidate = checkpoint.with_name('voice_candidate.pt')
    model.save(candidate, fp16=True)
    stored = UnifiedVoiceModel.load(candidate).float()
    after = evaluate(stored, valid)
    full_after = evaluate(stored, full_valid)
    quality_gain = after['normalized_mse'] < before['normalized_mse']*.7
    silence_fix = (before['silence_rms'] > 1e-6 and after['silence_rms'] == 0
                   and after['normalized_mse'] <= before['normalized_mse']
                   and after['boundary_error'] <= before['boundary_error'])
    accepted = ((quality_gain or silence_fix) and after['correlation'] >= before['correlation']
                and after['silence_rms'] <= .001
                and full_after['normalized_mse'] <= full_before['normalized_mse'])
    if accepted:
        candidate.replace(checkpoint)
    else:
        candidate.unlink()
    report = {'before': before, 'initial': initial, 'after': after, 'accepted': accepted,
              'full_before': full_before, 'full_after': full_after,
              'publication_reason': 'silence_removed_without_fidelity_regression' if silence_fix else 'large_fidelity_improvement',
              'selected_step': best_step, 'validation': logs,
              'hop_samples': 80, 'codebooks': codebooks, 'bits_per_second': codebooks*1600, 'gain_bins': gain_bins,
              'external_weights': False, 'human_listening_verified': False,
              'scope': 'Reconstruction, chronologically held-out same-video audio; not text-to-speech',
              'parameters': model.count_parameters(), 'checkpoint_bytes': checkpoint.stat().st_size,
              'training_clips': len(train), 'data_provenance': str(root/'provenance.json'),
              'validation_data': str(validation_root)}
    report_path = Path(f'reports/codec_short_frames_{codebooks}_gain{gain_bins}_{root.name}.json')
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    if accepted:
        save_training_metadata(checkpoint, report_path, report)
    print('Published:', accepted, 'Stored:', after, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--codebooks', type=int, choices=[8, 12, 16, 24], default=24)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--gain-bins', type=int, choices=[1, 256], default=256)
    parser.add_argument('--data', default='data/training/video_native_v1')
    parser.add_argument('--validation-data')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('Steps must be positive')
    run(args.steps, args.codebooks, args.gain_bins, args.data, args.validation_data)
