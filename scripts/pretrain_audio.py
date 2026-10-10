"""Train the shared native backbone on real audio, without inventing transcripts."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.voice import read_audio
from src.models.voice_unified import UnifiedVoiceModel
from src.data.video_audio import file_sha256


@torch.no_grad()
def audio_validation(model, tokens):
    if not tokens:
        raise ValueError('Held-out tokens required')
    previous_mode = model.training
    model.eval()
    values = [model.audio_loss(tokens=t) for t in tokens]
    model.train(previous_mode)
    return {key: sum(float(row[key]) for row in values)/len(values) for key in values[0]}


def verify_split(training_root, validation_root):
    training = json.loads((training_root/'provenance.json').read_text())
    validation = json.loads((validation_root/'provenance.json').read_text())
    if training['video_sha256'] != validation['video_sha256']:
        raise ValueError('This workflow expects one source video with separate time ranges')
    train = [r for r in training['segments'] if r['split']=='train']
    valid = [r for r in validation['segments'] if r['split']=='validation']
    if not train or not valid or max(r['end_seconds'] for r in train)>min(r['start_seconds'] for r in valid):
        raise ValueError('Training and validation time ranges overlap or are empty')
    return train, valid


def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(83)
    model = UnifiedVoiceModel.load(args.checkpoint).float()
    generator = torch.Generator().manual_seed(83+int(model.audio_steps))
    if int(model.codec_steps)==0 or int(model.speech_steps)>0:
        raise ValueError('Pretrain audio after codec training and before paired speech training')
    train_rows, valid_rows = verify_split(args.data, args.validation_data)
    data_metadata = json.loads((args.data/'provenance.json').read_text())
    input_hash = file_sha256(args.checkpoint)
    codec_state = copy.deepcopy(model.codec.state_dict())
    stop_state = copy.deepcopy(model.stop_head.state_dict())
    def tokenize(rows):
        result = []
        with torch.no_grad():
            for row in rows:
                wav = read_audio(row['audio'], model.config.sample_rate)
                _, tokens, _ = model.codec.encode(wav)
                if tokens.shape[-1]+1 > model.config.max_seq_len:
                    raise ValueError('Audio clip exceeds context; use shorter matched segments')
                result.append(tokens)
        return result
    train = tokenize(train_rows)
    valid = tokenize(valid_rows)
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith('backbone.'))
    before = audio_validation(model, valid)
    starting_step = int(model.audio_steps)
    print('Audio pretraining:', len(train), 'training clips;', len(valid), 'held-out clips', flush=True)
    print('Before:', before, flush=True)
    best_loss = before['loss']
    best_state = copy.deepcopy(model.state_dict())
    logs = []
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    order = []
    for step in range(args.steps):
        if not order:
            order = torch.randperm(len(train), generator=generator).tolist()
        tokens = train[order.pop()]
        optimizer.zero_grad(set_to_none=True)
        loss = model.audio_loss(tokens=tokens)['loss']
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite acoustic loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        model.audio_steps.add_(1)
        if (step+1)%50==0 or step+1==args.steps:
            metrics = audio_validation(model, valid)
            logs.append({'step': int(model.audio_steps), **metrics})
            print('Step', step+1, metrics, flush=True)
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    candidate = args.checkpoint.with_name('voice_audio_candidate.pt')
    model.save(candidate, fp16=True)
    stored = UnifiedVoiceModel.load(candidate).float()
    after = audio_validation(stored, valid)
    codec_unchanged = all(torch.equal(value, stored.codec.state_dict()[key]) for key,value in codec_state.items())
    stop_unchanged = all(torch.equal(value, stored.stop_head.state_dict()[key]) for key,value in stop_state.items())
    accepted = after['loss'] < before['loss'] and codec_unchanged and stop_unchanged
    if accepted:
        candidate.replace(args.checkpoint)
    else:
        candidate.unlink()
    report = {'before': before, 'after': after, 'accepted': accepted,
        'input_checkpoint_sha256': input_hash, 'output_checkpoint_sha256': file_sha256(args.checkpoint),
        'steps_attempted': args.steps, 'starting_audio_steps': starting_step,
        'selected_audio_steps': int(stored.audio_steps), 'codec_steps': int(stored.codec_steps),
        'speech_steps': int(stored.speech_steps), 'train_clips': len(train), 'validation_clips': len(valid),
        'training_provenance': str(args.data/'provenance.json'),
        'validation_provenance': str(args.validation_data/'provenance.json'),
        'codec_unchanged': codec_unchanged, 'stop_head_unchanged': stop_unchanged,
        'external_weights': False, 'new_text_synthesis_verified': False,
        'source_language': data_metadata.get('language', 'unknown'),
        'source_language_evidence': data_metadata.get('language_source', 'not supplied'),
        'note': 'Learns patterns in recorded audio. Does not learn written-word/audio alignment, verify speaker identity or learn utterance-stop labels.',
        'validation': logs, 'optimizer_saved': False}
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report_path = Path('reports/training')/f'audio_{run_id}.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    if accepted:
        Path(str(args.checkpoint)+'.training.json').write_text(json.dumps({
            'stage': 'audio', 'audio_steps': int(stored.audio_steps), 'speech_steps': int(stored.speech_steps),
            'codec_steps': int(stored.codec_steps), 'checkpoint_sha256': file_sha256(args.checkpoint),
            'training_report': str(report_path), 'data_provenance': str(args.data/'provenance.json'),
            'external_weights': False, 'new_text_synthesis_verified': False}, indent=2)+'\n')
    Path('reports/audio_pretraining_latest.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Published:', accepted, 'After:', after, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('models/checkpoints/voice_unified.pt'))
    parser.add_argument('--data', type=Path, default=Path('data/training/video_native_dense_v1'))
    parser.add_argument('--validation-data', type=Path, default=Path('data/training/video_native_v1'))
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    if args.steps < 1 or not 0 < args.lr <= .01:
        parser.error('Positive steps and learning rate <= .01 required')
    run(args)
