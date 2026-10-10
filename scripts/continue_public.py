"""Continue the public-data model with frozen codec and validation-based selection."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.voice import read_audio
from scripts.train_public import rows, speech_validation
from src.models.voice_unified import UnifiedVoiceModel
from src.data.video_audio import file_sha256


def run(args):
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS GPU unavailable; use --device cpu or GPU-enabled execution')
    torch.set_num_threads(4)
    checkpoint = Path('models/checkpoints/voice_unified.pt')
    sidecar = json.loads(Path(str(checkpoint)+'.training.json').read_text())
    digest = file_sha256(checkpoint)
    if sidecar.get('user_audio_used') is not False or sidecar.get('external_weights') is not False:
        raise ValueError('Only the model trained from public data may be continued here')
    if sidecar.get('checkpoint_sha256') != digest:
        raise ValueError('Training metadata does not match checkpoint')
    model = UnifiedVoiceModel.load(checkpoint, args.device).float()
    if getattr(model, 'model_kind', None) in ('spectral_continuous','phoneme_aligned_spectrum'):
        raise ValueError('Use scripts/train_spectral.py or train_aligned.py for the matching acoustic model')
    if int(model.speech_steps) < 1 or int(model.codec_steps) < 1:
        raise ValueError('Initial public codec and paired training must finish first')
    starting_step = int(model.speech_steps)
    torch.manual_seed(97 + starting_step)
    root = Path('data/public/ljspeech_v1')
    train = [json.loads(line) for line in (root/'train.jsonl').read_text().splitlines() if line.strip()]
    random.Random(97 + starting_step).shuffle(train)
    train = train[:args.clips]
    valid = rows(root/'validation.jsonl', 16)
    test = rows(root/'test.jsonl', 16)
    if not train or not valid or not test:
        raise ValueError('Nonempty, separate real paired splits required')
    for row in train + valid + test:
        if row.get('dataset') != 'LJSpeech-1.1' or row.get('source') != 'human_recording':
            raise ValueError('Only public human-recorded LJSpeech pairs are accepted')
        if not Path(row['audio']).resolve().is_relative_to(Path('data/public/LJSpeech-1.1/wavs').resolve()):
            raise ValueError('Recording lies outside the public dataset')
    cache = {}
    def pair(row):
        if row['id'] not in cache:
            with torch.no_grad():
                wav = read_audio(row['audio'], model.config.sample_rate).to(args.device)
                _, tokens, _ = model.codec.encode(wav)
            if tokens.shape[-1] + len(model.tokenizer.encode(row['text'])) + 1 > model.config.max_seq_len:
                raise ValueError('Complete pair exceeds context; do not truncate its audio')
            cache[row['id']] = tokens.cpu()
        return row, cache[row['id']]
    valid_pairs = [pair(row) for row in valid]
    @torch.no_grad()
    def validation(current):
        metrics = speech_validation(current, valid_pairs)
        if args.rollout_selection:
            errors=[]
            for row,tokens in valid_pairs[:8]:
                tokens=tokens.to(args.device)
                predicted=current.predicted_training_tokens(row['text'],tokens)
                target=current.codec.decode(tokens)
                reconstructed=current.codec.decode(predicted)
                errors.append(float((target-reconstructed).square().mean()/target.square().mean().clamp_min(1e-9)))
            metrics['rollout_wave_normalized_mse']=sum(errors)/len(errors)
        return metrics
    before = validation(model)
    for name, p in model.named_parameters():
        p.requires_grad_(not name.startswith('codec.'))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    optimizer_path = Path('models/checkpoints/public_optimizer.pt')
    optimizer_resumed = False
    if optimizer_path.exists():
        state = torch.load(optimizer_path, map_location='cpu', weights_only=True)
        if state.get('checkpoint_sha256') != digest or state.get('step') != starting_step:
            raise ValueError('Optimizer does not match public checkpoint; no silent reset')
        optimizer.load_state_dict(state['optimizer'])
        for group in optimizer.param_groups:
            group['lr'] = args.lr
        optimizer_resumed = True
    candidate = checkpoint.with_name('voice_public_candidate.pt')
    candidate_optimizer = optimizer_path.with_name('public_optimizer_candidate.pt')
    selection_metric = 'rollout_wave_normalized_mse' if args.rollout_selection else 'loss'
    best_loss, best_step = before[selection_metric], starting_step
    logs = []
    model.train()
    for step in range(args.steps):
        row, tokens = pair(train[step % len(train)])
        tokens = tokens.to(args.device)
        input_tokens = None
        if args.self_conditioning > 0:
            predicted = model.predicted_training_tokens(row['text'], tokens,
                             pad_to_multiple=128 if args.device=='mps' else 1)
            mask = torch.rand((1, 1, tokens.shape[-1]), device=args.device) < args.self_conditioning
            input_tokens = torch.where(mask, predicted, tokens)
        optimizer.zero_grad(set_to_none=True)
        loss = model.speech_loss(row['text'], tokens=tokens, input_tokens=input_tokens,
                                 coarse_weight=args.coarse_weight,
                                 pad_to_multiple=128 if args.device=='mps' else 1)['loss']
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite paired objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        model.speech_steps.add_(1)
        if (step+1) % 10 == 0:
            print('Public speech', int(model.speech_steps), 'loss', float(loss.detach()), flush=True)
        if (step+1) % 100 == 0 or step+1 == args.steps:
            metrics = validation(model)
            logs.append({'step':int(model.speech_steps), **metrics})
            print('Validation', metrics, flush=True)
            if metrics[selection_metric] < best_loss:
                best_loss, best_step = metrics[selection_metric], int(model.speech_steps)
                model.save(candidate, fp16=True)
                torch.save({'step':best_step, 'checkpoint_sha256':file_sha256(candidate),
                            'optimizer':optimizer.state_dict()}, candidate_optimizer)
        del loss
        if input_tokens is not None:
            del input_tokens, predicted, mask
        if args.device == 'mps' and (step+1) % 10 == 0:
            torch.mps.synchronize()
            torch.mps.empty_cache()
    accepted = False
    after = before
    if best_step > starting_step:
        stored = UnifiedVoiceModel.load(candidate, args.device).float()
        after = validation(stored)
        accepted = after[selection_metric] < before[selection_metric]
    else:
        stored = UnifiedVoiceModel.load(checkpoint, args.device).float()
    if not accepted:
        stored = UnifiedVoiceModel.load(checkpoint, args.device).float()
        after = validation(stored)
    test_metrics = speech_validation(stored, [pair(row) for row in test])
    if accepted:
        candidate.replace(checkpoint)
        candidate_optimizer.replace(optimizer_path)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report_path = Path('reports/training') / f'public_speech_{run_id}.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {'fresh_initialization':False, 'user_audio_used':False, 'external_weights':False,
              'input_checkpoint_sha256':digest, 'checkpoint_sha256':file_sha256(checkpoint),
              'optimizer_resumed':optimizer_resumed, 'device':args.device,
              'starting_speech_steps':starting_step, 'steps_attempted':args.steps,
              'depth_hidden':model.config.depth_hidden,
              'self_conditioning_probability':args.self_conditioning,
              'coarse_weight':args.coarse_weight,'selection_metric':selection_metric,
              'selected_speech_steps':int(stored.speech_steps), 'accepted':accepted,
              'training_ids':[r['id'] for r in train[:min(args.steps,len(train))]],
              'selected_training_ids':[r['id'] for r in train[:min(max(0,int(stored.speech_steps)-starting_step),len(train))]] if accepted else [],
              'validation_ids':[r['id'] for r in valid], 'test_ids':[r['id'] for r in test],
              'speech_before':before, 'speech_after':after, 'validation':logs,
              'test_teacher_forcing':test_metrics, 'free_generation_quality_verified':False}
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    if accepted:
        sidecar.update(checkpoint_sha256=file_sha256(checkpoint), speech_steps=int(stored.speech_steps),
                       speech_ready=False,intelligibility_verified=False,
                       training_report=str(report_path))
        if stored.config.depth_hidden:
            origin=sidecar.get('depth_started_at_speech_step',starting_step)
            sidecar.update(depth_started_at_speech_step=origin,
                           depth_training_steps=int(stored.speech_steps)-origin)
        Path(str(checkpoint)+'.training.json').write_text(json.dumps(sidecar, indent=2)+'\n')
    print('Published', accepted, 'Report', report_path, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu','mps'], default='cpu')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--clips', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--self-conditioning', type=float, default=0.,
                        help='Fraction of history frames replaced by causal model predictions')
    parser.add_argument('--coarse-weight',type=float,default=1.)
    parser.add_argument('--rollout-selection',action='store_true',help='Select by actual predicted-code waveform error on 8 validation clips')
    args = parser.parse_args()
    if min(args.steps,args.clips) < 1 or not 0 < args.lr <= .01 or not 0 <= args.self_conditioning <= .5 or not 1 <= args.coarse_weight <= 10:
        parser.error('Positive steps/clips and learning rate <= .01 required')
    run(args)
