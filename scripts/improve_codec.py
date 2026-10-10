"""Rehabilitate the native codec; train waveform fidelity before RVQ initialization."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
import soundfile as sf
from scripts.voice import read_audio, training_crop
from src.models.voice_unified import UnifiedVoiceModel


@torch.no_grad()
def initialize_codebooks(model, waves, generator):
    latents = torch.cat([model.codec.encode_latents(wav[:, :model.config.sample_rate]).transpose(1,2).reshape(-1,model.config.codec_dim)
                         for wav in waves],0)
    # Train-only Lloyd initialization; no pretrained codec or teacher.
    residual = latents
    for quantizer in model.codec.rvq.quantizers:
        count = quantizer.codebook_size
        if len(residual) < count:
            raise ValueError('Codebook initialization needs at least one frame per centroid')
        centers = residual[torch.randperm(len(residual), generator=generator)[:count]].clone()
        for _ in range(12):
            distances = residual.square().sum(-1,keepdim=True) - 2*residual@centers.T + centers.square().sum(-1)[None]
            assignments = distances.argmin(-1)
            sums = torch.zeros_like(centers).index_add_(0,assignments,residual)
            hits = torch.bincount(assignments,minlength=count)
            active = hits>0
            centers[active] = sums[active]/hits[active,None]
        quantizer.embedding.weight.copy_(centers)
        distances = residual.square().sum(-1,keepdim=True)-2*residual@centers.T+centers.square().sum(-1)[None]
        residual = residual-centers[distances.argmin(-1)]


@torch.no_grad()
def evaluate(model, waves):
    if not waves:
        raise ValueError('Evaluation requires audio')
    metrics=[]
    for wav in waves:
        _, tokens, _ = model.codec.encode(wav)
        reconstructed=model.codec.decode(tokens)[:,:wav.shape[-1]]
        mse=(reconstructed-wav).square().mean().item()
        energy=wav.square().mean().item()
        correlation=torch.nn.functional.cosine_similarity(wav,reconstructed).item()
        jumps=(reconstructed[:,model.config.hop_length::model.config.hop_length]-reconstructed[:,model.config.hop_length-1:-1:model.config.hop_length])-(wav[:,model.config.hop_length::model.config.hop_length]-wav[:,model.config.hop_length-1:-1:model.config.hop_length])
        boundary=float(jumps.abs().mean()/wav.square().mean().sqrt().clamp_min(.01)) if jumps.numel() else 0.
        metrics.append({'normalized_mse':mse/max(energy,1e-9),'correlation':correlation,'boundary_error':boundary})
    silence=torch.zeros(1,model.config.sample_rate)
    _,tokens,_=model.codec.encode(silence)
    noise=float(model.codec.decode(tokens).square().mean().sqrt())
    return {'normalized_mse':float(np.mean([m['normalized_mse'] for m in metrics])),
            'correlation':float(np.mean([m['correlation'] for m in metrics])),
            'silence_rms':noise, 'boundary_error':float(np.mean([m['boundary_error'] for m in metrics]))}


def run(args):
    torch.manual_seed(43)
    generator=torch.Generator().manual_seed(43)
    model=UnifiedVoiceModel.load(args.checkpoint).float()
    if model.speech_steps.item()+model.audio_steps.item()>0:
        raise ValueError('Acoustic tokens would change; only run before acoustic/speech training')
    train=[read_audio(p,model.config.sample_rate) for p in sorted(Path(args.data,'train').glob('*.wav'))]
    validation=[read_audio(p,model.config.sample_rate)[:,:model.config.sample_rate] for p in sorted(Path(args.data,'validation').glob('*.wav'))]
    if not train or not validation:
        raise ValueError('Separate train/validation recordings required')
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith('codec.decoder') if args.decoder_only else name.startswith('codec.'))
    before=evaluate(model,validation)
    starting_step=int(model.codec_steps)
    original=copy.deepcopy(model.codec.state_dict())
    print('Before:',before,flush=True)
    optimizer=torch.optim.AdamW(model.codec.parameters(),lr=3e-4,weight_decay=1e-4)
    logs=[];best_score=before['normalized_mse']+.1*before['boundary_error'];best_state=original;best_step=starting_step
    phases = [('decoder',args.steps)] if args.decoder_only else [('continuous',args.warmup),('quantized',args.steps)]
    for phase,steps in phases:
        if phase=='quantized':
            initialize_codebooks(model,train,generator)
            optimizer=torch.optim.AdamW(model.codec.parameters(),lr=1e-4,weight_decay=1e-4)
        for step in range(steps):
            index=int(torch.randint(len(train),(),generator=generator))
            wav=training_crop(train[index],model.config.sample_rate,generator)
            # Include exact silence to prevent a persistent learned noise floor.
            if step%20==0:
                wav=torch.zeros_like(wav)
            optimizer.zero_grad(set_to_none=True)
            losses=model.codec_loss(wav,quantize=phase!='continuous')
            if not torch.isfinite(losses['loss']):
                raise FloatingPointError('Non-finite loss')
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.codec.parameters(),1.)
            optimizer.step()
            model.codec_steps.add_(1)
            if (step+1)%250==0 or step+1==steps:
                print(phase,step+1,'loss',float(losses['loss'].detach()),flush=True)
                if phase!='continuous':
                    metrics=evaluate(model,validation)
                    logs.append({'step':int(model.codec_steps),**metrics})
                    print('Validation:',metrics,flush=True)
                    if metrics['normalized_mse']+.1*metrics['boundary_error']<best_score:
                        best_score=metrics['normalized_mse']+.1*metrics['boundary_error']
                        best_state=copy.deepcopy(model.codec.state_dict())
                        best_step=int(model.codec_steps)
    model.codec.load_state_dict(best_state)
    model.codec_steps.fill_(best_step)
    after=evaluate(model,validation)
    # Refuse to silently publish a regression in either fidelity or silence noise.
    accepted=after['normalized_mse']<before['normalized_mse'] and after['silence_rms']<=max(.001,before['silence_rms']*1.1)
    if accepted:
        model.save(args.checkpoint,fp16=True)
    else:
        print('Candidate not published: fidelity/noise checks failed',flush=True)
    Path('reports/codec_boundary_fix.json' if args.decoder_only else 'reports/codec_noise_fix.json').write_text(json.dumps({'before':before,'after':after,
        'accepted':accepted,'starting_step':starting_step,'selected_step':best_step,'warmup_steps':0 if args.decoder_only else args.warmup,
        'quantized_steps':args.steps,'validation':logs,'external_weights':False,
        'human_listening_verified':False,'scope':'Chronologically held-out parts of the same video'},indent=2)+'\n')
    print('After:',after,'published',accepted,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,default=Path('models/checkpoints/voice_unified.pt'))
    parser.add_argument('--data',default='data/training/video_native_v1')
    parser.add_argument('--decoder-only',action='store_true')
    parser.add_argument('--warmup',type=int,default=2500)
    parser.add_argument('--steps',type=int,default=2500)
    args=parser.parse_args()
    if args.warmup<1 or args.steps<1:
        parser.error('Steps must be positive')
    run(args)
