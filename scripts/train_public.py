"""Fresh native codec and paired TTS training using public human recordings only."""
import argparse
import copy
import json
from pathlib import Path
import random
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.voice import read_audio, training_crop
from scripts.improve_codec import initialize_codebooks, evaluate
from scripts.upgrade_codec import initialize_short_frames
from src.models.voice_unified import UnifiedVoiceModel
from src.data.video_audio import file_sha256


def rows(path, limit):
    records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    random.Random(97).shuffle(records)
    return records[:limit]


@torch.no_grad()
def speech_validation(model, pairs):
    previous=model.training
    model.eval()
    values=[model.speech_loss(row['text'],tokens=tokens.to(next(model.parameters()).device))
            for row,tokens in pairs]
    model.train(previous)
    return {key:sum(float(v[key]) for v in values)/len(values) for key in values[0]}


def run(args):
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS GPU is unavailable in this environment; use --device cpu or GPU-enabled execution')
    torch.set_num_threads(4)
    torch.manual_seed(97)
    generator=torch.Generator().manual_seed(97)
    root=Path('data/public/ljspeech_v1')
    train=rows(root/'train.jsonl',args.clips)
    valid=rows(root/'validation.jsonl',16)
    test=rows(root/'test.jsonl',16)
    if not train or not valid or not test:
        raise ValueError('Real aligned train, validation and test pairs required')
    # Never inherit user-video weights, even for codec initialization.
    model=UnifiedVoiceModel().eval()
    initialize_short_frames(model)
    waves=[read_audio(row['audio'],model.config.sample_rate) for row in train]
    valid_waves=[read_audio(row['audio'],model.config.sample_rate) for row in valid]
    crops=[training_crop(w,model.config.sample_rate,generator) for w in waves[:384]]
    initialize_codebooks(model,crops,generator)
    with torch.no_grad():
        for q in model.codec.rvq.quantizers:
            q.embedding.weight[0].zero_()
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith('codec.decoder'))
    initial=evaluate(model,valid_waves)
    print('Public codec initial:',initial,flush=True)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    best=None;best_score=float('inf');best_step=0;codec_logs=[]
    for step in range(args.codec_steps):
        wav=training_crop(waves[int(torch.randint(len(waves),(),generator=generator))],model.config.sample_rate,generator)
        if step%10==0:
            wav=torch.zeros_like(wav)
        optimizer.zero_grad(set_to_none=True)
        loss=model.codec_loss(wav)['loss']
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite codec objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.codec.parameters(),1.)
        optimizer.step()
        if (step+1)%100==0 or step+1==args.codec_steps:
            metrics=evaluate(model,valid_waves)
            codec_logs.append({'step':step+1,**metrics})
            print('Codec',step+1,metrics,flush=True)
            score=metrics['normalized_mse']+.1*metrics['boundary_error']
            if score<best_score:
                best_score=score;best=copy.deepcopy(model.codec.state_dict());best_step=step+1
    model.codec.load_state_dict(best)
    model.codec_steps.fill_(best_step)
    # Token labels must be derived from the exact FP16-rounded codec that we publish.
    model.codec.half().float()
    codec_after=evaluate(model,valid_waves)
    if codec_after['normalized_mse']>=.2 or codec_after['silence_rms']!=0:
        raise ValueError('Public codec fidelity/silence checks failed')
    candidate=Path('models/checkpoints/voice_public_candidate.pt')
    model.save(candidate,fp16=True)
    model.to(args.device)
    def tokenize(records):
        pairs=[]
        with torch.no_grad():
            for i,row in enumerate(records):
                wav=read_audio(row['audio'],model.config.sample_rate).to(args.device)
                _,tokens,_=model.codec.encode(wav)
                if tokens.shape[-1]+len(model.tokenizer.encode(row['text']))+1>model.config.max_seq_len:
                    raise ValueError('Complete audio/text pair exceeds context')
                pairs.append((row,tokens.cpu()))
                if (i+1)%128==0:
                    print('Tokenized',i+1,'/',len(records),flush=True)
        return pairs
    train_pairs=tokenize(train);valid_pairs=tokenize(valid);test_pairs=tokenize(test)
    before=speech_validation(model,valid_pairs)
    print('Paired English before:',before,flush=True)
    for name,p in model.named_parameters():
        p.requires_grad_(not name.startswith('codec.'))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    best_loss=before['loss'];speech_logs=[];best_speech_step=0;order=[]
    model.train()
    for step in range(args.speech_steps):
        if not order:
            order=torch.randperm(len(train_pairs),generator=generator).tolist()
        row,tokens=train_pairs[order.pop()]
        optimizer.zero_grad(set_to_none=True)
        losses=model.speech_loss(row['text'],tokens=tokens.to(args.device),
                                 pad_to_multiple=128 if args.device=='mps' else 1)
        if not torch.isfinite(losses['loss']):
            raise FloatingPointError('Non-finite paired speech objective')
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step()
        model.speech_steps.add_(1)
        if (step+1)%10==0:
            print('Paired speech',step+1,'loss',float(losses['loss'].detach()),flush=True)
        if (step+1)%100==0 or step+1==args.speech_steps:
            metrics=speech_validation(model,valid_pairs)
            speech_logs.append({'step':step+1,**metrics})
            print('Paired validation:',metrics,flush=True)
            if metrics['loss']<best_loss:
                best_loss=metrics['loss'];best_speech_step=step+1
                model.save(candidate,fp16=True)
                # An auxiliary optimizer file supports continuing the same run.
                torch.save({'step':step+1,'checkpoint_sha256':file_sha256(candidate),'optimizer':optimizer.state_dict()},
                           Path('models/checkpoints/public_optimizer.pt'))
        del losses
        if args.device == 'mps' and (step+1)%10 == 0:
            torch.mps.synchronize()
            torch.mps.empty_cache()
    stored=UnifiedVoiceModel.load(candidate,args.device).float()
    after=speech_validation(stored,valid_pairs)
    test_metrics=speech_validation(stored,test_pairs)
    accepted=best_speech_step>0 and after['loss']<before['loss']
    destination=Path('models/checkpoints/voice_unified.pt')
    if accepted:
        archive=Path('archive/checkpoints');archive.mkdir(parents=True,exist_ok=True)
        if destination.exists() and not (archive/'user_video_before_public_data.pt').exists():
            shutil.copy2(destination,archive/'user_video_before_public_data.pt')
            sidecar=Path(str(destination)+'.training.json')
            if sidecar.exists():
                shutil.copy2(sidecar,archive/'user_video_before_public_data.pt.training.json')
        candidate.replace(destination)
    report={'dataset_provenance':str(root/'provenance.json'),
        'fresh_initialization':True,'external_weights':False,'user_audio_used':False,
        'device':args.device,'train_clips_selected':len(train),'validation_clips':len(valid),'test_clips':len(test),
        'training_ids':[r['id'] for r in train],'validation_ids':[r['id'] for r in valid],
        'test_ids':[r['id'] for r in test],
        'codec_initial':initial,'codec_after':codec_after,'codec_steps_attempted':args.codec_steps,
        'selected_codec_steps':best_step,'speech_steps_attempted':args.speech_steps,
        'selected_speech_steps':best_speech_step,
        'unique_paired_training_clips_seen':min(args.speech_steps,len(train_pairs)),
        'speech_before':before,'speech_after':after,
        'test_teacher_forcing':test_metrics,'accepted':accepted,
        'checkpoint_sha256':file_sha256(destination) if accepted else None,
        'codec_validation':codec_logs,'speech_validation':speech_logs,
        'human_listening_verified':False,'free_generation_quality_verified':False,
        'note':'Initial training run, not a finished English TTS system. Evaluation is teacher-forced; not intelligibility.'}
    report_path=Path('reports/public_ljspeech_training.json')
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    if accepted:
        Path(str(destination)+'.training.json').write_text(json.dumps({
            'stage':'speech','speech_steps':int(stored.speech_steps),'codec_steps':int(stored.codec_steps),
            'audio_steps':0,'external_weights':False,'user_audio_used':False,
            'checkpoint_sha256':file_sha256(destination),'training_report':str(report_path),
            'data_provenance':str(root/'provenance.json')},indent=2)+'\n')
    print('Public model published:',accepted,'Paired validation:',after,'Test:',test_metrics,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    parser.add_argument('--clips',type=int,default=1024)
    parser.add_argument('--codec-steps',type=int,default=2000)
    parser.add_argument('--speech-steps',type=int,default=1000)
    parser.add_argument('--lr',type=float,default=1e-4)
    args=parser.parse_args()
    if min(args.clips,args.codec_steps,args.speech_steps)<1 or not 0<args.lr<=.01:
        parser.error('Positive counts and learning rate <= .01 required')
    run(args)
