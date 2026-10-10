"""Train/evaluate the native spectral replacement using public human pairs only."""
import argparse,copy,json,time,sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
import soundfile as sf
from scripts.prepare_public import prepare
from scripts.train_public import rows
from scripts.voice import read_audio
from src.models.spectral_voice import SpectralConfig,SpectralVoiceModel
from src.models.voice_unified import UnifiedVoiceModel
from src.data.video_audio import file_sha256

@torch.inference_mode()
def evaluate(model,pairs):
    model.eval();results=[]
    for row,target in pairs:
        free,total=model.predict(row['text'])
        resized=F.interpolate(free.transpose(1,2).float(),size=target.shape[1],mode='linear',align_corners=False).transpose(1,2)
        truth=target.to(resized)
        pred,_=model.predict(row['text'],target.shape[1],64)
        results.append({'id':row['id'],'free_spectral_mse':float((resized-truth).square().mean()),
            'teacher_length_mse':float((pred-truth).square().mean()),'constant_mean_mse':float(truth.square().mean()),
            'duration_relative_error':abs(float(total)-target.shape[1])/target.shape[1],
            'free_temporal_delta':float(free.float().diff(dim=1).square().mean())})
    return {'clips':results,**{k:sum(r[k] for r in results)/len(results) for k in results[0] if k!='id'}}

def run(args):
    torch.set_num_threads(4);torch.manual_seed(97)
    if args.device=='mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS required but unavailable')
    active=Path('models/checkpoints/voice_unified.pt');metadata=Path(str(active)+'.training.json')
    source_meta=json.loads(metadata.read_text())
    if source_meta.get('user_audio_used') is not False or source_meta.get('external_weights') is not False:
        raise ValueError('Only verified public native weights may be used')
    source_sha=file_sha256(active)
    if source_meta.get('checkpoint_sha256') != source_sha:
        raise ValueError('Training metadata does not match active checkpoint')
    root=Path('data/public/ljspeech_spectral_v1')
    if not (root/'provenance.json').exists(): prepare('data/public/LJSpeech-1.1',root,config=SpectralConfig())
    train=rows(root/'train.jsonl',args.clips);valid=rows(root/'validation.jsonl',8);test=rows(root/'test.jsonl',8)
    for row in train+valid+test:
        if row.get('dataset')!='LJSpeech-1.1' or row.get('source')!='human_recording' or not Path(row['audio']).resolve().is_relative_to(Path('data/public/LJSpeech-1.1/wavs').resolve()):
            raise ValueError('Only public human-recorded LJSpeech pairs are permitted')
    source=UnifiedVoiceModel.load(active).float()
    if getattr(source,'model_kind',None)=='phoneme_aligned_spectrum':
        raise ValueError('Use train_aligned.py to continue the aligned model; refusing acoustic reset')
    if isinstance(source,SpectralVoiceModel):
        model=source;transfer={'kind':'spectral_resume','source_sha256':source_sha,'source_steps':int(source.speech_steps)}
    else:
        model=SpectralVoiceModel()
        for name in ('text_embedding','layers','final_norm'):
            getattr(model.backbone,name).load_state_dict(getattr(source.backbone,name).state_dict())
        transfer={'kind':'own_public_text_backbone','source_sha256':source_sha,'source_steps':int(source.speech_steps)}
    del source
    waves=[read_audio(row['audio'],model.config.sample_rate) for row in train]
    if transfer['kind']!='spectral_resume': model.codec.fit(waves[:256])
    pairs=[(row,model.codec.encode(w)[0]) for row,w in zip(train,waves)]
    del waves
    def encode(records):return [(row,model.codec.encode(read_audio(row['audio'],model.config.sample_rate))[0]) for row in records]
    valid_pairs=encode(valid);test_pairs=encode(test)
    model.to(args.device);initial=evaluate(model,valid_pairs)
    print('Initial', {k:v for k,v in initial.items() if k!='clips'},flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    optimizer_path=Path('models/checkpoints/spectral_optimizer.pt')
    if transfer['kind']=='spectral_resume' and not optimizer_path.exists():
        raise ValueError('Matching spectral optimizer is missing; refusing a silent optimizer reset')
    if transfer['kind']=='spectral_resume' and optimizer_path.exists():
        saved=torch.load(optimizer_path,map_location='cpu',weights_only=True)
        if saved['checkpoint_sha256']!=source_sha or saved['step']!=int(model.speech_steps): raise ValueError('Stale spectral optimizer')
        optimizer.load_state_dict(saved['optimizer'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v) and k!='step':state[k]=v.to(args.device)
        for group in optimizer.param_groups:group['lr']=args.lr
    start=int(model.speech_steps);best=initial['free_spectral_mse'];logs=[];best_step=start
    candidate=Path('models/checkpoints/voice_spectral_candidate.pt')
    candidate_opt=Path('models/checkpoints/spectral_candidate_optimizer.pt')
    for step in range(args.steps):
        row,target=pairs[step%len(pairs)];model.train();optimizer.zero_grad(set_to_none=True)
        loss=model.speech_loss(row['text'],target.to(args.device))['loss']
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite spectral objective')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();model.speech_steps.add_(1)
        if (step+1)%10==0 and args.device=='mps':torch.mps.empty_cache()
        if (step+1)%100==0 or step+1==args.steps:
            metrics=evaluate(model,valid_pairs);logs.append({'step':int(model.speech_steps),**metrics})
            print('Validation',int(model.speech_steps),{k:round(v,5) for k,v in metrics.items() if k!='clips'},flush=True)
            if metrics['free_spectral_mse']<best and metrics['free_temporal_delta']>1e-5:
                best=metrics['free_spectral_mse'];best_step=int(model.speech_steps);model.save(candidate,fp16=True)
                torch.save({'optimizer':optimizer.state_dict(),'checkpoint_sha256':file_sha256(candidate),'step':best_step},candidate_opt)
    accepted=best_step>start
    if accepted:
        selected=UnifiedVoiceModel.load(candidate).float().to(args.device)
        final=evaluate(selected,valid_pairs);testing=evaluate(selected,test_pairs)
        accepted=final['free_spectral_mse']<initial['free_spectral_mse'] and final['free_spectral_mse']<final['constant_mean_mse']
    else:final=initial;testing=None
    output=Path('reports/training')/('spectral_'+time.strftime('%Y%m%dT%H%M%S')+'.json')
    report={'model_kind':'spectral_continuous','external_weights':False,'user_audio_used':False,'transfer':transfer,
        'attempted_steps':args.steps,'selected_step':best_step,'accepted':accepted,'initial':initial,'validation':final,'test':testing,
        'history':logs,'selected_training_ids':[p[0]['id'] for p in pairs[:min(len(pairs),best_step-start)]],
        'speech_ready':False,'intelligibility_verified':False,'phase_decoder':'Fixed local Griffin-Lim DSP; block synthesis, no learned vocoder',
        'note':'Free duration prediction, no target audio history. Spectral errors do not prove intelligibility or frontier quality.'}
    if accepted:
        archive=Path('archive/checkpoints')/('before_spectral_'+time.strftime('%Y%m%dT%H%M%S'));archive.mkdir(parents=True)
        shutil.copy2(active,archive/active.name);shutil.copy2(metadata,archive/metadata.name)
        shutil.copy2(candidate,active);shutil.copy2(candidate_opt,optimizer_path)
        report['checkpoint_sha256']=file_sha256(active)
        metadata.write_text(json.dumps({'stage':'speech','model_kind':'spectral_continuous','speech_steps':best_step,'codec_steps':0,'checkpoint_sha256':report['checkpoint_sha256'],
            'external_weights':False,'user_audio_used':False,'training_report':str(output),'data_provenance':str(root/'provenance.json'),
            'speech_ready':False,'intelligibility_verified':False,'transfer':transfer},indent=2)+'\n')
        prompt='The sun was setting over a quiet village.'
        wave=torch.cat(list(selected.stream(prompt,max_frames=600)),-1).flatten().float().numpy()
        audio=Path('reports/generation_spectral_latest.wav');sf.write(audio,wave,selected.config.sample_rate)
        report.update(prompt=prompt,free_output=str(audio.resolve()),free_seconds=len(wave)/selected.config.sample_rate)
        real=valid_pairs[0];sf.write('reports/spectral_reference_reconstruction.wav',selected.codec.decode(real[1]).flatten().numpy(),selected.config.sample_rate)
    selected_exposures={}
    if accepted:
        for index in range(best_step-start):
            identifier=pairs[index%len(pairs)][0]['id']
            selected_exposures[identifier]=selected_exposures.get(identifier,0)+1
    report['selected_training_exposures']=selected_exposures
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    Path('reports/spectral_training_latest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Saved',output,'accepted',accepted,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--steps',type=int,default=2000);p.add_argument('--clips',type=int,default=1024)
    p.add_argument('--device',default='mps');p.add_argument('--lr',type=float,default=1e-4);a=p.parse_args()
    if min(a.steps,a.clips)<1 or a.lr<=0:p.error('Positive steps, clips and learning rate required')
    run(a)
