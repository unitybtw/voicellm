"""Sanity-first training of one native alignment/pitch/spectrum model, public data only."""
import argparse,json,time,sys,shutil,random,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import soundfile as sf
from torch.nn import functional as F
from scripts.voice import read_audio
from src.models.voice_unified import UnifiedVoiceModel
from src.models.aligned_voice import AlignedVoiceModel
from src.codec.pitch import pitch_labels
from src.data.video_audio import file_sha256
from src.checkpoint import write_json_atomic
from src.training.aligned_state import TrainingOrder, optimizer_path_for, restore_optimizer, save_training_state, step_optimizer


def select(split,count,min_seconds,max_seconds):
    records=[json.loads(x) for x in Path('data/public/ljspeech_spectral_v1',split+'.jsonl').read_text().splitlines()]
    records=[r for r in records if min_seconds<=r['duration_seconds']<=max_seconds]
    random.Random(103).shuffle(records)
    # A short fixed sanity set is explicitly a memorization check, not a generalization claim.
    return records[:count]

@torch.inference_mode()
def evaluate(model,pairs,oracle=False):
    model.eval();results=[]
    for row,target,pitch in pairs:
        predicted,controls,total=model.predict_acoustics(row['text']);truth=target.to(predicted)
        pred=F.interpolate(predicted.float().transpose(1,2),size=truth.shape[1],mode='linear',align_corners=False).transpose(1,2)
        delta=truth.diff(dim=1).square().mean().clamp_min(1e-8)
        r={'id':row['id'],'free_mse':float((pred-truth).square().mean()),'mean_baseline_mse':float(truth.square().mean()),
           'delta_ratio':float(pred.diff(dim=1).square().mean()/delta),'duration_error':abs(float(total)-truth.shape[1])/truth.shape[1]}
        controls=F.interpolate(controls.float().transpose(1,2),size=truth.shape[1],mode='linear',align_corners=False).transpose(1,2)
        f0,voice=(x.to(controls) for x in pitch)
        r['voiced_log_f0_mae']=float(((controls[...,0]-f0).abs()*voice).sum()/voice.sum().clamp_min(1))
        r['voicing_bce']=float(F.binary_cross_entropy_with_logits(controls[...,1],voice))
        r['voicing_error']=float(((controls[...,1]>=0)!=(voice>=.5)).float().mean())
        energy_error=model.codec.log_rms(pred)-model.codec.log_rms(truth)
        r['log_rms_mae']=float(energy_error.abs().mean())
        r['median_frame_rms_ratio']=float(energy_error.median().exp())
        if oracle:
            encoded,_,prior=model.encode_text(row['text']);path=model.alignment(prior,truth)
            expanded,_=model.expand(encoded,path);aligned,_=model.decode_hidden(expanded)
            r['oracle_alignment_mse']=float((aligned-truth).square().mean())
        results.append(r)
    return {'clips':results,**{key:sum(r[key] for r in results)/len(results) for key in results[0] if key!='id'}}

def selection_score(training,validation,clips):
    metrics=training if clips<=16 else validation
    return (metrics['free_mse']+.1*metrics['duration_error']
            +.1*metrics['voiced_log_f0_mae']+.1*metrics['voicing_bce']+.1*metrics['log_rms_mae']
            +.05*abs(math.log(max(.01,metrics['delta_ratio']))))


def run(args):
    torch.set_num_threads(4);torch.manual_seed(103)
    if args.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    active=Path('models/checkpoints/voice_unified.pt');meta_path=Path(str(active)+'.training.json')
    source_path=args.checkpoint or active
    source_meta_path=Path(str(source_path)+'.training.json')
    meta=json.loads(source_meta_path.read_text());sha=file_sha256(source_path)
    if meta.get('external_weights') is not False or meta.get('user_audio_used') is not False or meta.get('checkpoint_sha256')!=sha:raise ValueError('Verified public native checkpoint required')
    source=UnifiedVoiceModel.load(source_path).float()
    resume=isinstance(source,AlignedVoiceModel)
    source_modes=(source.config.clock_mode,source.config.periodic_prior_mode) if resume else None
    if resume:model=source
    else:
        model=AlignedVoiceModel()
        for part in ('text_embedding','layers','final_norm'):getattr(model.backbone,part).load_state_dict(getattr(source.backbone,part).state_dict())
    transfer={'kind':'aligned_resume' if resume else 'own_public_backbone','source_sha256':sha,'source_speech_steps':int(source.speech_steps)}
    del source
    if args.soft_clock:
        model.config.clock_mode='soft'
        transfer['clock_upgrade']='differentiable predicted clock with acoustic supervision'
    if args.confident_periodic_prior:
        model.config.periodic_prior_mode='confidence'
        transfer['periodic_prior_upgrade']='No harmonic spectrum constraint for voicing probability <= 0.5'
    snapshot=Path('archive/checkpoints')/('aligned_source_'+time.strftime('%Y%m%dT%H%M%S'))
    snapshot.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source_path,snapshot/source_path.name);shutil.copy2(source_meta_path,snapshot/source_meta_path.name)
    records=select('train',args.clips,args.min_seconds,args.max_seconds)
    validation=select('validation',32,args.min_seconds,args.max_seconds);test=select('test',32,args.min_seconds,args.max_seconds)
    if not records or not validation or not test:raise ValueError('Missing public data split')
    waves=[]
    for row in records+validation+test:
        if row.get('dataset')!='LJSpeech-1.1' or row.get('source')!='human_recording' or not Path(row['audio']).resolve().is_relative_to(Path('data/public/LJSpeech-1.1/wavs').resolve()):raise ValueError('Only public human recordings allowed')
    for row in records:waves.append(read_audio(row['audio'],16000))
    if not resume:model.codec.fit(waves)
    def pair(row,wave=None):
        wave=read_audio(row['audio'],16000) if wave is None else wave
        features=model.codec.encode(wave)[0];pitch=pitch_labels(wave,model.config.n_fft,model.config.hop_length,model.config.sample_rate)
        if features.shape[1]>model.config.max_seq_len or len(model.phonemizer.encode(row['text']))>features.shape[1]:raise ValueError('Complete aligned pair exceeds context')
        return row,features,pitch
    pairs=[pair(r,w) for r,w in zip(records,waves)];del waves
    valid_pairs=[pair(r) for r in validation];test_pairs=[pair(r) for r in test]
    model.to(args.device)
    cached_paths={}
    if args.fixed_alignments:
        model.eval()
        with torch.no_grad():
            for row,target,_ in pairs:
                _,_,prior=model.encode_text(row['text'])
                cached_paths[row['id']]=model.alignment(prior,target).cpu()
    initial_train=evaluate(model,pairs[:8],True);initial_valid=evaluate(model,valid_pairs)
    upgraded_modes=(model.config.clock_mode,model.config.periodic_prior_mode)
    if resume and source_modes!=upgraded_modes:
        model.config.clock_mode,model.config.periodic_prior_mode=source_modes
        try:before_train=evaluate(model,pairs[:8],True);before_valid=evaluate(model,valid_pairs)
        finally:model.config.clock_mode,model.config.periodic_prior_mode=upgraded_modes
    else:before_train,before_valid=initial_train,initial_valid
    print('Initial train', {k:v for k,v in before_train.items() if k!='clips'},flush=True)
    print('Initial valid', {k:v for k,v in before_valid.items() if k!='clips'},flush=True)
    if source_modes!=upgraded_modes:
        print('Upgrade initial valid', {k:v for k,v in initial_valid.items() if k!='clips'},flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    optimizer_path=Path('models/checkpoints/aligned_optimizer.pt')
    saved_order=None
    if resume:
        optimizer_source=optimizer_path_for(source_path,meta,active)
        saved_order=restore_optimizer(optimizer,source_path,optimizer_source,int(model.speech_steps),args.device,args.lr)
    start=int(model.speech_steps);best=selection_score(before_train,before_valid,args.clips) if resume else float('inf');selected_step=start;logs=[]
    order=TrainingOrder([r['id'] for r in records],seed=103+start,state=saved_order)
    initial_order=order.state_dict();exposures={r['id']:0 for r in records};selected_exposures=dict(exposures)
    candidate=Path('models/checkpoints/voice_aligned_candidate.pt');candidate_opt=Path('models/checkpoints/aligned_candidate_optimizer.pt')
    if resume:
        # Retain the verified starting model and its moments if this run never improves it.
        shutil.copy2(optimizer_source,snapshot/optimizer_source.name)
        if source_path.resolve()!=candidate.resolve():shutil.copy2(source_path,candidate)
        if optimizer_source.resolve()!=candidate_opt.resolve():shutil.copy2(optimizer_source,candidate_opt)
        retained_meta={**meta,'optimizer_file':candidate_opt.name}
        write_json_atomic(retained_meta,Path(str(candidate)+'.training.json'))
    executed_steps=0
    latest_checkpoint=Path('models/checkpoints/voice_aligned_last_attempt.pt')
    latest_optimizer=latest_checkpoint.with_suffix('.optimizer.pt')
    for step in range(args.steps):
        row,target,pitch=pairs[order.next_index()];exposures[row['id']]+=1;model.train();optimizer.zero_grad(set_to_none=True)
        losses=model.speech_loss(row['text'],target.to(args.device),pitch,guidance=max(.02,.2*(1-step/max(1,args.steps))),alignment_path=cached_paths.get(row['id']))
        loss=losses['loss']
        step_optimizer(model,optimizer,loss)
        executed_steps=step+1
        if (step+1)%10==0 and args.device=='mps':torch.mps.empty_cache()
        if (step+1)%100==0 or step+1==args.steps:
            training=evaluate(model,pairs[:8],True);valid=evaluate(model,valid_pairs)
            metrics={'step':int(model.speech_steps),'train':training,'validation':valid};logs.append(metrics)
            print('Step',int(model.speech_steps),'train',{k:round(v,5) for k,v in training.items() if k!='clips'},'valid',{k:round(v,5) for k,v in valid.items() if k!='clips'},flush=True)
            # Sanity before scale: prioritize actual free synthesis on known training text.
            score=selection_score(training,valid,args.clips)
            if score<best:
                best=score;selected_step=int(model.speech_steps);selected_exposures=dict(exposures)
                save_training_state(model,optimizer,candidate,candidate_opt,transfer,order)
            save_training_state(model,optimizer,latest_checkpoint,latest_optimizer,transfer,order,kind='attempt')
            if args.sanity_only and training['free_mse']<.2 and training['duration_error']<.05 and training['delta_ratio']>.15:
                print('Free synthesis sanity threshold reached; stopping this small-set test',flush=True)
                break
    selected=UnifiedVoiceModel.load(candidate).float().to(args.device)
    final_train=evaluate(selected,pairs[:8],True);final_valid=evaluate(selected,valid_pairs);final_test=evaluate(selected,test_pairs)
    accepted=(not args.sanity_only and final_train['free_mse']<.4 and final_train['delta_ratio']>.1 and final_valid['free_mse']<before_valid['free_mse']
        and final_valid['free_mse']<final_valid['mean_baseline_mse'])
    stamp=time.strftime('%Y%m%dT%H%M%S');output=Path('reports/training')/('aligned_'+stamp+'.json')
    report={'model_kind':model.model_kind,'transfer':transfer,'external_weights':False,'user_audio_used':False,
        'attempted_steps':executed_steps,'requested_max_steps':args.steps,'sanity_only':args.sanity_only,'selected_step':selected_step,'accepted':accepted,'train_sanity_before':before_train,
        'train_sanity_after':final_train,'validation_before':before_valid,'validation':final_valid,'test':final_test,'history':logs,
        'configuration_initial_train':initial_train,'configuration_initial_validation':initial_valid,
        'training_ids':[r['id'] for r in records],'training_texts':[r['text'] for r in records],
        'data_scope':'Fixed public subset with separate validation and test splits; training audio is never played as synthesized speech.',
        'selection_objective':'Free spectrum MSE + 0.1 duration error + 0.1 voiced log-F0 MAE + 0.1 voicing BCE + 0.1 log-RMS MAE + 0.05 abs(log(delta ratio)); training for <=16 clips, validation otherwise.',
        'source_retained':selected_step==start,
        'dictionary_provenance':'data/pronunciation/provenance.json','unknown_words':{r['id']:model.phonemizer.encode_with_metadata(r['text'])[1]['unknown_words'] for r in records},
        'alignment':'Native monotonic path from predicted phone spectra and real target spectrum; not human timestamp labels.',
        'speech_ready':False,'intelligibility_verified':False,'user_rejected_previous_generation':True}
    report['candidate_sha256']=file_sha256(candidate)
    report['clock_mode']=selected.config.clock_mode
    report['periodic_prior_mode']=selected.config.periodic_prior_mode
    report['fixed_alignment_paths']={key:value.tolist() for key,value in cached_paths.items()}
    report['selected_training_exposures']=selected_exposures
    report['attempted_training_exposures']=exposures
    report['training_order_before']=initial_order
    report['training_order_restored']=order.restored
    report['training_order_after']=order.state_dict()
    report['last_attempt_checkpoint_sha256']=file_sha256(latest_checkpoint)
    report['last_attempt_optimizer']=str(latest_optimizer.resolve())
    report['last_attempt_resumable']=True
    for name,row in [('sanity',records[0]),('unseen',validation[0])]:
        wave=torch.cat(list(selected.stream(row['text'],max_frames=600)),-1).flatten().float().numpy()
        audio=Path('reports')/('aligned_'+name+'.wav');sf.write(audio,wave,16000)
        report[name+'_sample']={'text':row['text'],'id':row['id'],'audio':str(audio.resolve()),'training_text':name=='sanity'}
    sf.write('reports/aligned_dsp_reference.wav',selected.codec.decode(pairs[0][1]).flatten().numpy(),16000)
    if accepted:
        archive=Path('archive/checkpoints')/('before_alignment_'+stamp);archive.mkdir(parents=True)
        for path in (active,meta_path,Path('models/checkpoints/spectral_optimizer.pt'),optimizer_path):
            if path.exists():shutil.copy2(path,archive/path.name)
        shutil.copy2(candidate,active);shutil.copy2(candidate_opt,optimizer_path)
        report['checkpoint_sha256']=file_sha256(active)
        write_json_atomic({'stage':'speech','model_kind':selected.model_kind,'speech_steps':selected_step,'codec_steps':0,
            'external_weights':False,'user_audio_used':False,'checkpoint_sha256':report['checkpoint_sha256'],'training_report':str(output),
            'optimizer_file':optimizer_path.name,'speech_ready':False,'intelligibility_verified':False,'transfer':transfer},meta_path)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    Path('reports/aligned_training_latest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Saved',output,'accepted',accepted,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--steps',type=int,default=1600);p.add_argument('--clips',type=int,default=16)
    p.add_argument('--min-seconds',type=float,default=2.);p.add_argument('--max-seconds',type=float,default=4.5)
    p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--device',default='mps')
    p.add_argument('--checkpoint',type=Path,help='Verified public development candidate to resume instead of the active model')
    p.add_argument('--soft-clock',action='store_true',help='Train free predicted timing directly through the acoustic loss')
    p.add_argument('--sanity-only',action='store_true',help='Stop at the small-set learning threshold; never promote an overfit sanity model')
    p.add_argument('--fixed-alignments',action='store_true',help='Cache own data-derived alignments for a stable small-set learning test')
    p.add_argument('--confident-periodic-prior',action='store_true',help='Constrain harmonic spectra only when the voiced classifier is confident')
    a=p.parse_args()
    if a.steps<1 or a.clips<1 or a.min_seconds<=0 or a.max_seconds<a.min_seconds or a.lr<=0:p.error('Positive, consistent training arguments required')
    run(a)
