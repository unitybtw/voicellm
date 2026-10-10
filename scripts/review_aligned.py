"""Inspect genuine generated waves and held-out data without claiming intelligibility."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import soundfile as sf
from scripts.voice import read_audio
from scripts.train_aligned import select,evaluate
from src.models.voice_unified import UnifiedVoiceModel
from src.models.aligned_voice import AlignedVoiceModel
from src.codec.pitch import pitch_labels
from src.data.video_audio import file_sha256

@torch.inference_mode()
def review(checkpoint,output,device='cpu'):
    torch.set_num_threads(4);digest=file_sha256(checkpoint)
    decoder_path=Path(__file__).resolve().parents[1]/'src/codec/linear_spectrum.py'
    decoder_digest=file_sha256(decoder_path)
    model=UnifiedVoiceModel.load(checkpoint).float().to(device)
    if not isinstance(model,AlignedVoiceModel):raise ValueError('Native aligned model required')
    results={};examples=[]
    for split in ('train','validation','test'):
        records=select(split,8,2,4.5)
        pairs=[(r,model.codec.encode(read_audio(r['audio'],16000))[0],pitch_labels(read_audio(r['audio'],16000))) for r in records]
        results[split]=evaluate(model,pairs,oracle=split=='train')
        if split=='test':continue
        row=records[0];wave=torch.cat(list(model.stream(row['text'],max_frames=600)),-1).flatten().float().cpu()
        _,controls,_=model.predict_acoustics(row['text']);controls=controls[:,:600].float().cpu()
        probabilities=controls[...,1].sigmoid()
        _,voice=pitch_labels(wave);_,realvoice=pitch_labels(read_audio(row['audio'],16000))
        path=output.parent/('generation_aligned_'+split+'.wav');sf.write(path,wave.numpy(),16000)
        examples.append({'id':row['id'],'text':row['text'],'audio':str(path.resolve()),'training_text':split=='train',
            'seconds':wave.numel()/16000,'finite':bool(torch.isfinite(wave).all()),'rms':float(wave.square().mean().sqrt()),
            'clipped_sample_fraction':float((wave.abs()>=.999).float().mean()),'measured_voiced_fraction':float(voice.mean()),
            'reference_voiced_fraction':float(realvoice.mean()),'starts_at_zero':float(wave[0])==0,'ends_at_zero':float(wave[-1])==0})
        examples[-1].update({'network_voiced_frame_fraction':float((probabilities>=.5).float().mean()),
            'network_mean_voicing_probability':float(probabilities.mean()),
            'predicted_f0_min_hz':float((controls[...,0].clamp(-2.1,1.4)*.5+torch.tensor(200.).log()).exp().min()),
            'predicted_f0_max_hz':float((controls[...,0].clamp(-2.1,1.4)*.5+torch.tensor(200.).log()).exp().max())})
    if file_sha256(checkpoint)!=digest:raise RuntimeError('Checkpoint changed during review; retry after training completes')
    if file_sha256(decoder_path)!=decoder_digest:raise RuntimeError('Decoder source changed during review; retry after edits finish')
    report={'checkpoint_sha256':digest,'model_kind':model.model_kind,'clock_mode':model.config.clock_mode,
        'periodic_prior_mode':model.config.periodic_prior_mode,
        'speech_steps':int(model.speech_steps),'parameters':model.count_parameters(),'checkpoint_mib':checkpoint.stat().st_size/1048576,
        'external_weights':False,'user_audio_used':False,'dictionary_provenance':'data/pronunciation/provenance.json',
        'phase_decoder':'Local harmonic source/filter with dynamic harmonic count, Schroeder phase dispersion, pitch-smoothed aperiodic noise, frame energy calibration and interpolated voiced/unvoiced class transitions',
        'decoder_source_sha256':decoder_digest,
        'speech_ready':False,'human_listening_verified':False,'intelligibility_verified':False,'examples':examples,**results,
        'note':'Numerical spectrum and periodicity metrics do not establish correct words. The train sample is an explicit memorization sanity check. Synthesized audio never copies or mixes reference waveform.'}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k not in ('train','validation','test')})
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path,default=Path('reports/generation_aligned_review.json'));a=p.parse_args()
    review(a.checkpoint,a.output,a.device)
