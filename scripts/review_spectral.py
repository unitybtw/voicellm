"""Independent free text-to-spectrum evaluation; never equate metrics with intelligibility."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import soundfile as sf
from scripts.voice import read_audio
from scripts.train_public import rows
from scripts.train_spectral import evaluate
from src.models.voice_unified import UnifiedVoiceModel
from src.models.spectral_voice import SpectralVoiceModel
from src.data.video_audio import file_sha256

@torch.inference_mode()
def review(checkpoint,output,clips=8,device='cpu'):
    torch.set_num_threads(4)
    model=UnifiedVoiceModel.load(checkpoint).float().to(device)
    if not isinstance(model,SpectralVoiceModel):raise ValueError('Native spectral checkpoint required')
    root=Path('data/public/ljspeech_spectral_v1');results={}
    for split in ('validation','test'):
        pairs=[(row,model.codec.encode(read_audio(row['audio'],model.config.sample_rate))[0]) for row in rows(root/(split+'.jsonl'),clips)]
        results[split]=evaluate(model,pairs)
    prompt='The sun was setting over a quiet village.'
    chunks=list(model.stream(prompt,max_frames=600));wave=torch.cat(chunks,-1).flatten().float().cpu()
    output.parent.mkdir(parents=True,exist_ok=True);audio=output.with_suffix('.wav');sf.write(audio,wave.numpy(),model.config.sample_rate)
    report={'checkpoint_sha256':file_sha256(checkpoint),'model_kind':model.model_kind,'speech_steps':int(model.speech_steps),
        'parameters':model.count_parameters(),'checkpoint_mib':checkpoint.stat().st_size/1048576,
        'external_weights':False,'user_audio_used':False,'prompt':prompt,'free_output':str(audio.resolve()),
        'free_seconds':wave.numel()/model.config.sample_rate,'finite_audio':bool(torch.isfinite(wave).all()),
        'audio_rms':float(wave.square().mean().sqrt()),'clipped_sample_fraction':float((wave.abs()>=.999).float().mean()),
        'speech_ready':False,'human_listening_verified':False,'intelligibility_verified':False,**results,
        'note':'Free predictions without target audio. DSP phase reconstruction runs after the whole spectrum. Numerical quality does not establish speech intelligibility.'}
    output.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k not in ('validation','test')})
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('reports/generation_spectral_review.json'));p.add_argument('--clips',type=int,default=8);p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if a.clips<1:p.error('Positive clip count required')
    review(a.checkpoint,a.output,a.clips,a.device)
