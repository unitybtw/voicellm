"""Measure real token rollout and free output, not only teacher-forced cross entropy."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
import soundfile as sf
from scripts.train_public import rows
from scripts.voice import read_audio
from src.models.voice_unified import UnifiedVoiceModel
from src.data.video_audio import file_sha256


@torch.inference_mode()
def review(checkpoint, output, clips=8, temperature=0.):
    torch.set_num_threads(4)
    torch.manual_seed(97)
    model=UnifiedVoiceModel.load(checkpoint).float()
    if getattr(model, 'model_kind', None) in ('spectral_continuous','phoneme_aligned_spectrum'):
        raise ValueError('RVQ review is incompatible with spectral features; use train_spectral.py evaluation')
    records=rows(Path('data/public/ljspeech_v1/validation.jsonl'),clips)
    results=[]
    for row in records:
        wav=read_audio(row['audio'],model.config.sample_rate)
        _,tokens,_=model.codec.encode(wav)
        ids=torch.tensor([model.tokenizer.encode(row['text'])+[model.tokenizer.speech]])
        embeddings=torch.cat([model.backbone.text_embedding(ids),model.backbone.embed_audio_tokens(tokens)],1)
        outputs,_=model.backbone(inputs_embeds=embeddings,compute_audio_logits=False)
        hidden=outputs['hidden_states'][:,ids.shape[1]-1:ids.shape[1]-1+tokens.shape[-1]]
        logits=model.conditioned_audio_logits(hidden,tokens)
        rollout=model.sample_audio_tokens(hidden)
        rec=model.codec.decode(rollout)[...,:wav.shape[-1]]
        codec=model.codec.decode(tokens)[...,:wav.shape[-1]]
        energy=wav.square().mean().clamp_min(1e-9)
        results.append({'id':row['id'],
            'gold_prefix_accuracy':float((logits.argmax(-1)==tokens).float().mean()),
            'predicted_prefix_accuracy':float((rollout==tokens).float().mean()),
            'accuracy_by_book':(rollout==tokens).float().mean((0,2)).tolist(),
            'predicted_wave_normalized_mse':float((rec-wav).square().mean()/energy),
            'codec_normalized_mse':float((codec-wav).square().mean()/energy)})
    prompt='The sun was setting over a quiet village.'
    chunks=list(model.stream(prompt,max_frames=1200,temperature=temperature))
    generated=torch.cat(chunks,-1).flatten().float().numpy()
    audio=output.with_suffix('.wav');audio.parent.mkdir(parents=True,exist_ok=True)
    sf.write(audio,generated,model.config.sample_rate)
    report={'checkpoint_sha256':file_sha256(checkpoint),'speech_steps':int(model.speech_steps),
        'depth_hidden':model.config.depth_hidden,'temperature':temperature,'clips':results,
        'mean_predicted_wave_normalized_mse':sum(r['predicted_wave_normalized_mse'] for r in results)/len(results),
        'mean_predicted_prefix_accuracy':sum(r['predicted_prefix_accuracy'] for r in results)/len(results),
        'prompt':prompt,'free_output':str(audio),'free_seconds':len(generated)/model.config.sample_rate,
        'max_seconds':6.,'human_listening_verified':False,'intelligibility_verified':False,
        'note':'Gold temporal history, predicted within-frame prefixes. This is still easier than fully free speech; no intelligibility claim.'}
    frames=generated[:len(generated)//model.config.hop_length*model.config.hop_length].reshape(-1,model.config.hop_length)
    differences=np.mean(np.diff(frames,axis=0)**2,axis=1)
    energy=np.mean(frames[1:]**2,axis=1)
    repeated=float(np.mean(differences/np.maximum(energy,1e-9)<.001)) if len(differences) else 1.
    report.update(free_repeated_frame_fraction=repeated,speech_ready=False,
                  failure_flags=['repeated_frame_collapse'] if repeated>.8 else ['intelligibility_not_verified'])
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='clips'})
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--clips',type=int,default=8)
    parser.add_argument('--temperature',type=float,default=0.)
    args=parser.parse_args()
    if args.clips<1:parser.error('At least one real validation clip required')
    review(args.checkpoint,args.output,args.clips,args.temperature)
