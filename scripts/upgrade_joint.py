"""Upgrade only our public native checkpoint to a joint within-frame predictor."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.models.voice_unified import UnifiedVoiceModel
from src.data.video_audio import file_sha256


def upgrade():
    path=Path('models/checkpoints/voice_unified.pt')
    metadata=Path(str(path)+'.training.json')
    sidecar=json.loads(metadata.read_text())
    if sidecar.get('user_audio_used') is not False or sidecar.get('checkpoint_sha256')!=file_sha256(path):
        raise ValueError('A verified public-data checkpoint is required')
    old=UnifiedVoiceModel.load(path).float()
    if old.config.depth_hidden:
        raise ValueError('Joint predictor already enabled; continue its training instead')
    model=UnifiedVoiceModel(replace(old.config,depth_hidden=128)).float()
    missing,unexpected=model.load_state_dict(old.state_dict(),strict=False)
    assert set(missing)=={'depth_in.weight','depth_out.weight'} and not unexpected
    old_names=[n for n,p in old.named_parameters() if not n.startswith('codec.')]
    new_names=[n for n,p in model.named_parameters() if not n.startswith('codec.')]
    assert new_names[:len(old_names)]==old_names
    optimizer_path=Path('models/checkpoints/public_optimizer.pt')
    state=torch.load(optimizer_path,map_location='cpu',weights_only=True)
    assert state['checkpoint_sha256']==file_sha256(path) and state['step']==int(old.speech_steps)
    groups=state['optimizer']['param_groups'];assert len(groups)==1
    assert groups[0]['params']==list(range(len(old_names)))
    # Preserve every existing Adam moment; initialize only the two new matrices.
    groups[0]['params']=list(range(len(new_names)))
    archive=Path('archive/checkpoints/public_parallel_1000');archive.mkdir(parents=True,exist_ok=True)
    for source in (path,metadata,optimizer_path):
        target=archive/source.name
        if not target.exists():shutil.copy2(source,target)
    candidate=path.with_name('voice_joint_upgrade.pt');model.save(candidate,fp16=True)
    state['checkpoint_sha256']=file_sha256(candidate)
    candidate.replace(path);torch.save(state,optimizer_path)
    sidecar.update(checkpoint_sha256=file_sha256(path),architecture_upgrade='joint_frame_prefix_128',
                   optimizer_moments_preserved=True,depth_training_steps=0,depth_started_at_speech_step=int(old.speech_steps))
    metadata.write_text(json.dumps(sidecar,indent=2)+'\n')
    print({'parameters':model.count_parameters(),'new_parameters':model.count_parameters()-old.count_parameters(),
           'codec_unchanged':True,'speech_steps':int(model.speech_steps),'moments_preserved':True})


if __name__=='__main__':
    upgrade()
