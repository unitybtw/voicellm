"""One native text-to-spectrum network with learned monotonic duration pooling."""
from dataclasses import dataclass,asdict
from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.models.micro_voice_slm import MicroVoiceSLM
from src.models.voice_unified import ByteTokenizer
from src.codec.spectral_codec import SpectralCodec
from src.checkpoint import save_atomic

@dataclass
class SpectralConfig:
    sample_rate:int=16000
    hop_length:int=320
    n_fft:int=1024
    mel_bins:int=80
    hidden_size:int=384
    num_layers:int=8
    num_heads:int=6
    num_kv_heads:int=2
    intermediate_size:int=1024
    max_seq_len:int=2048
    griffin_lim_iterations:int=24
    def __post_init__(self):
        if any(type(v)!=int or v<1 for v in asdict(self).values()): raise ValueError('Positive integer configuration required')
        if self.hop_length>self.n_fft//2: raise ValueError('Spectral hop too large')

class SpectralVoiceModel(nn.Module):
    format_version=2
    model_kind='spectral_continuous'
    def __init__(self,config=None):
        super().__init__();self.config=config or SpectralConfig(); c=self.config
        self.tokenizer=ByteTokenizer()
        self.codec=SpectralCodec(c.sample_rate,c.n_fft,c.hop_length,c.mel_bins,c.griffin_lim_iterations)
        self.backbone=MicroVoiceSLM(vocab_size=260,num_audio_codebooks=0,hidden_size=c.hidden_size,num_layers=c.num_layers,
            num_heads=c.num_heads,num_kv_heads=c.num_kv_heads,intermediate_size=c.intermediate_size,max_seq_len=c.max_seq_len)
        self.backbone.text_head.weight=self.backbone.text_embedding.weight
        self.text_context=nn.Conv1d(c.hidden_size,c.hidden_size,5,padding=2)
        self.duration=nn.Linear(c.hidden_size,1)
        nn.init.zeros_(self.duration.weight);nn.init.constant_(self.duration.bias,3.)
        self.position=nn.Linear(4,c.hidden_size)
        self.mel_out=nn.Linear(c.hidden_size,c.mel_bins)
        for name in ('speech_steps','text_steps','audio_steps','codec_steps'): self.register_buffer(name,torch.tensor(0))

    def predict(self,text,frames=None,pad_to_multiple=1):
        if not text.strip(): raise ValueError('Nonempty text required')
        param=self.mel_out.weight;device=param.device
        ids=torch.tensor([list(text.encode('utf-8'))],device=device)+4
        if ids.shape[1]>self.config.max_seq_len: raise ValueError('Text exceeds context')
        encoded=self.backbone.text_embedding(ids)
        encoded=encoded+F.silu(self.text_context(encoded.transpose(1,2)).transpose(1,2))
        durations=F.softplus(self.duration(encoded).float().squeeze(-1))+.1
        total=durations.sum(-1)
        if frames is None: frames=max(2,min(self.config.max_seq_len,int(total.detach().round().item())))
        if frames<2 or frames>self.config.max_seq_len: raise ValueError('Invalid spectral length')
        centers=(durations.cumsum(-1)-durations*.5)/total[:,None]
        widths=(durations/total[:,None]*1.5).clamp_min(.008)
        pos=torch.linspace(0,1,frames,device=device)
        weights=(-(pos[None,:,None]-centers[:,None,:]).square()/(2*widths[:,None,:].square())).softmax(-1)
        pooled=weights.to(encoded.dtype)@encoded
        phases=torch.stack([pos,pos.square(),torch.sin(pos*math.pi*2),torch.cos(pos*math.pi*2)],-1)
        pooled=pooled+self.position(phases.to(param.dtype))[None]
        out,_=self.backbone(inputs_embeds=pooled,pad_to_multiple=pad_to_multiple,compute_audio_logits=False)
        return self.mel_out(out['hidden_states']),total

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def speech_loss(self,text,features):
        predicted,total=self.predict(text,features.shape[1],64)
        target=features.to(predicted)
        spectral=(predicted-target).square().mean()
        delta=(predicted.diff(dim=1)-target.diff(dim=1)).square().mean()
        duration=(total.log()-math.log(features.shape[1])).square().mean()
        return {'loss':spectral+.2*delta+.2*duration,'spectral':spectral,'delta':delta,'duration':duration}

    @torch.inference_mode()
    def stream(self,text,max_frames=400,temperature=0.,reference_wav=None,allow_untrained=False):
        if type(max_frames)!=int or max_frames<2: raise ValueError('Spectral synthesis needs at least two frames')
        if temperature != 0.: raise ValueError('Continuous spectral synthesis is deterministic; use temperature 0')
        if reference_wav is not None: raise ValueError('Public model does not use voice references')
        if int(self.speech_steps)==0 and not allow_untrained: raise ValueError('Model has not been trained')
        _,total=self.predict(text,2)
        frames=max(2,min(max_frames,self.config.max_seq_len,int(total.round().item())))
        features,_=self.predict(text,frames)
        wave=self.codec.decode(features.clamp(-5,5))
        # Spectrum/phase reconstruction is block synthesis, not low-latency streaming.
        yield from wave.split(self.config.hop_length,dim=-1)

    def save(self,path,fp16=False):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        state={k:(v.detach().cpu().half() if fp16 and v.is_floating_point() and not k.startswith('codec.') else v.detach().cpu())
               for k,v in self.state_dict().items() if k!='backbone.text_head.weight'}
        save_atomic({'format_version':2,'model_kind':self.model_kind,'external_weights':False,'config':asdict(self.config),'state':state},path)
        return path.stat().st_size

    @classmethod
    def from_checkpoint(cls,checkpoint,device='cpu'):
        if checkpoint.get('format_version')!=2 or checkpoint.get('model_kind')!=cls.model_kind or checkpoint.get('external_weights') is not False:
            raise ValueError('Not a native spectral checkpoint')
        model=cls(SpectralConfig(**checkpoint['config']))
        state=dict(checkpoint['state']);state['backbone.text_head.weight']=state['backbone.text_embedding.weight']
        model.to(dtype=state['mel_out.weight'].dtype);model.codec.float()
        model.load_state_dict(state,strict=True)
        return model.to(device).eval()
