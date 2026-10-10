"""Local autocorrelation pitch labels from public waveforms, without model weights."""
import torch
import torch.nn.functional as F

@torch.no_grad()
def pitch_labels(wav,n_fft=1024,hop=256,sample_rate=16000):
    audio=wav.float().cpu().reshape(-1)
    frames=F.pad(audio,(n_fft//2,n_fft//2)).unfold(0,n_fft,hop)
    energy=frames.square().mean(-1).sqrt()
    centered=frames-frames.mean(-1,keepdim=True)
    centered=centered*torch.hann_window(n_fft)
    spectrum=torch.fft.rfft(centered,n=n_fft*2)
    corr=torch.fft.irfft(spectrum.abs().square(),n=n_fft*2)[...,:n_fft]
    corr=corr/corr[:,:1].clamp_min(1e-9)
    lo,hi=int(sample_rate/400),int(sample_rate/70)
    neighborhood=corr[:,lo:hi]
    peaks=(neighborhood>corr[:,lo-1:hi-1])&(neighborhood>=corr[:,lo+1:hi+1])
    strength,index=neighborhood.masked_fill(~peaks,float('-inf')).max(-1);lag=(index+lo).float()
    voice=(strength>.5)&(energy>.004)
    f0=sample_rate/lag
    return ((f0.clamp(70,400).log()-torch.tensor(200.).log())/.5).reshape(1,-1),voice.float().reshape(1,-1)
