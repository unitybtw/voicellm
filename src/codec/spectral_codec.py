"""Fixed spectral analysis and phase reconstruction. No learned/external vocoder."""
import torch
from torch import nn

class SpectralCodec(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=320, bins=80, iterations=24):
        super().__init__()
        self.sample_rate,self.n_fft,self.hop_length,self.iterations=sample_rate,n_fft,hop_length,iterations
        edges=700*(10**(torch.linspace(0,2595*torch.log10(torch.tensor(1+sample_rate/1400)),bins+2)/2595)-1)
        freq=torch.linspace(0,sample_rate/2,n_fft//2+1)
        filt=torch.minimum((freq[None]-edges[:-2,None])/(edges[1:-1,None]-edges[:-2,None]),
                           (edges[2:,None]-freq[None])/(edges[2:,None]-edges[1:-1,None])).clamp_min(0)
        filt=filt/filt.sum(1,keepdim=True).clamp_min(1e-8)
        self.register_buffer('filters',filt)
        self.register_buffer('inverse',torch.linalg.pinv(filt))
        self.register_buffer('window',torch.hann_window(n_fft))
        self.register_buffer('mean',torch.zeros(bins))
        self.register_buffer('std',torch.ones(bins))

    def spectrum(self,wav):
        return torch.stft(wav.detach().float().cpu().reshape(-1),self.n_fft,self.hop_length,
                          window=self.window.float().cpu(),center=True,pad_mode='constant',return_complex=True)

    def encode(self,wav):
        if wav.ndim not in (1,2,3) or any(s != 1 for s in wav.shape[:-1]) or not torch.isfinite(wav).all() or wav.numel()==0:
            raise ValueError('One finite nonempty mono recording required')
        mel=(self.filters.float().cpu()@self.spectrum(wav).abs()).clamp_min(1e-5).log().T
        features=((mel-self.mean.float().cpu())/self.std.float().cpu()).unsqueeze(0)
        return features,features,torch.tensor(0.)

    @torch.no_grad()
    def fit(self,waves):
        values=torch.cat([self.encode(w)[0][0] for w in waves])
        self.mean.copy_(values.mean(0)); self.std.copy_(values.std(0).clamp_min(.3))

    @torch.no_grad()
    def decode(self,features):
        if features.ndim!=3 or features.shape[0]!=1 or features.shape[1]<1 or features.shape[2]!=self.mean.numel() or not torch.isfinite(features).all():
            raise ValueError('One finite feature sequence required')
        logmel=features.detach().float().cpu()[0]*self.std.float().cpu()+self.mean.float().cpu()
        magnitude=(self.inverse.float().cpu()@logmel.clamp(-12,8).exp().T).clamp_min(0)
        if magnitude.shape[1]<2: magnitude=magnitude.repeat(1,2)
        length=(magnitude.shape[1]-1)*self.hop_length
        window=self.window.float().cpu()
        generator=torch.Generator().manual_seed(97)
        phase=torch.polar(torch.ones_like(magnitude),torch.rand(magnitude.shape,generator=generator)*6.2831853)
        for _ in range(self.iterations):
            wave=torch.istft(magnitude*phase,self.n_fft,self.hop_length,window=window,length=length)
            rebuilt=self.spectrum(wave)
            phase=rebuilt/rebuilt.abs().clamp_min(1e-8)
        wave=torch.istft(magnitude*phase,self.n_fft,self.hop_length,window=window,length=length)
        return wave.clamp(-1,1).reshape(1,1,-1)
