"""Full FFT magnitude preserves harmonic detail discarded by a mel inverse."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.codec.spectral_codec import SpectralCodec

class LinearSpectrum(SpectralCodec):
    def __init__(self,sample_rate=16000,n_fft=1024,hop_length=256,iterations=32):
        nn.Module.__init__(self)
        self.sample_rate,self.n_fft,self.hop_length,self.iterations=sample_rate,n_fft,hop_length,iterations
        self.register_buffer('window',torch.hann_window(n_fft))
        self.register_buffer('mean',torch.zeros(n_fft//2+1))
        self.register_buffer('std',torch.ones(n_fft//2+1))

    def encode(self,wav):
        if wav.ndim not in (1,2,3) or any(s!=1 for s in wav.shape[:-1]) or wav.numel()==0 or not torch.isfinite(wav).all():raise ValueError('One finite nonempty recording required')
        log=self.spectrum(wav).abs().clamp_min(1e-5).log().T
        features=((log-self.mean.float().cpu())/self.std.float().cpu()).unsqueeze(0)
        return features,features,torch.tensor(0.)

    def log_rms(self,features):
        """Differentiable frame energy from the one-sided FFT, including window gain."""
        logmag=features.float()*self.std.to(features.device).float()+self.mean.to(features.device).float()
        weights=torch.full_like(self.mean.to(features.device).float(),2.)
        weights[0]=1.
        if self.n_fft%2==0:weights[-1]=1.
        gain=self.window.to(features.device).float().square().mean()
        return .5*(torch.logsumexp(2*logmag+weights.log(),dim=-1)-2*torch.log(features.new_tensor(float(self.n_fft)))-gain.log())

    @torch.no_grad()
    def decode(self,features):
        if features.ndim!=3 or features.shape[0]!=1 or features.shape[1]<2 or features.shape[2]!=self.mean.numel() or not torch.isfinite(features).all():raise ValueError('Finite full-spectrum sequence required')
        mag=(features.detach().float().cpu()[0]*self.std.float().cpu()+self.mean.float().cpu()).clamp(-12,8).exp().T
        length=(mag.shape[1]-1)*self.hop_length
        gen=torch.Generator().manual_seed(97)
        phase=torch.polar(torch.ones_like(mag),torch.rand(mag.shape,generator=gen)*6.2831853)
        window=self.window.float().cpu();previous=torch.zeros_like(phase)
        for _ in range(self.iterations):
            wave=torch.istft(mag*phase,self.n_fft,self.hop_length,window=window,length=length)
            spectrum=self.spectrum(wave)
            accelerated=spectrum-(.99/1.99)*previous
            phase=accelerated/accelerated.abs().clamp_min(1e-8);previous=spectrum
        wave=torch.istft(mag*phase,self.n_fft,self.hop_length,window=window,length=length)
        return wave.clamp(-1,1).reshape(1,1,-1)

    @torch.no_grad()
    def decode_source_filter(self,features,pitch):
        """Coherent harmonic excitation plus filtered noise, driven by this model's heads."""
        if features.ndim!=3 or features.shape[0]!=1 or features.shape[1]<2 or features.shape[2]!=self.mean.numel() or pitch.shape!=(*features.shape[:2],2):
            raise ValueError('One matching spectrum/pitch sequence required')
        if not torch.isfinite(features).all() or not torch.isfinite(pitch).all():raise ValueError('Finite acoustic predictions required')
        mag=(features.detach().float().cpu()[0]*self.std.float().cpu()+self.mean.float().cpu()).clamp(-12,8).exp()
        pitch=pitch.detach().float().cpu()[0]
        frames=mag.shape[0];length=(frames-1)*self.hop_length
        f0=(pitch[:,0].clamp(-2.1,1.4)*.5+torch.tensor(200.).log()).exp()
        # This head predicts a binary class probability, not harmonic energy.
        # Interpolation below smooths class changes over one acoustic hop.
        voice=(pitch[:,1]>=0).float()
        # A fixed 64-harmonic cap discarded audible upper bands at low pitches.
        harmonics=torch.arange(1,math.ceil(self.sample_rate/2/float(f0.min()))+1).float()
        frequencies=f0[:,None]*harmonics[None]
        bins=(frequencies*self.n_fft/self.sample_rate).clamp(0,mag.shape[1]-1)
        left=bins.floor().long().clamp(0,mag.shape[1]-2);fraction=bins-left
        envelope=F.max_pool1d(mag[:,None],kernel_size=5,stride=1,padding=2)[:,0]
        amplitudes=(envelope.gather(1,left)*(1-fraction)+envelope.gather(1,left+1)*fraction)*2/self.window.float().cpu().sum()
        amplitudes=amplitudes*(frequencies<self.sample_rate/2)
        target_rms=(mag.square().sum(-1)*2).sqrt()/self.n_fft/(.375**.5)
        harmonic_rms=(amplitudes.square().sum(-1)*.5).sqrt().clamp_min(1e-6)
        amplitudes=amplitudes*(target_rms/harmonic_rms).clamp(.2,3)[:,None]
        def interpolate(values):
            # STFT frame centers are hop-spaced; match those times rather than scaling T to N.
            positions=torch.arange(length)/self.hop_length
            a=positions.long().clamp_max(frames-2);mix=(positions-a).reshape(-1,*([1]*(values.ndim-1)))
            return values[a]*(1-mix)+values[a+1]*mix
        sample_f0=interpolate(f0)
        phase=torch.cumsum(sample_f0,0)*(6.2831853/self.sample_rate)
        sample_amplitude=interpolate(amplitudes)
        sample_amplitude=sample_amplitude*(sample_f0[:,None]*harmonics[None]<self.sample_rate/2)
        # Disperse harmonic phases to eliminate zero-phase impulse-train buzzer timbre.
        schroeder=-math.pi*harmonics*(harmonics-1)/max(len(harmonics),1)
        harmonic=(torch.sin(phase[:,None]*harmonics[None]+schroeder[None])*sample_amplitude).sum(-1)
        generator=torch.Generator().manual_seed(97)
        # Random phase alone cannot make a harmonic spectrum aperiodic. Smooth
        # its power over one pitch spacing before filtering the noise source.
        widths=(f0*self.n_fft/self.sample_rate).ceil().long().clamp_min(3)
        widths=widths+(widths%2==0).long()
        noise_mag=torch.empty_like(mag)
        power=mag.square()
        for width in widths.unique().tolist():
            mask=widths==width
            noise_mag[mask]=F.avg_pool1d(power[mask,None],width,stride=1,
                padding=width//2,count_include_pad=False)[:,0].sqrt()
        noise_phase=torch.rand(noise_mag.T.shape,generator=generator)*6.2831853
        noise=torch.istft(torch.polar(noise_mag.T,noise_phase),self.n_fft,self.hop_length,window=self.window.float().cpu(),length=length)
        # Independent random phases lose energy during overlap/add. Match the
        # predicted frame energy before mixing; otherwise even q=.2 becomes a buzz.
        noise_rms=F.avg_pool1d(noise.square()[None,None],self.n_fft,self.hop_length,
            padding=self.n_fft//2,count_include_pad=False).flatten().clamp_min(1e-16).sqrt()
        noise=noise*interpolate((target_rms/noise_rms).clamp(.25,4.))
        sample_voice=interpolate(voice).clamp(0,1)
        wave=harmonic*sample_voice.sqrt()+noise*(1-sample_voice).sqrt()
        fade=min(self.sample_rate//100,length//2)
        if fade:
            ramp=torch.linspace(0,1,fade)
            wave[:fade]*=ramp;wave[-fade:]*=ramp.flip(0)
        return wave.clamp(-1,1).reshape(1,1,-1)
