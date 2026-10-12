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
    def decode_source_filter(self,features,pitch,mean=None,std=None):
        """Coherent harmonic excitation plus filtered noise, driven by this model's heads."""
        if features.ndim!=3 or features.shape[0]!=1 or features.shape[1]<2 or features.shape[2]!=self.mean.numel() or pitch.shape!=(*features.shape[:2],2):
            raise ValueError('One matching spectrum/pitch sequence required')
        if not torch.isfinite(features).all() or not torch.isfinite(pitch).all():raise ValueError('Finite acoustic predictions required')
        target_mean = self.mean.float().cpu() if mean is None else mean.float().cpu()
        target_std = self.std.float().cpu() if std is None else std.float().cpu()
        mag=(features.detach().float().cpu()[0]*target_std+target_mean).clamp(-12,8).exp()
        pitch=pitch.detach().float().cpu()[0]
        frames=mag.shape[0];length=(frames-1)*self.hop_length
        f0=(pitch[:,0].clamp(-2.1,1.4)*.5+torch.tensor(200.).log()).exp()
        # Smooth continuous confidence voicing prevents abrupt harmonic generator toggling.
        voice=(2*pitch[:,1].sigmoid()-1).clamp_min(0)
        # A fixed 64-harmonic cap discarded audible upper bands at low pitches.
        harmonics=torch.arange(1,math.ceil(self.sample_rate/2/float(f0.min()))+1).float()
        frequencies=f0[:,None]*harmonics[None]
        bins=(frequencies*self.n_fft/self.sample_rate).clamp(0,mag.shape[1]-1)
        left=bins.floor().long().clamp(0,mag.shape[1]-2);fraction=bins-left
        envelope=F.max_pool1d(mag[:,None],kernel_size=5,stride=1,padding=2)[:,0]
        amplitudes=(envelope.gather(1,left)*(1-fraction)+envelope.gather(1,left+1)*fraction)*2/self.window.float().cpu().sum()
        # Smooth Nyquist roll-off taper avoids sudden spectral step discontinuities.
        taper=((self.sample_rate/2-frequencies)/800).clamp(0,1)
        amplitudes=amplitudes*taper
        # Attenuate sub-audible mechanical rumble (<90 Hz) while preserving male voice fundamental (100-180 Hz).
        sub_mask=frequencies<90
        sub_weight=torch.ones_like(frequencies)
        sub_weight[sub_mask]=0.5+0.5*(frequencies[sub_mask]/90)
        formant_boost=torch.ones_like(frequencies)
        formant_boost[(frequencies>=1200)&(frequencies<3500)]=1.55
        formant_boost[(frequencies>=3500)&(frequencies<7500)]=1.25
        amplitudes=amplitudes*sub_weight*formant_boost
        target_rms=(mag.square().sum(-1)*2).sqrt()/self.n_fft/(.375**.5)
        harmonic_rms=(amplitudes.square().sum(-1)*.5).sqrt().clamp_min(1e-6)
        gain=(target_rms/harmonic_rms).clamp(.4,2.5)
        if frames>4:
            kernel=torch.tensor([0.1,0.2,0.4,0.2,0.1],device=mag.device,dtype=mag.dtype).view(1,1,5)
            gain=F.conv1d(F.pad(gain.view(1,1,-1),(2,2),mode='replicate'),kernel).view(-1)
        amplitudes=amplitudes*gain[:,None]
        def interpolate(values):
            # STFT frame centers are hop-spaced; match those times rather than scaling T to N.
            positions=torch.arange(length)/self.hop_length
            a=positions.long().clamp_max(frames-2);mix=(positions-a).reshape(-1,*([1]*(values.ndim-1)))
            return values[a]*(1-mix)+values[a+1]*mix
        sample_f0=interpolate(f0)
        phase=torch.cumsum(sample_f0,0)*(6.2831853/self.sample_rate)
        sample_amplitude=interpolate(amplitudes)
        sample_freqs=sample_f0[:,None]*harmonics[None]
        sample_amplitude=sample_amplitude*((self.sample_rate/2-sample_freqs)/800).clamp(0,1)
        # Disperse harmonic phases to eliminate zero-phase impulse-train buzzer timbre.
        schroeder=-math.pi*harmonics*(harmonics-1)/max(len(harmonics),1)
        # Diffuse phase jitter above Maximum Voiced Frequency (~2000 Hz) eliminates
        # synthetic upper-harmonic comb-filter whistle while preserving low-harmonic pitch clarity.
        gen=torch.Generator().manual_seed(97)
        diffuse_weight=((frequencies-2000)/2500).clamp(0,1)
        frame_jitter=(torch.rand((frames,len(harmonics)),generator=gen)*2-1)*math.pi*diffuse_weight
        sample_jitter=interpolate(frame_jitter)
        harmonic=(torch.sin(phase[:,None]*harmonics[None]+schroeder[None]+sample_jitter)*sample_amplitude).sum(-1)
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
        noise_gain=(target_rms/noise_rms).clamp(.3,3.)
        if noise_gain.numel()>4:
            kernel=torch.tensor([0.1,0.2,0.4,0.2,0.1],device=noise_gain.device,dtype=noise_gain.dtype).view(1,1,5)
            noise_gain=F.conv1d(F.pad(noise_gain.view(1,1,-1),(2,2),mode='replicate'),kernel).view(-1)
        noise=noise*interpolate(noise_gain)
        sample_voice=interpolate(voice).clamp(0,1)
        # Subtle glottal breathiness floor (HNR ~ 25 dB) gives natural human vocal cord texture.
        eff_voice=sample_voice*0.97
        eff_noise=(1-sample_voice)+sample_voice*0.03
        wave=harmonic*eff_voice.sqrt()+noise*eff_noise.sqrt()
        fade=min(self.sample_rate//100,length//2)
        if fade:
            ramp=torch.linspace(0,1,fade)
            wave[:fade]*=ramp;wave[-fade:]*=ramp.flip(0)
        peak = wave.abs().max()
        if peak > 0.95:
            wave = wave * (0.95 / peak)
        return wave.clamp(-1,1).reshape(1,1,-1)
