import torch
import pytest
from src.models.spectral_voice import SpectralVoiceModel,SpectralConfig
from src.models.voice_unified import UnifiedVoiceModel
from src.codec.spectral_codec import SpectralCodec

def small():
    return SpectralVoiceModel(SpectralConfig(hidden_size=32,num_layers=2,num_heads=4,num_kv_heads=2,intermediate_size=48,max_seq_len=128,griffin_lim_iterations=3))

def test_spectral_shift_stability_and_finite_phase_decode():
    c=SpectralCodec(iterations=3)
    t=torch.arange(16000)/16000
    wav=(.03*torch.sin(t*440*6.2831853)).reshape(1,-1)
    a=c.encode(wav)[0];b=c.encode(torch.nn.functional.pad(wav,(1,0))[...,:16000])[0]
    # Compare magnitudes: log flooring amplifies tiny end-padding differences.
    assert (a.exp()-b.exp()).square().mean()/a.exp().square().mean()<1e-4
    decoded=c.decode(a)
    assert decoded.shape==(1,1,16000) and torch.isfinite(decoded).all()
    assert decoded.square().mean()>0

def test_real_gradients_and_no_audio_target_in_predict():
    torch.manual_seed(97);m=small()
    target=torch.randn(1,50,80)
    loss=m.speech_loss('Hello',target)['loss'];loss.backward()
    assert m.duration.weight.grad.abs().sum()>0
    assert m.mel_out.weight.grad.abs().sum()>0
    assert m.text_context.weight.grad.abs().sum()>0
    m.eval();a,_=m.predict('Hello',50);b,_=m.predict('World',50)
    assert not torch.equal(a,b)

def test_native_fp16_roundtrip_and_bounded_generation(tmp_path):
    m=small();m.speech_steps.fill_(1);m.codec.mean.fill_(-5)
    path=tmp_path/'spectral.pt';m.save(path,fp16=True)
    loaded=UnifiedVoiceModel.load(path)
    assert isinstance(loaded,SpectralVoiceModel)
    assert loaded.mel_out.weight.dtype==torch.float16
    assert loaded.codec.mean.dtype==torch.float32
    assert loaded.backbone.text_head.weight is loaded.backbone.text_embedding.weight
    wave=torch.cat(list(loaded.stream('Hello',max_frames=8)),-1)
    assert wave.shape[-1]<=8*320 and torch.isfinite(wave).all()
    loaded.save(tmp_path/'again.pt',fp16=True)
    with pytest.raises(ValueError):list(loaded.stream('Hello',reference_wav=torch.zeros(1,10)))

def test_spectral_configuration_and_empty_input():
    with pytest.raises(ValueError):SpectralConfig(hop_length=0)
    with pytest.raises(ValueError):small().predict(' ')
    with pytest.raises(ValueError):SpectralCodec().encode(torch.zeros(1,0))
    with pytest.raises(ValueError):SpectralCodec().encode(torch.zeros(2,1600))
    with pytest.raises(ValueError):list(small().stream('Hello',max_frames=1,allow_untrained=True))
    with pytest.raises(ValueError):list(small().stream('Hello',temperature=.3,allow_untrained=True))
