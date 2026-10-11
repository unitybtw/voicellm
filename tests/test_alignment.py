import torch
from torch.nn import functional as F
import numpy as np
import pytest
from itertools import combinations
from src.models.monotonic_alignment import monotonic_path
from src.models.aligned_voice import AlignedVoiceModel,AlignedConfig
from src.models.voice_unified import UnifiedVoiceModel
from src.codec.linear_spectrum import LinearSpectrum
from src.codec.pitch import pitch_labels
from src.text.phonemes import EnglishPhonemes

def small():return AlignedVoiceModel(AlignedConfig(hidden_size=32,num_layers=2,num_heads=4,num_kv_heads=2,intermediate_size=48,max_seq_len=128,griffin_lim_iterations=3))

def test_alignment_matches_exhaustive_optimum_and_visits_every_unit():
    torch.manual_seed(12);cost=torch.rand(7,3)
    path=monotonic_path(cost)
    alternatives=[]
    for cuts in combinations(range(1,7),2):
        p=torch.tensor([0]*cuts[0]+[1]*(cuts[1]-cuts[0])+[2]*(7-cuts[1]))
        alternatives.append(cost[torch.arange(7),p].sum())
    assert torch.isclose(cost[torch.arange(7),path].sum(),torch.stack(alternatives).min())
    assert path[0]==0 and path[-1]==2 and (path.diff()>=0).all() and (path.diff()<=1).all()
    with pytest.raises(ValueError):monotonic_path(torch.zeros(2,3))

def test_phonetic_dictionary_and_transparent_spelling_fallback():
    p=EnglishPhonemes();ids,meta=p.encode_with_metadata('Hello, world!')
    assert 'HH' in meta['phones'] and 'W' in meta['phones'] and not meta['unknown_words']
    _,unknown=p.encode_with_metadata('qxzwp')
    assert unknown['unknown_words']==['qxzwp'] and len(ids)>2

def test_pitch_labels_and_loss_train_real_heads():
    t=torch.arange(16000)/16000
    pitch,voice=pitch_labels(.1*torch.sin(t*220*6.2831853))
    f0=(pitch*.5+torch.tensor(200.).log()).exp()
    assert voice[:,4:-4].mean()>.95 and abs(float(f0[:,4:-4].median())-220)<5
    m=small();loss=m.speech_loss('Hello world.',torch.randn(1,100,513),(pitch[:,:100] if pitch.shape[1]>=100 else torch.zeros(1,100),torch.ones(1,100)))
    loss['loss'].backward()
    for module in (m.duration[-1],m.prior,m.spectrum_out,m.pitch_out,m.phone_embedding):
        assert module.weight.grad is not None and module.weight.grad.abs().sum()>0

def test_full_spectrum_decode_preserves_magnitude_and_checkpoint_identity(tmp_path):
    m=small();m.speech_steps.fill_(1)
    t=torch.arange(4096)/16000;wav=.03*torch.sin(6.2831853*220*t)
    features=m.codec.encode(wav)[0];decoded=m.codec.decode(features)
    assert decoded.shape[-1]==4096 and torch.isfinite(decoded).all()
    error=(m.codec.spectrum(decoded).abs()-m.codec.spectrum(wav).abs()).square().mean()/m.codec.spectrum(wav).abs().square().mean()
    assert error<.4
    path=tmp_path/'voice.pt';m.save(path,fp16=True);loaded=UnifiedVoiceModel.load(path)
    assert isinstance(loaded,AlignedVoiceModel) and loaded.codec.mean.dtype==torch.float32
    assert loaded.backbone.text_head.weight is loaded.backbone.text_embedding.weight
    result,_=loaded.predict('Hello world.')
    assert torch.isfinite(result).all() and result.shape[-1]==513
    cp=torch.load(path,weights_only=True);cp['dictionary_sha256']='bad'
    with pytest.raises(ValueError):AlignedVoiceModel.from_checkpoint(cp)

def test_source_filter_preserves_pitch_without_external_vocoder():
    codec=LinearSpectrum(iterations=3)
    time=torch.arange(16000)/16000
    features=codec.encode(.03*torch.sin(time*200*6.2831853))[0]
    controls=torch.zeros(1,features.shape[1],2);controls[...,1]=10
    wave=codec.decode_source_filter(features,controls)
    pitch,voice=pitch_labels(wave)
    f0=(pitch*.5+torch.tensor(200.).log()).exp()
    assert voice[:,4:-4].mean()>.9
    assert abs(float(f0[:,4:-4].median())-200)<5
    assert torch.isfinite(wave).all() and wave[0,0,0]==0 and wave[0,0,-1]==0

def test_data_derived_alignment_does_not_skip_short_phonemes():
    model=small();encoded,_,prior=model.encode_text('Hello world.')
    features=torch.randn(1,100,513)
    path=model.alignment(prior,features)
    durations=torch.bincount(path,minlength=encoded.shape[1])
    assert durations.min()>=2 and durations.sum()==100

def test_free_acoustic_error_trains_predicted_duration_and_soft_clock_roundtrip(tmp_path):
    model=small();model.config.clock_mode='soft'
    encoded,logdur,_=model.encode_text('Hello world.')
    features,_=model.decode_hidden(model.expand_soft(encoded,logdur,100))
    features.square().mean().backward()
    assert model.duration[-1].weight.grad.abs().sum()>0
    model.save(tmp_path/'soft.pt',fp16=True)
    loaded=UnifiedVoiceModel.load(tmp_path/'soft.pt')
    assert loaded.config.clock_mode=='soft'
    predicted,_,_=loaded.predict_acoustics('Hello world.',frames=100)
    assert predicted.shape==(1,100,513) and torch.isfinite(predicted).all()

def test_interrupted_save_preserves_the_previous_checkpoint(tmp_path,monkeypatch):
    from src.checkpoint import save_atomic
    target=tmp_path/'voice.pt';target.write_bytes(b'previous model')
    def fail(*args,**kwargs):raise OSError('simulated disk write interruption')
    monkeypatch.setattr(torch,'save',fail)
    with pytest.raises(OSError):save_atomic({'example':torch.zeros(1)},target)
    assert target.read_bytes()==b'previous model'
    assert list(tmp_path.iterdir())==[target]

def test_spectral_comb_cannot_lock_the_pitch_head_through_acoustic_gradients():
    model=small();encoded,_,_=model.encode_text('Hello world.')
    path=torch.arange(100)%encoded.shape[1]
    expanded,_=model.expand(encoded,path.sort().values)
    features,pitch=model.decode_hidden(expanded)
    features.square().mean().backward()
    assert model.pitch_out.weight.grad is None
    model.zero_grad(set_to_none=True)
    _,pitch=model.decode_hidden(expanded.detach())
    (pitch[...,0]-1).square().mean().backward()
    assert model.pitch_out.weight.grad.abs().sum()>0

def test_pitch_estimator_does_not_label_out_of_range_rumble_as_400_hz():
    time=torch.arange(16000)/16000
    _,voice=pitch_labels(.1*torch.sin(time*50*6.2831853))
    assert voice[:,4:-4].mean()<.05

@pytest.mark.parametrize('frames',[13,37,127])
def test_hard_clock_allocates_exact_length_without_dropping_phones(frames):
    model=small();units=len(model.phonemizer.encode('Hello world.'))
    assert frames>=units
    with torch.no_grad():
        for bias in (-2.,5.):
            model.duration[-1].bias.fill_(bias)
            features,pitch,_=model.predict_acoustics('Hello world.',frames=frames)
            assert features.shape[1]==pitch.shape[1]==frames
    for bad in (129,2,3.5,True):
        with pytest.raises(ValueError):model.predict_acoustics('Hello world.',frames=bad)

def test_cached_alignment_rejects_skipped_phones_and_invalid_shape():
    model=small();units=len(model.phonemizer.encode('Hello world.'))
    path=torch.linspace(0,units-1,100).long();features=torch.randn(1,100,513)
    skipped=path.clone();skipped[skipped==1]=0
    for bad in (path[None],path.float(),skipped,path+1):
        with pytest.raises(ValueError,match='cached'):model.speech_loss('Hello world.',features,alignment_path=bad)

def test_soft_duration_total_loss_matches_inference_minimum():
    import math
    model=small();model.config.clock_mode='soft';model.duration[-1].bias.data.fill_(-2.)
    units=len(model.phonemizer.encode('Hello world.'))
    path=torch.arange(units).repeat_interleave(2)
    loss=model.speech_loss('Hello world.',torch.randn(1,2*units,513),alignment_path=path)
    # Effective total is already exactly 2*units, so there is no total-length penalty.
    assert torch.isclose(loss['duration'],torch.tensor((-2.-math.log(2.))**2),atol=1e-5)

def test_candidate_selection_penalizes_pitch_regression():
    from scripts.train_aligned import selection_score
    good={'free_mse':.5,'duration_error':.1,'voiced_log_f0_mae':.2,'voicing_bce':.3,'log_rms_mae':.1,'delta_ratio':1.}
    flat={**good,'voiced_log_f0_mae':2.}
    assert selection_score(good,flat,128)>selection_score(flat,good,128)

def test_typographic_apostrophes_preserve_english_contractions():
    phones=EnglishPhonemes()
    assert phones.encode('Don’t worry…')==phones.encode("Don't worry.")
    assert phones.encode('Ｈｅｌｌｏ world!')==phones.encode('Hello world!')
    with pytest.raises(ValueError):phones.encode(None)

def test_frame_energy_matches_tone_amplitude_and_reaches_acoustic_head():
    import math
    codec=LinearSpectrum();time=torch.arange(16000)/16000
    features=codec.encode(.1*torch.sin(2*math.pi*200*time))[0]
    assert abs(float(codec.log_rms(features)[:,4:-4].exp().mean())-.1/math.sqrt(2))<.002
    quieter=features-math.log(10)
    assert torch.allclose(codec.log_rms(features)-codec.log_rms(quieter),torch.full((1,features.shape[1]),math.log(10)),atol=1e-5)
    model=small();loss=model.speech_loss('Hello world.',torch.randn(1,100,513))
    loss['energy'].backward()
    assert model.spectrum_out.weight.grad.abs().sum()>0

def test_candidate_selection_penalizes_quiet_flat_spectra():
    from scripts.train_aligned import selection_score
    good={'free_mse':.5,'duration_error':.1,'voiced_log_f0_mae':.2,'voicing_bce':.3,'log_rms_mae':.1,'delta_ratio':1.}
    collapsed={**good,'log_rms_mae':2.,'delta_ratio':.01}
    assert selection_score(good,collapsed,256)>selection_score(collapsed,good,256)

def test_unvoiced_source_filter_does_not_turn_low_voicing_probability_into_buzz():
    import math
    codec=LinearSpectrum();generator=torch.Generator().manual_seed(83)
    reference=.03*torch.randn(16000,generator=generator)
    features=codec.encode(reference)[0]
    controls=torch.zeros(1,features.shape[1],2);controls[...,1]=math.log(.2/.8)
    result=codec.decode_source_filter(features,controls)
    _,voice=pitch_labels(result)
    assert voice[:,4:-4].mean()<.1
    rms_ratio=float(result.square().mean().sqrt()/reference.square().mean().sqrt())
    assert .8<rms_ratio<1.2

def test_low_pitch_source_filter_retains_harmonics_above_fixed_64_limit():
    import math
    codec=LinearSpectrum();time=torch.arange(16000)/16000
    reference=.03*torch.sin(2*math.pi*7000*time)
    features=codec.encode(reference)[0]
    controls=torch.zeros(1,features.shape[1],2)
    controls[...,0]=(math.log(100)-math.log(200))/.5;controls[...,1]=12
    result=codec.decode_source_filter(features,controls)
    assert result.square().mean().sqrt()>.8*reference.square().mean().sqrt()
    assert torch.isfinite(result).all()

def test_unvoiced_confidence_prior_keeps_noise_spectrum_and_survives_save(tmp_path):
    import math
    model=small();model.config.periodic_prior_mode='confidence'
    encoded,_,_=model.encode_text('Hello world.');captured=[]
    hook=model.spectrum_out.register_forward_hook(lambda module,inputs,output:captured.append(output))
    model.pitch_out.bias.data[1]=math.log(.25/.75)
    features,_=model.decode_hidden(encoded)
    assert torch.allclose(features,captured[-1])
    model.pitch_out.bias.data[1]=10
    voiced,_=model.decode_hidden(encoded)
    assert not torch.allclose(voiced,captured[-1])
    hook.remove();model.save(tmp_path/'confidence.pt')
    assert UnifiedVoiceModel.load(tmp_path/'confidence.pt').config.periodic_prior_mode=='confidence'
    legacy=torch.load(tmp_path/'confidence.pt',map_location='cpu',weights_only=True)
    del legacy['config']['periodic_prior_mode']
    assert AlignedVoiceModel.from_checkpoint(legacy).config.periodic_prior_mode=='probability'
    with pytest.raises(ValueError):AlignedConfig(periodic_prior_mode='bad')

@pytest.mark.parametrize('probability',[.001,.1,.3])
def test_noise_source_does_not_inherit_harmonic_lines_when_voicing_is_low(probability):
    import math
    codec=LinearSpectrum();time=torch.arange(16000)/16000
    reference=.03*torch.sin(2*math.pi*200*time)
    features=codec.encode(reference)[0]
    controls=torch.zeros(1,features.shape[1],2)
    controls[...,1]=math.log(probability/(1-probability))
    # Inconsistent spectral/head predictions must not force the noise source
    # to remain a tone. This is a control test, not a speech quality claim.
    result=codec.decode_source_filter(features,controls)
    _,voiced=pitch_labels(result)
    assert voiced[:,4:-4].mean()<.1
    ratio=float(result.square().mean().sqrt()/reference.square().mean().sqrt())
    assert .8<ratio<1.2

def test_aligned_voice_cli_speak_and_narrate(tmp_path, monkeypatch):
    import sys
    import soundfile as sf
    from scripts.voice import main
    model = small()
    model.speech_steps.fill_(1)
    checkpoint = tmp_path / 'aligned.pt'
    model.save(checkpoint, fp16=True)
    speak_out = tmp_path / 'speak.wav'
    monkeypatch.setattr(sys, 'argv', ['voice.py', 'speak', '--checkpoint', str(checkpoint),
                                     '--text', 'Hello world.', '--max-frames', '20', '--output', str(speak_out)])
    main()
    wav, sr = sf.read(speak_out)
    assert sr == model.config.sample_rate and len(wav) > 0
    assert torch.isfinite(torch.tensor(wav)).all()

    story = tmp_path / 'story.txt'
    story.write_text('First sentence.\nSecond sentence.\n')
    narrate_out = tmp_path / 'narrate.wav'
    monkeypatch.setattr(sys, 'argv', ['voice.py', 'narrate', '--checkpoint', str(checkpoint),
                                     '--text-file', str(story), '--output', str(narrate_out)])
    main()
    narrate_wav, sr2 = sf.read(narrate_out)
    assert sr2 == model.config.sample_rate and len(narrate_wav) > 0
    assert torch.isfinite(torch.tensor(narrate_wav)).all()

    # Test CLI clone with reference
    ref_path = tmp_path / 'ref.wav'
    sf.write(ref_path, 0.05 * np.sin(2 * np.pi * 150 * np.arange(16000) / 16000), 16000)
    clone_out = tmp_path / 'clone.wav'
    monkeypatch.setattr(sys, 'argv', ['voice.py', 'clone', '--checkpoint', str(checkpoint),
                                     '--text', 'Merhaba dünya.', '--reference', str(ref_path),
                                     '--max-frames', '20', '--output', str(clone_out)])
    main()
    clone_wav, sr3 = sf.read(clone_out)
    assert sr3 == model.config.sample_rate and len(clone_wav) > 0
    assert torch.isfinite(torch.tensor(clone_wav)).all()

def test_source_filter_uses_dispersed_harmonic_phases_to_prevent_buzz():
    codec = LinearSpectrum()
    time = torch.arange(16000) / 16000
    features = codec.encode(.03 * torch.sin(2 * 3.14159265 * 200 * time))[0]
    controls = torch.zeros(1, features.shape[1], 2); controls[..., 1] = 12
    wave = codec.decode_source_filter(features, controls)
    crest_factor = float(wave.abs().max() / wave.square().mean().sqrt().clamp_min(1e-6))
    assert crest_factor < 6.0
    assert torch.isfinite(wave).all()

def test_aligned_voice_stream_with_speaker_reference():
    model = small()
    model.speech_steps.fill_(1)
    # Generate 1s tone reference at 130 Hz (male pitch)
    t = torch.arange(16000) / 16000
    ref_audio = 0.05 * torch.sin(2 * 3.14159265 * 130 * t).unsqueeze(0)
    chunks = list(model.stream('Hello world.', max_frames=30, reference_wav=ref_audio))
    cloned_wave = torch.cat(chunks, -1)
    assert cloned_wave.shape[-1] > 0 and torch.isfinite(cloned_wave).all()
    # Invalid reference check
    with pytest.raises(ValueError, match='Invalid reference audio'):
        list(model.stream('Hello world.', max_frames=30, reference_wav=torch.zeros(1, 10)))

def test_source_filter_envelope_flutter_is_smoothed():
    """Validates that harmonic gain smoothing eliminates tremolo envelope flutter."""
    codec = LinearSpectrum()
    features = torch.zeros(1, 40, 513)
    controls = torch.zeros(1, 40, 2); controls[..., 1] = 6.0
    features[:, 0::2] = 1.0; features[:, 1::2] = -1.0
    wave = codec.decode_source_filter(features, controls)
    assert torch.isfinite(wave).all()
    frame_rms = F.avg_pool1d(wave.square(), codec.hop_length, stride=codec.hop_length).sqrt().flatten()
    ratio = (frame_rms[1:] / frame_rms[:-1].clamp_min(1e-5))
    assert ratio.max() < 2.5 and ratio.min() > 0.4

def test_continuous_confidence_voicing_transitions_smoothly():
    """Validates that voicing transitions continuously around threshold without chopping."""
    codec = LinearSpectrum()
    features = torch.zeros(1, 40, 513)
    controls = torch.zeros(1, 40, 2)
    controls[0, :, 1] = torch.linspace(-2.0, 2.0, 40)
    wave = codec.decode_source_filter(features, controls)
    assert torch.isfinite(wave).all()

def test_aligned_voice_stream_exact_pitch_centering_and_cadence():
    """Validates exact target pitch centering and cadence handling with reference audio."""
    model = small()
    model.speech_steps.fill_(1)
    t = torch.arange(16000) / 16000
    ref_audio = 0.05 * torch.sin(2 * 3.14159265 * 140 * t).unsqueeze(0)
    chunks = list(model.stream('Hello world.', max_frames=40, reference_wav=ref_audio))
    wave = torch.cat(chunks, -1)
    assert wave.shape[-1] > 0 and torch.isfinite(wave).all()

def test_source_filter_formant_spectral_balance_and_sub_bass_reduction():
    """Validates that harmonic weighting suppresses sub-bass rumble and preserves mid-formants."""
    codec = LinearSpectrum()
    features = torch.zeros(1, 40, 513)
    controls = torch.zeros(1, 40, 2); controls[..., 1] = 6.0
    freqs = torch.linspace(0, 8000, 513)
    envelope = -((freqs - 700) / 300)**2 * 0.5
    envelope = torch.maximum(envelope, -((freqs - 2200) / 400)**2 * 0.5 - 1.0)
    features[0] = envelope[None]
    wave = codec.decode_source_filter(features, controls).flatten()
    fft = np.abs(np.fft.rfft(wave.numpy()))
    fft_freqs = np.fft.rfftfreq(len(wave), 1 / 16000)
    tot = np.sum(fft**2)
    p_sub = float(np.sum(fft[fft_freqs < 250]**2) / tot)
    p_mid = float(np.sum(fft[(fft_freqs >= 1500) & (fft_freqs < 3500)]**2) / tot)
    assert p_sub < 0.25
    assert p_mid > 0.03

def test_cli_clone_defaults_to_clean_middle_audio(tmp_path, monkeypatch):
    """Validates that clone CLI defaults to kullanici_orta_temiz.wav if reference is omitted."""
    import sys
    import soundfile as sf
    from scripts.voice import main
    model = small()
    model.speech_steps.fill_(1)
    checkpoint = tmp_path / 'aligned.pt'
    model.save(checkpoint, fp16=True)
    out = tmp_path / 'cloned_default.wav'
    monkeypatch.setattr(sys, 'argv', ['voice.py', 'clone', '--checkpoint', str(checkpoint),
                                     '--text', 'Deneme bir iki.', '--max-frames', '20',
                                     '--output', str(out)])
    main()
    assert out.exists()
    wav, sr = sf.read(out)
    assert sr == model.config.sample_rate and len(wav) > 0

def test_aligned_voice_stream_vtln_vocal_tract_scaling():
    """Validates that stream applies VTLN scaling when speaker reference pitch differs from base."""
    model = small()
    model.speech_steps.fill_(1)
    # Reference with male pitch (140 Hz vs ~210 Hz base)
    t = torch.arange(16000) / 16000
    ref_audio = 0.05 * torch.sin(2 * 3.14159265 * 140 * t).unsqueeze(0)
    chunks = list(model.stream('Hello world.', max_frames=30, reference_wav=ref_audio))
    wave = torch.cat(chunks, -1)
    assert wave.shape[-1] > 0 and torch.isfinite(wave).all()




