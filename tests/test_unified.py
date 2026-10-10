import importlib
from pathlib import Path
import socket
import pytest
import torch
from src.models.voice_unified import ByteTokenizer, UnifiedConfig, UnifiedVoiceModel


def tiny():
    return UnifiedVoiceModel(UnifiedConfig(hop_length=320, codec_dim=8, codebooks=2, codebook_size=16, gain_bins=1,
                             hidden_size=32, layers=2, heads=4, kv_heads=2,
                             intermediate_size=48, max_seq_len=128, depth_hidden=0))


def test_no_external_model_loaders_or_dependencies(monkeypatch):
    monkeypatch.setattr(socket, 'socket', lambda *a, **k: (_ for _ in ()).throw(AssertionError('network forbidden')))
    for path in Path('src').rglob('*.py'):
        assert 'from_pretrained' not in path.read_text()
        assert 'transformers' not in path.read_text()
        importlib.import_module('.'.join(path.with_suffix('').parts))
    model = tiny().eval()
    assert len(list(model.stream('Hello', max_frames=3, allow_untrained=True))) == 3


def test_byte_tokenizer_unicode_roundtrip():
    tokenizer = ByteTokenizer()
    text = 'English Unicode coverage: café, naïve! 2026 🎤'
    assert tokenizer.decode(tokenizer.encode(text)) == text
    with pytest.raises(ValueError):
        tokenizer.encode(' ')


def test_tied_weights_and_single_checkpoint(tmp_path):
    model = tiny()
    assert model.backbone.text_head.weight is model.backbone.text_embedding.weight
    for a, b in zip(model.backbone.audio_heads, model.backbone.audio_embeddings):
        assert a.weight is b.weight
    model.codec_steps.add_(1)
    model.save(tmp_path/'voice.pt', fp16=True)
    loaded = UnifiedVoiceModel.load(tmp_path/'voice.pt')
    assert loaded.codec_steps.item() == 1
    for a, b in zip(model.parameters(), loaded.parameters()):
        torch.testing.assert_close(a, b, atol=.001, rtol=.001, check_dtype=False)
    assert loaded.backbone.text_head.weight is loaded.backbone.text_embedding.weight


@pytest.mark.parametrize('length', [1, 319, 320, 321, 1920, 3200])
def test_causal_codec_streaming_matches_full_decode(length):
    model = tiny().eval()
    _, tokens, _ = model.codec.encode(torch.randn(1, length))
    full = model.codec.decode(tokens)
    history, parts = None, []
    for index in range(tokens.shape[-1]):
        chunk, history = model.codec.decode_frame(tokens[..., index:index+1], history)
        parts.append(chunk)
    torch.testing.assert_close(torch.cat(parts, -1), full, atol=1e-6, rtol=1e-5)
    assert full.shape[-1] == ((length+319)//320)*320


def test_codec_and_speech_training_have_real_gradients():
    torch.manual_seed(2)
    model = tiny()
    wav = torch.randn(1, 1600)*.1
    loss = model.codec_loss(wav)['loss']
    loss.backward()
    assert torch.isfinite(loss)
    assert model.codec.encoder_in.weight.grad.abs().sum() > 0
    assert model.codec.decoder_out.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    speech_loss = model.speech_loss('Hello', wav)['loss']
    speech_loss.backward()
    assert torch.isfinite(speech_loss)
    assert model.backbone.text_embedding.weight.grad.abs().sum() > 0
    assert model.backbone.audio_embeddings[0].weight.grad.abs().sum() > 0
    assert model.stop_head.weight.grad.abs().sum() > 0
    assert model.codec.encoder_in.weight.grad is None


def test_cache_matches_full_sequence():
    model = tiny().eval().backbone
    tokens = torch.randint(0, 16, (1, 2, 8))
    with torch.no_grad():
        full, _ = model(audio_tokens=tokens)
        cache, parts = None, []
        for index in range(8):
            output, cache = model(audio_tokens=tokens[..., index:index+1], past_kv_caches=cache)
            parts.append(output['audio_logits'])
        torch.testing.assert_close(torch.cat(parts, 2), full['audio_logits'], atol=1e-5, rtol=1e-5)


def test_cached_speech_targets_match_waveform_objective():
    model = tiny().eval()
    wav = torch.randn(1, 1600) * .1
    with torch.no_grad():
        _, tokens, _ = model.codec.encode(wav)
        direct = model.speech_loss('English speech.', wav)
        cached = model.speech_loss('English speech.', tokens=tokens)
        padded = model.speech_loss('English speech.', tokens=tokens, pad_to_multiple=32)
    for name in direct:
        torch.testing.assert_close(direct[name], cached[name])
        torch.testing.assert_close(direct[name], padded[name], atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match='one nonempty'):
        model.speech_loss('English speech.', tokens=tokens.expand(2, -1, -1))
    with pytest.raises(ValueError, match='one nonempty'):
        model.speech_loss('English speech.', tokens=tokens[..., :0])


def test_training_padding_preserves_gradients_and_cache_prefix():
    model = tiny().eval()
    tokens = torch.randint(0, 16, (1, 2, 5))
    model.speech_loss('Hello.', tokens=tokens)['loss'].backward()
    gradients = {name:p.grad.clone() for name,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    model.speech_loss('Hello.', tokens=tokens, pad_to_multiple=32)['loss'].backward()
    for name, p in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(p.grad, gradients[name], atol=1e-6, rtol=1e-4)
    with torch.no_grad():
        full, _ = model.backbone(audio_tokens=tokens)
        padded, cache = model.backbone(audio_tokens=tokens[..., :3], pad_to_multiple=32)
        assert cache[0]['k'].shape[2] == 3
        continuation, _ = model.backbone(audio_tokens=tokens[..., 3:], past_kv_caches=cache)
        torch.testing.assert_close(continuation['audio_logits'], full['audio_logits'][..., 3:, :], atol=1e-5, rtol=1e-5)


def test_speak_cli_exports_audio_from_fp16_checkpoint(tmp_path, monkeypatch):
    import sys
    import soundfile as sf
    from scripts.voice import main
    model = tiny()
    model.speech_steps.fill_(1)
    checkpoint = tmp_path/'native.pt'
    model.save(checkpoint, fp16=True)
    output = tmp_path/'speech.wav'
    monkeypatch.setattr(sys, 'argv', ['voice.py','speak','--checkpoint',str(checkpoint),
                                     '--text','Hello.','--max-frames','3','--output',str(output)])
    main()
    wav, rate = sf.read(output)
    assert rate == model.config.sample_rate and len(wav) > 0
    assert torch.isfinite(torch.tensor(wav)).all()


def test_untrained_generation_and_interruption_guards():
    model = tiny().eval()
    with pytest.raises(RuntimeError, match='speech training'):
        list(model.stream('Hello'))
    assert list(model.stream('Hello', max_frames=0, allow_untrained=True)) == []
    with pytest.raises(ValueError):
        list(model.stream('Hello', max_frames=500, allow_untrained=True))
    flag = {}
    generator = model.stream('Hello', max_frames=3, interruption=flag, allow_untrained=True)
    next(generator)
    flag['interrupt'] = True
    assert list(generator) == []


def test_codec_does_not_look_into_future():
    model = tiny().eval()
    wav = torch.randn(1, 3200)
    _, first, _ = model.codec.encode(wav)
    changed = wav.clone()
    changed[:, 1600:] = 100.
    _, second, _ = model.codec.encode(changed)
    torch.testing.assert_close(first[..., :5], second[..., :5])


def test_shared_language_objective_and_generation():
    model = tiny()
    losses = model.text_loss('Hello world.')
    losses['loss'].backward()
    assert model.backbone.text_embedding.weight.grad.abs().sum() > 0
    model.eval()
    with pytest.raises(RuntimeError, match='Language training'):
        model.generate_text('Hello')
    model.text_steps.add_(1)
    assert isinstance(model.generate_text('Hello', max_tokens=3), str)


def test_default_native_checkpoint_fits_target(tmp_path):
    model = UnifiedVoiceModel()
    size = model.save(tmp_path/'one.pt', fp16=True)
    assert size / 1024**2 < 40
    loaded = UnifiedVoiceModel.load(tmp_path/'one.pt')
    assert next(loaded.parameters()).dtype == torch.float16
    assert len(list(loaded.stream('Hello', max_frames=2, allow_untrained=True))) == 2


def test_video_split_keeps_validation_separate(tmp_path):
    import numpy as np
    from src.data.video_audio import split_audio
    audio = np.sin(np.arange(10000)*.1).astype('float32')
    segments = split_audio(audio, 1000, tmp_path, seconds=1, train_fraction=.8)
    training = [r for r in segments if r['split'] == 'train']
    validation = [r for r in segments if r['split'] == 'validation']
    assert len(training) == 8 and len(validation) == 2
    assert max(r['end_seconds'] for r in training) <= min(r['start_seconds'] for r in validation)
    with pytest.raises(ValueError):
        split_audio(audio, 1000, tmp_path/'bad', seconds=-1)


def test_training_crop_covers_the_whole_recording():
    from scripts.voice import training_crop
    audio = torch.arange(10000).reshape(1,-1).float()
    generator = torch.Generator().manual_seed(9)
    crops = [training_crop(audio, 1000, generator) for _ in range(30)]
    assert max(float(crop[0,0]) for crop in crops) > 8000
    assert min(float(crop[0,0]) for crop in crops) < 1000
    assert all(crop.shape == (1,1000) for crop in crops)
    assert training_crop(audio[:,:2], 5, generator).tolist() == [[0.,1.,0.,0.,0.]]


def test_codec_validation_does_not_mutate_training_mode():
    from scripts.voice import codec_validation
    model = tiny().train()
    loss = codec_validation(model, [torch.randn(1,1600)*.1])
    assert loss > 0 and model.training
    assert codec_validation(model, []) is None


def test_continuous_codec_warmup_updates_encoder_and_decoder_only():
    model=tiny()
    loss=model.codec_loss(torch.randn(1,1600)*.1,quantize=False)['loss']
    loss.backward()
    assert model.codec.encoder_in.weight.grad.abs().sum()>0
    assert model.codec.decoder_out.weight.grad.abs().sum()>0
    assert all(q.embedding.weight.grad is None for q in model.codec.rvq.quantizers)


def test_data_driven_codebooks_reduce_quantization_error():
    from scripts.improve_codec import initialize_codebooks
    model=tiny()
    waves=[torch.randn(1,3200)*.1 for _ in range(4)]
    with torch.no_grad():
        latent=model.codec.encode_latents(waves[0])
        before=model.codec.rvq(latent)[0]
        initial_error=(latent-before).square().mean()
        initialize_codebooks(model,waves,torch.Generator().manual_seed(2))
        after=model.codec.rvq(latent)[0]
        assert (latent-after).square().mean()<initial_error
        _,tokens,_=model.codec.encode(waves[0])
        assert torch.unique(tokens[:,0]).numel()>1


def test_codec_loss_handles_subframe_records_without_nan():
    model=tiny()
    for length in [1,319,320,321]:
        losses=model.codec_loss(torch.randn(1,length)*.1)
        assert all(torch.isfinite(value) for value in losses.values())


def test_short_frame_initialization_and_streaming():
    from scripts.upgrade_codec import initialize_short_frames
    model = UnifiedVoiceModel(UnifiedConfig(hop_length=8, codec_dim=12, codebooks=8,
        codebook_size=16, gain_bins=1, hidden_size=32, layers=2, heads=4, kv_heads=2,
        intermediate_size=48, max_seq_len=128)).eval()
    initialize_short_frames(model)
    wav = torch.randn(1, 41)*.02
    latent = model.codec.encode_latents(wav)
    continuous = model.codec.decode_latents(latent)[:, :41]
    torch.testing.assert_close(continuous, wav.tanh())
    _, tokens, _ = model.codec.encode(wav)
    history, chunks = None, []
    for index in range(tokens.shape[-1]):
        chunk, history = model.codec.decode_frame(tokens[..., index:index+1], history)
        chunks.append(chunk)
    torch.testing.assert_close(torch.cat(chunks, -1), model.codec.decode(tokens))
    assert model.codec.decode(tokens).count_nonzero() == 0


def test_evaluation_short_audio_and_empty_input():
    from scripts.improve_codec import evaluate, initialize_codebooks
    model = tiny()
    metrics = evaluate(model, [torch.randn(1, 1)*.1])
    assert metrics['boundary_error'] == 0
    with pytest.raises(ValueError, match='requires audio'):
        evaluate(model, [])
    with pytest.raises(ValueError, match='one frame per centroid'):
        initialize_codebooks(model, [torch.randn(1, 1)], torch.Generator())


def test_gain_codec_preserves_silence_and_streaming(tmp_path):
    model = UnifiedVoiceModel(UnifiedConfig(hop_length=8, codec_dim=12, codebooks=4,
        codebook_size=16, gain_bins=16, hidden_size=32, layers=2, heads=4,
        kv_heads=2, intermediate_size=48, max_seq_len=128)).eval()
    # Even a biased/noisy decoder cannot synthesize a nonzero sample for gain zero.
    _, tokens, _ = model.codec.encode(torch.zeros(1, 25))
    assert tokens.shape == (1, 4, 4)
    assert model.codec.decode(tokens).count_nonzero() == 0
    wav = torch.randn(1, 25)*.1
    _, tokens, _ = model.codec.encode(wav)
    history, chunks = None, []
    for index in range(tokens.shape[-1]):
        chunk, history = model.codec.decode_frame(tokens[..., index:index+1], history)
        chunks.append(chunk)
    torch.testing.assert_close(torch.cat(chunks, -1), model.codec.decode(tokens))
    for quantize in [True, False]:
        loss = model.codec_loss(wav, quantize=quantize)['loss']
        assert torch.isfinite(loss)
        model.zero_grad(set_to_none=True)
        loss.backward()
        assert model.codec.decoder_out.weight.grad.abs().sum() > 0
    model.save(tmp_path/'gain.pt', fp16=True)
    loaded = UnifiedVoiceModel.load(tmp_path/'gain.pt')
    assert loaded.config.gain_bins == 16
    assert loaded.codec.decode(torch.zeros(1, 4, 3, dtype=torch.long)).count_nonzero() == 0


def test_gain_zero_is_only_reserved_for_exact_silence():
    model = UnifiedVoiceModel(UnifiedConfig(hop_length=8, codec_dim=12, codebooks=4,
        codebook_size=16, gain_bins=16, hidden_size=32, layers=2, heads=4,
        kv_heads=2, intermediate_size=48, max_seq_len=128)).eval()
    assert model.codec.frame_gain_tokens(torch.full((1,8), 1e-6)).item() > 0
    with pytest.raises(ValueError, match='Gain coding'):
        UnifiedConfig(gain_bins=16)


def test_prepare_real_pairs_validates_before_writing(tmp_path):
    import json
    import numpy as np
    import soundfile as sf
    from scripts.prepare_pairs import prepare
    model = tiny()
    checkpoint = tmp_path/'voice.pt'
    model.save(checkpoint)
    wav = tmp_path/'source.wav'
    sf.write(wav, np.zeros(16000), 16000)
    labels = tmp_path/'labels.jsonl'
    rows = [{'start': 0., 'end': .2, 'text': 'Hello.', 'source': 'human_recording'},
            {'start': .8, 'end': 1., 'text': 'Goodbye.', 'source': 'human_recording'}]
    labels.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    records = prepare(wav, labels, tmp_path/'pairs', checkpoint)
    assert len(records['train']) == len(records['validation']) == 1
    assert sf.info(tmp_path/'pairs'/records['train'][0]['audio']).frames == 3200
    rows[0]['text'] = ''
    labels.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError, match='actual text'):
        prepare(wav, labels, tmp_path/'bad', checkpoint)
    assert not (tmp_path/'bad').exists()


def test_narration_rejects_untrained_voice_without_output(tmp_path, monkeypatch):
    import sys
    from scripts.voice import main
    checkpoint = tmp_path/'voice.pt'
    tiny().save(checkpoint)
    story = tmp_path/'story.txt'
    story.write_text('Bir gece, bir yolcu fener buldu.')
    output = tmp_path/'story.wav'
    monkeypatch.setattr(sys, 'argv', ['voice.py', 'narrate', '--checkpoint', str(checkpoint),
                                    '--text-file', str(story), '--output', str(output)])
    with pytest.raises(RuntimeError, match='No speech training'):
        main()
    assert not output.exists()


def test_legacy_checkpoint_without_gain_metadata_loads(tmp_path):
    model = tiny()
    path = tmp_path/'legacy.pt'
    model.save(path)
    checkpoint = torch.load(path, weights_only=True)
    del checkpoint['config']['gain_bins']
    del checkpoint['state']['codec.gain_levels']
    del checkpoint['state']['audio_steps']
    torch.save(checkpoint, path)
    loaded = UnifiedVoiceModel.load(path)
    assert loaded.config.gain_bins == 1
    assert loaded.audio_steps.item() == 0
    wav = torch.randn(1, 321)
    with torch.no_grad():
        _, tokens, _ = model.codec.encode(wav)
        torch.testing.assert_close(model.codec.decode(tokens), loaded.codec.decode(tokens))


def test_audio_pretraining_has_gradients_without_claiming_speech():
    model = tiny()
    losses = model.audio_loss(wav=torch.randn(1,1600)*.1)
    losses['loss'].backward()
    assert model.backbone.audio_embeddings[0].weight.grad.abs().sum()>0
    assert model.codec.encoder_in.weight.grad is None
    assert model.stop_head.weight.grad is None
    assert 0 <= losses['token_accuracy'] <= 1
    model.audio_steps.add_(1)
    model.eval()
    with pytest.raises(RuntimeError, match='No speech training'):
        list(model.stream('Hello, this is my voice.'))
    with pytest.raises(RuntimeError, match='Language training'):
        model.generate_text('Hello')


def test_audio_validation_preserves_mode_and_temporal_split_guard(tmp_path):
    import json
    from scripts.pretrain_audio import audio_validation, verify_split
    model = tiny().train()
    tokens = torch.randint(0,16,(1,2,5))
    metrics = audio_validation(model,[tokens])
    assert model.training and metrics['loss']>0
    train = tmp_path/'train'
    valid = tmp_path/'valid'
    train.mkdir(); valid.mkdir()
    (train/'provenance.json').write_text(json.dumps({'video_sha256':'same', 'segments':[
        {'split':'train', 'start_seconds':0, 'end_seconds':2}]}))
    (valid/'provenance.json').write_text(json.dumps({'video_sha256':'same', 'segments':[
        {'split':'validation', 'start_seconds':1, 'end_seconds':3}]}))
    with pytest.raises(ValueError, match='overlap'):
        verify_split(train,valid)


def test_codec_training_is_blocked_after_audio_pretraining():
    from types import SimpleNamespace
    from scripts.voice import train
    model = tiny()
    model.audio_steps.add_(1)
    with pytest.raises(ValueError, match='invalidate trained acoustic/speech tokens'):
        train(model, SimpleNamespace(stage='codec', seed=1, checkpoint=None))


def test_joint_frame_prediction_has_no_future_book_leakage():
    from dataclasses import replace
    model = UnifiedVoiceModel(replace(tiny().config, codebooks=4, depth_hidden=16))
    torch.nn.init.normal_(model.depth_out.weight, std=.1)
    hidden = torch.randn(1, 5, model.config.hidden_size)
    targets = torch.randint(0,16,(1,4,5))
    first = model.conditioned_audio_logits(hidden,targets)
    changed = targets.clone()
    changed[:,2] = (changed[:,2]+1)%16
    second = model.conditioned_audio_logits(hidden,changed)
    torch.testing.assert_close(first[:,:3],second[:,:3])
    assert not torch.allclose(first[:,3],second[:,3])
    padded = model.conditioned_audio_logits(hidden,targets,pad_to_multiple=8)
    torch.testing.assert_close(first,padded,atol=1e-6,rtol=1e-5)
    first.sum().backward()
    assert model.depth_out.weight.grad.abs().sum()>0
    assert model.depth_in.weight.grad.abs().sum()>0


def test_default_stream_is_deterministic_for_joint_native_model():
    from dataclasses import replace
    model = UnifiedVoiceModel(replace(tiny().config,depth_hidden=16)).eval()
    torch.manual_seed(1)
    a=torch.cat(list(model.stream('Hello.',max_frames=3,allow_untrained=True)),-1)
    torch.manual_seed(200)
    b=torch.cat(list(model.stream('Hello.',max_frames=3,allow_untrained=True)),-1)
    torch.testing.assert_close(a,b)


def test_joint_rollout_matches_teacher_logits_using_its_own_prefixes():
    from dataclasses import replace
    model=UnifiedVoiceModel(replace(tiny().config,codebooks=4,depth_hidden=16)).eval()
    torch.nn.init.normal_(model.depth_out.weight,std=.1)
    hidden=torch.randn(1,5,model.config.hidden_size)
    predicted=model.sample_audio_tokens(hidden)
    torch.testing.assert_close(predicted,model.sample_audio_tokens(hidden,pad_to_multiple=8))
    teacher=model.conditioned_audio_logits(hidden,predicted).argmax(-1)
    torch.testing.assert_close(predicted,teacher)
    sampled=model.sample_audio_tokens(hidden,temperature=.5)
    assert sampled.shape==predicted.shape
    assert sampled.min()>=0 and sampled.max()<16
    with pytest.raises(ValueError,match='temperature'):
        model.sample_audio_tokens(hidden,float('nan'))


def test_exposure_predictions_are_causal_and_do_not_replace_targets():
    from dataclasses import replace
    model=UnifiedVoiceModel(replace(tiny().config,depth_hidden=16)).eval()
    tokens=torch.randint(0,16,(1,2,5))
    first=model.predicted_training_tokens('Hello.',tokens)
    changed=tokens.clone();changed[:,:,3:]=(changed[:,:,3:]+1)%16
    second=model.predicted_training_tokens('Hello.',changed)
    torch.testing.assert_close(first[:,:,:4],second[:,:,:4])
    direct=model.speech_loss('Hello.',tokens=tokens)
    unchanged=model.speech_loss('Hello.',tokens=tokens,input_tokens=tokens)
    for key in direct:torch.testing.assert_close(direct[key],unchanged[key])
    loss=model.speech_loss('Hello.',tokens=tokens,input_tokens=first)['loss']
    loss.backward()
    assert torch.isfinite(loss) and model.backbone.text_embedding.weight.grad.abs().sum()>0
    with pytest.raises(ValueError,match='aligned'):
        model.speech_loss('Hello.',tokens=tokens,input_tokens=first[:,:,:3])


def test_coarse_priority_preserves_reported_unweighted_metrics():
    torch.manual_seed(42)
    from dataclasses import replace
    model=UnifiedVoiceModel(replace(tiny().config,codebooks=6,depth_hidden=16)).eval()
    tokens=torch.randint(0,16,(1,6,5))
    ordinary=model.speech_loss('Hello.',tokens=tokens)
    prioritized=model.speech_loss('Hello.',tokens=tokens,coarse_weight=4.)
    for key in ('audio','coarse_audio','token_accuracy','text','stop'):
        torch.testing.assert_close(ordinary[key],prioritized[key])
    assert not torch.allclose(ordinary['loss'],prioritized['loss'],rtol=1e-6,atol=1e-6)
    prioritized['loss'].backward()
    assert torch.isfinite(model.backbone.audio_embeddings[0].weight.grad).all()
    with pytest.raises(ValueError,match='Coarse'):
        model.speech_loss('Hello.',tokens=tokens,coarse_weight=float('nan'))


def test_speak_default_duration_respects_checkpoint_context(tmp_path,monkeypatch):
    import sys
    import soundfile as sf
    from scripts.voice import main
    model=tiny();model.speech_steps.fill_(1)
    model.stop_head.bias.data.fill_(10.)
    checkpoint=tmp_path/'native.pt';model.save(checkpoint,fp16=True)
    output=tmp_path/'speech.wav'
    monkeypatch.setattr(sys,'argv',['voice.py','speak','--checkpoint',str(checkpoint),
                                   '--text','Hello.','--output',str(output)])
    main()
    wav,rate=sf.read(output)
    assert len(wav)>0 and len(wav)/rate<=8.
