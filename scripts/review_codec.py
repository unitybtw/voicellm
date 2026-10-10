"""Reproducible listening samples and full-length held-out codec evaluation."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
import torch
from scripts.voice import read_audio
from scripts.improve_codec import evaluate
from src.models.voice_unified import UnifiedVoiceModel
from src.data.video_audio import file_sha256


@torch.inference_mode()
def run(version='v4'):
    torch.set_num_threads(4)
    checkpoint = Path('models/checkpoints/voice_unified.pt')
    model = UnifiedVoiceModel.load(checkpoint).float()
    directory = Path('data/native_samples')
    validation = sorted(Path('data/training/video_native_v1/validation').glob('*.wav'))
    if not validation:
        raise ValueError('Held-out audio missing')
    full_validation = evaluate(model, [read_audio(p, model.config.sample_rate) for p in validation])
    sources = {'familiar': Path('data/native_samples/video_original_2_to_8s.wav'),
               'heldout': validation[0]}
    samples = {}
    for label, source in sources.items():
        original = read_audio(source, model.config.sample_rate)
        _, tokens, _ = model.codec.encode(original)
        decoded = model.codec.decode(tokens)[:, :original.shape[-1]]
        if not torch.isfinite(decoded).all():
            raise FloatingPointError('Invalid decoded audio')
        # Identical gain across original and model, no denoising or mixing original audio.
        gain = min(3., .8/max(float(original.abs().max()), float(decoded.abs().max()), 1e-6))
        raw = directory/f'{label}_codec_{version}_raw.wav'
        listen = directory/f'{label}_codec_{version}_listen.wav'
        compare = directory/f'{label}_original_then_codec_{version}.wav'
        sf.write(raw, decoded[0].numpy(), model.config.sample_rate, subtype='FLOAT')
        sf.write(listen, (decoded[0]*gain).numpy(), model.config.sample_rate, subtype='PCM_16')
        joined = torch.cat([original[0]*gain, torch.zeros(model.config.sample_rate), decoded[0]*gain])
        sf.write(compare, joined.numpy(), model.config.sample_rate, subtype='PCM_16')
        error = float((decoded-original).square().mean())
        energy = float(original.square().mean())
        samples[label] = {'source': str(source.resolve()), 'raw': str(raw.resolve()),
            'listen': str(listen.resolve()), 'comparison': str(compare.resolve()),
            'seconds': original.shape[-1]/model.config.sample_rate, 'common_gain': gain,
            'correlation': float(torch.nn.functional.cosine_similarity(original, decoded)),
            'normalized_mse': error/max(energy, 1e-9),
            'snr_db': float(10*np.log10(max(energy,1e-12)/max(error,1e-12))),
            'clipped_samples': int((decoded.abs() >= 1).sum()),
            'dc_mean': float(decoded.mean())}
        for path in [raw, listen, compare]:
            audio, rate = sf.read(path)
            if rate != model.config.sample_rate or not np.isfinite(audio).all():
                raise ValueError(f'Invalid output file: {path}')
    report = {'checkpoint_sha256': file_sha256(checkpoint), 'codec_steps': int(model.codec_steps),
              'audio_steps': int(model.audio_steps), 'speech_steps': int(model.speech_steps),
              'config': model.config.__dict__, 'full_heldout': full_validation,
              'validation_files': len(validation), 'validation_seconds': sum(sf.info(p).duration for p in validation),
              'samples': samples, 'human_listening_verified': False, 'external_weights': False,
              'note': 'Stored FP16 weights cast to FP32 for reconstruction. Codec reconstruction only, not new-text speech.'}
    Path(f'reports/codec_{version}_listening.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
