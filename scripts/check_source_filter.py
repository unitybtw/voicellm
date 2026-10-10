"""Seeded DSP mechanism checks; these are not text-to-speech quality evaluations.

No checkpoint is loaded and no training runs. Human recordings, when available,
are analyzed only to isolate spectrum/pitch labels from neural prediction errors.
All synthesized waveforms remain in memory; they are never presented as human
references or evidence of intelligible speech.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.voice import read_audio
from src.codec.linear_spectrum import LinearSpectrum
from src.codec.pitch import pitch_labels


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prior_gain(probability, mode):
    """Independent replay of the two documented periodic-prior coefficients."""
    if mode == 'probability':
        return probability
    if mode == 'confidence':
        return max(0., 2 * probability - 1)
    raise ValueError('Unknown periodic prior mode')


def apply_comb(features, controls, codec, mode):
    # This deliberately replays the equation independently of the learned model.
    f0 = (controls[..., 0].clamp(-2.1, 1.4) * .5 + math.log(200.)).exp()
    frequency = torch.linspace(0, codec.sample_rate / 2, features.shape[-1])
    comb = (.08 + .92 * torch.exp(-8 * torch.sin(
        math.pi * frequency[None, None] / f0[..., None]).square())).log()
    probability = controls[..., 1:].sigmoid()
    gain = probability if mode == 'probability' else (2 * probability - 1).clamp_min(0)
    return features + comb * gain / codec.std.float().clamp_min(.3)


def wave_metrics(wave, reference, codec):
    output = wave.flatten().float()
    reference = reference.flatten().float()
    _, voice = pitch_labels(output, codec.n_fft, codec.hop_length, codec.sample_rate)
    _, reference_voice = pitch_labels(reference, codec.n_fft, codec.hop_length, codec.sample_rate)
    # Ignore padded frame edges and the decoder's fade in the mechanism checks.
    interior_voice = voice[:, 4:-4] if voice.shape[1] > 8 else voice
    interior_reference_voice = reference_voice[:, 4:-4] if reference_voice.shape[1] > 8 else reference_voice
    length = min(output.numel(), reference.numel())
    margin = min(codec.n_fft, length // 4)
    output_rms = output[margin:length - margin].square().mean().sqrt()
    reference_rms = reference[margin:length - margin].square().mean().sqrt()
    return {
        'seconds': output.numel() / codec.sample_rate,
        'finite': bool(torch.isfinite(output).all()),
        'interior_rms': float(output_rms),
        'reference_interior_rms': float(reference_rms),
        'interior_rms_ratio': float(output_rms / reference_rms.clamp_min(1e-12)),
        'interior_waveform_voiced_fraction': float(interior_voice.mean()),
        'reference_interior_voiced_fraction': float(interior_reference_voice.mean()),
        'clipped_sample_fraction': float((output.abs() >= .999).float().mean()),
    }


@torch.inference_mode()
def synthetic_checks(codec, seed):
    reference = .03 * torch.randn(3 * codec.sample_rate,
        generator=torch.Generator().manual_seed(seed))
    features = codec.encode(reference)[0]
    cases = []
    for probability in (.3, .4, .5):
        controls = torch.zeros(1, features.shape[1], 2)
        controls[..., 1] = math.log(probability / (1 - probability))
        modes = {}
        for mode in ('probability', 'confidence'):
            adjusted = apply_comb(features, controls, codec, mode)
            output = codec.decode_source_filter(adjusted, controls)
            modes[mode] = {
                'periodic_prior_gain': prior_gain(probability, mode),
                'max_feature_change_from_raw': float((adjusted - features).abs().max()),
                **wave_metrics(output, reference, codec),
            }
        cases.append({'voicing_probability': probability, 'f0_hz': 200., 'modes': modes})
    checks = {
        'confidence_prior_is_identity_at_or_below_half': all(
            case['modes']['confidence']['max_feature_change_from_raw'] == 0 for case in cases),
        'low_confidence_waveform_voiced_fraction_below_point_one': all(
            case['modes']['confidence']['interior_waveform_voiced_fraction'] < .1
            for case in cases if case['voicing_probability'] < .5),
        'low_confidence_interior_rms_ratio_between_point_eight_and_one_point_two': all(
            .8 < case['modes']['confidence']['interior_rms_ratio'] < 1.2
            for case in cases if case['voicing_probability'] < .5),
        'both_prior_coefficients_equal_one_at_full_voicing':
            prior_gain(1., 'probability') == prior_gain(1., 'confidence') == 1.,
        'all_waveforms_finite': all(
            mode['finite'] for case in cases for mode in case['modes'].values()),
    }
    return {
        'reference_kind': 'Seeded synthetic white noise; not a speech recording.',
        'seed': seed,
        'reference_samples': reference.numel(),
        'prior_application': 'Independent replay of decode_hidden comb equation; no neural model used.',
        'cases': cases,
        'checks': checks,
        'passed': all(checks.values()),
        'half_probability_note': 'q=.5 removes the confidence comb prior but is classified as voiced by the DSP head threshold; a low waveform-VAD assertion is deliberately not applied there.',
    }


@torch.inference_mode()
def public_isolation(codec):
    records = {}
    manifests = {}
    for split in ('train', 'validation'):
        path = ROOT / 'data/public/ljspeech_spectral_v1' / (split + '.jsonl')
        if not path.exists():
            return {'available': False, 'reason': 'Public manifests missing; nothing downloaded.'}
        manifests[split] = digest(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['id'] in ('LJ012-0259', 'LJ002-0007'):
                records[row['id']] = (split, row)
    examples = []
    public_root = (ROOT / 'data/public/LJSpeech-1.1/wavs').resolve()
    for identifier in ('LJ012-0259', 'LJ002-0007'):
        if identifier not in records:
            return {'available': False, 'reason': 'Expected public example missing; nothing downloaded.'}
        split, row = records[identifier]
        path = Path(row['audio']).resolve()
        if (row.get('dataset') != 'LJSpeech-1.1' or row.get('source') != 'human_recording'
            or not path.is_relative_to(public_root)):
            raise ValueError('Only publisher-paired public human recordings are allowed')
        if not path.exists():
            return {'available': False, 'reason': 'Public recording missing; nothing downloaded.'}
        before = digest(path)
        reference = read_audio(path, codec.sample_rate)
        features = codec.encode(reference)[0]
        f0, voice = pitch_labels(reference, codec.n_fft, codec.hop_length, codec.sample_rate)
        controls = torch.stack((f0, torch.where(voice > 0,
            torch.full_like(voice, 15.), torch.full_like(voice, -15.))), -1)
        output = codec.decode_source_filter(features, controls)
        if digest(path) != before:
            raise RuntimeError('Human recording changed during the isolation check')
        examples.append({
            'id': identifier,
            'split': split,
            'text': row['text'],
            'human_reference_path': str(path),
            'human_reference_sha256': before,
            'spectrum_source': 'Spectrum of the public human reference; no neural prediction.',
            'controls_source': 'Native autocorrelation F0/voicing estimates from that same human reference; these are not human timestamp/pitch labels.',
            'output_kind': 'DSP reconstruction in memory; not text-generated speech and not a human reference.',
            **wave_metrics(output, reference, codec),
        })
    return {'available': True, 'manifest_sha256': manifests, 'examples': examples}


def run(output, seed=83, include_public=True):
    torch.set_num_threads(4)
    source_paths = [Path(__file__), ROOT / 'src/codec/linear_spectrum.py',
        ROOT / 'src/codec/pitch.py', ROOT / 'src/models/aligned_voice.py']
    source_sha = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    codec = LinearSpectrum()
    synthetic = synthetic_checks(codec, seed)
    report = {
        'purpose': 'Reproducible DSP energy/periodicity mechanism regression; no speech quality claim.',
        'sample_rate': codec.sample_rate,
        'n_fft': codec.n_fft,
        'hop_length': codec.hop_length,
        'dsp_decoder': 'Current local source-filter decoder with a voiced/unvoiced class decision, Schroeder phase dispersion, interpolated equal-power transitions, and pitch-smoothed aperiodic noise.',
        'checkpoint_loaded': False,
        'training_performed': False,
        'external_weights': False,
        'user_audio_used': False,
        'audio_files_written': False,
        'intelligibility_verified': False,
        'human_listening_verified': False,
        'source_sha256': source_sha,
        'synthetic': synthetic,
        'public_controls_isolation': public_isolation(codec) if include_public else {'available': False, 'reason': 'Explicitly skipped.'},
        'limitations': 'RMS and autocorrelation VAD quantify DSP mechanics only. They do not establish correct words, naturalness, text generalization, or model readiness. Public controls checks reconstruct recorded spectra and are never labeled TTS output.',
    }
    if any(digest(path) != source_sha[str(path.relative_to(ROOT))] for path in source_paths):
        raise RuntimeError('DSP/model source changed during the check; rerun on a stable source snapshot')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'report': str(output.resolve()), 'mechanism_checks_passed': synthetic['passed'],
        'public_isolation_available': report['public_controls_isolation']['available']}, indent=2))
    if not synthetic['passed']:
        raise SystemExit(1)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/dsp_regression_comparison.json')
    parser.add_argument('--seed', type=int, default=83)
    parser.add_argument('--skip-public-examples', action='store_true')
    arguments = parser.parse_args()
    run(arguments.output, arguments.seed, not arguments.skip_public_examples)
