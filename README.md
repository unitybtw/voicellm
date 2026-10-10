# VoiceLLM — local speech synthesis research

Experimental speech synthesis in PyTorch, trained from scratch on public human recordings. No pretrained model weights, distillation, or cloud inference are used. **Intelligible speech has not been validated.** This is research code, not a finished voice assistant.

## Installation

The source code is MIT licensed. The CMU pronunciation dictionary retains its separate license in `data/pronunciation/LICENSE`. Personal recordings, training datasets, checkpoints, optimizer states, and local experiment reports are excluded from Git. No validated downloadable checkpoint has been released. References to `reports/` below describe local artifacts that are absent from a clean clone.

Python 3.11 or newer is recommended. Run commands from the repository root:

```bash
git clone https://github.com/unitybtw/voicellm.git
cd voicellm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests require neither downloaded training recordings nor trained checkpoints. To start training, download the official [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/) separately. Allow several gigabytes of storage and substantial compute time:

```bash
python scripts/download_public.py
python scripts/prepare_public.py
mkdir -p models/checkpoints reports
python scripts/train_public.py --device cpu
```

Apple Silicon systems with a compatible PyTorch installation can use `--device mps`. The initial training pipeline creates an active checkpoint only when its acceptance conditions pass. `train_spectral.py` and `train_aligned.py` require an existing verified local checkpoint and matching metadata; they do not run directly from a clean clone. Completing training does not establish intelligible speech.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for data provenance and licensing.

## Quality status

The latest user evaluation reported noise rather than intelligible speech. Nonzero output, lower spectral error, increased periodicity, and passing code tests do not establish speech quality. `speech_ready=false`.

The canonical active checkpoint is `models/checkpoints/voice_unified.pt`. Development candidates are kept separately in `models/checkpoints/voice_aligned_candidate.pt`. Learning a small training subset is insufficient for promotion: validation performance must also improve and beat a constant-mean spectral baseline. The checkpoint's `.training.json` sidecar records its type, step count, and checksum.

Training uses publisher-paired English recordings and transcripts from LJSpeech. Personal voice/video recordings are excluded from the current training pipeline. There is no validated support for Turkish speech, natural pronunciation, frontier quality, or superiority over competing models.

## Model and synthesis

One native network learns phoneme context, phoneme durations, full FFT spectra, pitch, and voicing. It avoids repeatedly feeding the previous acoustic frame back into generation. The aligned network has 14,670,213 parameters; a local FP16 checkpoint is approximately 28.02 MiB. These figures describe size, not quality.

English phonemes come from the [CMU pronunciation dictionary](https://github.com/cmusphinx/cmudict), a pronunciation data resource rather than pretrained weights. Its pinned revision, checksums, provenance, and [license](https://github.com/cmusphinx/cmudict/blob/master/LICENSE) are retained in `data/pronunciation/`. Unknown words use an explicit spelling fallback. Checkpoint loading verifies the dictionary checksum and phoneme inventory. Typographic apostrophes are normalized so English contractions remain intact.

Native dynamic programming aligns phoneme features with real target spectra while preserving order. Coarse frequency features guide alignment, and short phonemes cannot be skipped. This follows the [monotonic alignment research approach](https://arxiv.org/abs/2005.11129) without importing an external aligner's code or weights. Inferred paths are not human timestamp annotations.

A local DSP source/filter decoder uses the network's pitch, voicing, and spectrum predictions. Its aperiodic branch uses filtered random-phase noise, with spectral power smoothed over one pitch spacing so it does not inherit narrow harmonic lines. The voiced/unvoiced decision follows the binary classifier rather than interpreting its probability as harmonic energy. Decisions are interpolated between frames for smooth equal-power transitions. There is no separate learned vocoder, and synthesized output never copies or mixes a training waveform. Ten-millisecond fades reduce edge discontinuities. Harmonic counts follow Nyquist and minimum pitch rather than a fixed 64-harmonic limit. Noise energy is calibrated to predicted frame RMS. Output is synthesized in blocks before chunks are yielded; this does not establish live streaming latency.

The experimental `periodic_prior_mode=confidence` suppresses harmonic spectral constraints when the voiced classifier is uncertain or predicts an unvoiced frame. Earlier checkpoints preserve their `probability` behavior. The new mode requires retraining; candidate selection compares against the source checkpoint in its original mode.

Pitch and voicing learn from native autocorrelation labels derived from recordings. Spectral comb gradients are detached from the pitch head to avoid a fixed-pitch local minimum. Out-of-range monotonic correlation slopes are rejected as pitch peaks. These estimates are not human pitch annotations.

## Training and evaluation

Development started with an explicit 16-record memorization check and expanded to 128 and 256 training recordings. Recent runs use 32 validation and 32 test recordings. Neither split supplies gradient training; test metrics do not select checkpoints. These are subsets of the available corpus, not evidence that the full dataset has been trained.

`clock_mode=soft` directly supervises synthesis using the model's own differentiable duration estimates. Duration training and inference use the same minimum frame count. Explicitly requested output lengths preserve every phoneme and allocate the exact number of frames. Cached alignment paths are validated for shape, type, endpoints, and consecutive phoneme coverage.

Training includes frame energy alongside spectral, pitch, voicing, duration, and temporal-change losses. Candidate selection measures spectrum, duration, pitch, voicing, energy, and acoustic variation on validation data. A worse run cannot overwrite its verified starting candidate. Native checkpoints and optimizer files use atomic writes, and resume checks verify their matching checksums and step counts.

Every evaluation interval (100 steps), and the final step, saves a resumable last-attempt checkpoint with its matching optimizer and metadata, even when it is not selected or promoted. This limits lost work after interruptions without treating an experiment as accepted speech. Nonfinite losses or gradients cannot update weights, optimizer moments, or the successful-step counter. Training records are shuffled once per epoch; the saved order and cursor continue across resumes with the same record subset. Changing the subset starts a fresh, explicitly reported order. Metadata names the matching sibling optimizer file, so absolute checkpoint paths and archived training states can be resumed safely.

The following commands require verified local training artifacts:

```bash
# Resume a native development candidate with its matching optimizer.
python3 scripts/train_aligned.py --checkpoint models/checkpoints/voice_aligned_candidate.pt --steps 600 --clips 256 --min-seconds 1.5 --max-seconds 6 --soft-clock --confident-periodic-prior --device mps --lr .00005
python3 scripts/review_aligned.py --checkpoint models/checkpoints/voice_aligned_candidate.pt
python3 scripts/voice.py speak --checkpoint models/checkpoints/voice_aligned_candidate.pt --text "Hello world." --temperature 0
# Continue a resumable experimental last attempt created by the updated trainer.
python3 scripts/train_aligned.py --checkpoint models/checkpoints/voice_aligned_last_attempt.pt --steps 600 --clips 256 --min-seconds 1.5 --max-seconds 6 --soft-clock --device mps --lr .00005
```

Reproducible code and DSP checks:

```bash
python3 -m pytest -q
python3 -m compileall -q src scripts tests
python3 scripts/check_source_filter.py --skip-public-examples
```

The DSP checks isolate energy and periodicity mechanics. Public-control reconstruction checks, when recordings are available, reconstruct measured spectra and are explicitly distinct from new-text synthesis. Numerical metrics do not verify pronunciation or intelligibility.

Local run histories record source checksums, training records, attempted and selected steps, configuration changes, and measurements. Current local reports include `reports/aligned_training_latest.json`, `reports/generation_aligned_review.json`, `reports/dsp_regression_comparison.json`, and `reports/noise_development_latest.json`.
