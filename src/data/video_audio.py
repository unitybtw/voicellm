"""Local video-to-audio preparation with an auditable chronological split."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import soundfile as sf


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def split_audio(audio, sr, output_dir, seconds=4., train_fraction=.8, min_rms=.005):
    if audio.ndim != 1 or len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError('Expected nonempty finite mono audio')
    if sr <= 0 or seconds <= 0 or not 0 < train_fraction < 1:
        raise ValueError('Invalid split settings')
    output_dir = Path(output_dir)
    hop = int(sr*seconds)
    if hop < 1:
        raise ValueError('Segment length must include at least one sample')
    boundary = int(len(audio)*train_fraction)
    records = []
    for start in range(0, len(audio)-hop+1, hop):
        end = start + hop
        if start < boundary < end:
            continue
        segment = audio[start:end]
        rms = float(np.sqrt(np.mean(segment.astype(np.float64)**2)))
        if rms < min_rms:
            continue
        split = 'train' if end <= boundary else 'validation'
        path = output_dir/split/f'video_{start/sr:09.3f}.wav'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f'Use a new preparation directory: {path}')
        sf.write(path, segment, sr, subtype='PCM_16')
        records.append({'audio': str(path.resolve()), 'start_seconds': start/sr,
                        'end_seconds': end/sr, 'split': split, 'rms': rms})
    if not any(r['split']=='train' for r in records) or not any(r['split']=='validation' for r in records):
        raise ValueError('Not enough active audio for both training and validation')
    return records


def prepare_video(video, output_dir, sr=16000, seconds=4.):
    video, output_dir = Path(video).resolve(), Path(output_dir).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    # Keep the previous preparation intact instead of mixing old/new segments.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Video output directory must be empty')
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = output_dir/'video_audio.wav'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-i', str(video),
                    '-map', '0:a:0', '-vn', '-ac', '1', '-ar', str(sr), str(extracted)], check=True)
    audio, actual_sr = sf.read(extracted, dtype='float32')
    records = split_audio(audio, actual_sr, output_dir, seconds=seconds)
    report = {'source_video': str(video), 'video_sha256': file_sha256(video),
              'extracted_audio_sha256': file_sha256(extracted),
              'sample_rate': actual_sr, 'duration_seconds': len(audio)/actual_sr,
              'segment_seconds': seconds,
              'split': 'chronological 80/20; segments crossing split boundary discarded',
              'speaker_verified': False, 'note': 'Energy filtering does not identify a speaker or exclude music/other speakers.',
              'segments': records}
    (output_dir/'provenance.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    return report
