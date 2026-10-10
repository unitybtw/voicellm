"""Make real audio/transcript pairs from human-supplied, timed text. No ASR."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import soundfile as sf
from scripts.voice import read_audio
from src.data.video_audio import file_sha256
from src.models.voice_unified import UnifiedVoiceModel


def prepare(audio_path, transcripts_path, output, checkpoint):
    model = UnifiedVoiceModel.load(checkpoint)
    wav = read_audio(audio_path, model.config.sample_rate)[0].numpy()
    rows = [json.loads(line) for line in Path(transcripts_path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('Provide a nonempty, human-verified transcript')
    prepared = []
    for row in rows:
        start, end = float(row['start']), float(row['end'])
        text = row.get('text', '')
        if not isinstance(text, str) or not text.strip() or row.get('source') != 'human_recording':
            raise ValueError('Each segment requires actual text and source=human_recording')
        if not math.isfinite(start+end) or not 0 <= start < end <= len(wav)/model.config.sample_rate:
            raise ValueError('Invalid audio segment times')
        left, right = round(start*model.config.sample_rate), round(end*model.config.sample_rate)
        length = len(model.tokenizer.encode(text))+math.ceil((right-left)/model.config.hop_length)+1
        if length > model.config.max_seq_len:
            raise ValueError('Segment and text exceed model context; split at matching sentence boundaries')
        cutoff = len(wav)*.8
        if left < cutoff < right:
            raise ValueError('Segment crosses chronological train/validation split')
        prepared.append((left, right, text, 'train' if right <= cutoff else 'validation'))
    prepared.sort()
    if any(a[1] > b[0] for a,b in zip(prepared, prepared[1:])):
        raise ValueError('Transcript segments overlap')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory')
    (output/'clips').mkdir(parents=True, exist_ok=True)
    manifests = {'train': [], 'validation': []}
    for index, (left, right, text, split) in enumerate(prepared):
        name = f'clips/pair_{index:05d}.wav'
        sf.write(output/name, wav[left:right], model.config.sample_rate, subtype='FLOAT')
        manifests[split].append({'audio': name, 'text': text, 'source': 'human_recording',
                                'start': left/model.config.sample_rate, 'end': right/model.config.sample_rate})
    for split, records in manifests.items():
        (output/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    (output/'provenance.json').write_text(json.dumps({
        'audio': str(Path(audio_path).resolve()), 'audio_sha256': file_sha256(audio_path),
        'transcript_sha256': file_sha256(transcripts_path), 'external_transcription': False,
        'train_pairs': len(manifests['train']), 'validation_pairs': len(manifests['validation']),
        'note': 'Transcript correctness and speaker identity must come from the supplied human annotations.'
    }, indent=2)+'\n')
    return manifests


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--transcripts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('models/checkpoints/voice_unified.pt'))
    args = parser.parse_args()
    try:
        records = prepare(args.audio, args.transcripts, args.output, args.checkpoint)
        print({split: len(rows) for split,rows in records.items()})
    except (ValueError, KeyError) as error:
        raise SystemExit(str(error)) from None
