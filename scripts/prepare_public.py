"""Prepare publisher-aligned LJSpeech pairs; never load a pretrained model."""
import hashlib
import json
import math
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import soundfile as sf
from src.models.voice_unified import ByteTokenizer, UnifiedConfig
from src.data.video_audio import file_sha256


def prepare(root, output, archive=None, config=None):
    root, output = Path(root).resolve(), Path(output)
    output.mkdir(parents=True,exist_ok=True)
    config=config or UnifiedConfig()
    tokenizer=ByteTokenizer()
    splits={'train':[],'validation':[],'test':[]}
    excluded={'context':0,'duplicate_text':0}
    seen=set()
    for line in (root/'metadata.csv').read_text().splitlines():
        identifier, raw, text=line.split('|',2)
        if not re.fullmatch(r'LJ\d{3}-\d{4}',identifier) or not text.strip():
            raise ValueError('Invalid LJSpeech metadata')
        audio=root/'wavs'/f'{identifier}.wav'
        info=sf.info(audio)
        frames=math.ceil(math.ceil(info.frames*config.sample_rate/info.samplerate)/config.hop_length)+1
        if len(tokenizer.encode(text))+frames+1>config.max_seq_len:
            excluded['context']+=1
            continue
        canonical=' '.join(text.lower().split())
        if canonical in seen:
            excluded['duplicate_text']+=1
            continue
        seen.add(canonical)
        bucket=int(hashlib.sha256(canonical.encode()).hexdigest()[:8],16)%100
        split='test' if bucket<5 else 'validation' if bucket<10 else 'train'
        splits[split].append({'id':identifier,'audio':str(audio),'text':text,
            'source':'human_recording','language':'en','dataset':'LJSpeech-1.1',
            'duration_seconds':info.duration})
    for split, rows in splits.items():
        if not rows:
            raise ValueError('Missing dataset split')
        (output/f'{split}.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    report={'dataset':'LJSpeech-1.1','source':'https://keithito.com/LJ-Speech-Dataset/',
        'download':'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2',
        'archive_sha256':file_sha256(archive) if archive else None,
        'metadata_sha256':file_sha256(root/'metadata.csv'),
        'license_as_declared_by_publisher':'Public domain',
        'language':'en','speaker':'LJSpeech public dataset speaker; not the user',
        'counts':{key:len(value) for key,value in splits.items()},
        'hours':{key:sum(row['duration_seconds'] for row in value)/3600 for key,value in splits.items()},
        'excluded':excluded,'external_weights':False,'user_audio_used':False,
        'split':'Text-hash 90/5/5; duplicates removed; same speaker/books, not unseen-speaker evaluation',
        'config':config.__dict__}
    (output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    print(json.dumps(prepare('data/public/LJSpeech-1.1','data/public/ljspeech_v1',
                            'data/public/downloads/LJSpeech-1.1.tar.bz2'),indent=2))
