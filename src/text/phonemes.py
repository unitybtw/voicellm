"""Deterministic CMU dictionary lookup. This is data, not a pretrained model."""
from pathlib import Path
import re
import hashlib
import unicodedata
from functools import lru_cache
ROOT=Path(__file__).resolve().parents[2]/'data/pronunciation'

@lru_cache(maxsize=1)
def lexicon():
    dictionary={}
    for line in (ROOT/'cmudict.dict').read_text().splitlines():
        fields=line.split('#',1)[0].split()
        if len(fields)<2:continue
        word=re.sub(r'\(\d+\)$','',fields[0])
        dictionary.setdefault(word,fields[1:])
    return dictionary

class EnglishPhonemes:
    def __init__(self):
        self.symbols=['<pad>','<sil>']+(ROOT/'cmudict.symbols').read_text().split()
        self.indices={s:i for i,s in enumerate(self.symbols)}
        self.dictionary_sha256=hashlib.sha256((ROOT/'cmudict.dict').read_bytes()).hexdigest()

    def encode_with_metadata(self,text):
        if not isinstance(text,str) or not text.strip():raise ValueError('Nonempty English text required')
        text=unicodedata.normalize('NFKC',text).translate(str.maketrans({'’':"'",'‘':"'",'ʼ':"'",'…':'.'}))
        units=['<sil>'];unknown=[]
        words=re.findall(r"[a-z]+(?:'[a-z]+)*|\d+|[.,!?;:]",text.lower())
        for word in words:
            if re.fullmatch(r'[.,!?;:]',word):
                if units[-1]!='<sil>':units.append('<sil>')
                continue
            phones=lexicon().get(word)
            if phones is None:
                unknown.append(word);phones=[]
                # Explicit spelling fallback, never a hidden G2P model.
                for letter in word:
                    name={'0':'zero','1':'one','2':'two','3':'three','4':'four','5':'five','6':'six','7':'seven','8':'eight','9':'nine'}.get(letter,letter)
                    phones.extend(lexicon().get(name,['AH0']))
            units.extend(phones)
        if units[-1]!='<sil>':units.append('<sil>')
        if len(units)<3:raise ValueError('No pronounceable English words')
        return [self.indices[x] for x in units],{'unknown_words':unknown,'phones':units}

    def encode(self,text):return self.encode_with_metadata(text)[0]
