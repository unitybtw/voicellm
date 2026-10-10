"""Deterministic CMU dictionary lookup and Turkish phonetic routing. This is data, not a pretrained model."""
from pathlib import Path
import re
import hashlib
import unicodedata
from functools import lru_cache
from src.text.turkish_phonemes import TurkishPhonemes

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

def is_turkish_text(text: str) -> bool:
    """Accurately detects whether text is Turkish or English."""
    if re.search(r'[çğışöüÇĞİŞÖÜ]', text):
        return True
    words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)*", text.lower())
    if not words:
        return False
    if any(re.search(r'[qwx]', w) for w in words):
        return False
    lex = lexicon()
    in_english = sum(1 for w in words if w in lex)
    return (in_english / len(words)) < 0.5

class EnglishPhonemes:
    def __init__(self):
        self.symbols=['<pad>','<sil>']+(ROOT/'cmudict.symbols').read_text().split()
        self.indices={s:i for i,s in enumerate(self.symbols)}
        self.dictionary_sha256=hashlib.sha256((ROOT/'cmudict.dict').read_bytes()).hexdigest()
        self._turkish = TurkishPhonemes()

    def encode_with_metadata(self,text,lang=None):
        if not isinstance(text,str) or not text.strip():raise ValueError('Nonempty English text required')
        is_tr = (lang == 'tr') or (lang != 'en' and is_turkish_text(text))
        if is_tr:
            return self._turkish.encode_with_metadata(text)
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

UnifiedPhonemes = EnglishPhonemes

