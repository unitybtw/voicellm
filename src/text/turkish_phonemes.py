"""Deterministic Turkish phonetic orthography and pronunciation rules.

Maps Turkish text into the native phonetic symbol inventory without external
dependencies or pretrained G2P models. Follows the standard phonetic rules of
the Turkish language.
"""
from pathlib import Path
import re
import hashlib
import unicodedata
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[2] / 'data/pronunciation'

TURKISH_LETTER_MAP = {
    'a': 'AA1', 'b': 'B', 'c': 'JH', 'ç': 'CH', 'd': 'D',
    'e': 'EH1', 'f': 'F', 'g': 'G', 'h': 'HH', 'ı': 'IH1',
    'i': 'IY1', 'j': 'ZH', 'k': 'K', 'l': 'L', 'm': 'M',
    'n': 'N', 'o': 'AO1', 'ö': 'ER1', 'p': 'P', 'r': 'R',
    's': 'S', 'ş': 'SH', 't': 'T', 'u': 'UW1', 'ü': 'UW1',
    'v': 'V', 'y': 'Y', 'z': 'Z'
}

FRONT_VOWELS = set('eiöü')
BACK_VOWELS = set('aıou')
ALL_VOWELS = FRONT_VOWELS | BACK_VOWELS

TURKISH_DIGITS_ONES = ['', 'bir', 'iki', 'üç', 'dört', 'beş', 'altı', 'yedi', 'sekiz', 'dokuz']
TURKISH_DIGITS_TENS = ['', 'on', 'yirmi', 'otuz', 'kırk', 'elli', 'altmış', 'yetmiş', 'seksen', 'doksan']


def expand_turkish_number(number: int) -> str:
    """Expands an integer into its standard Turkish word representation."""
    if number == 0:
        return 'sıfır'
    if number < 0:
        return 'eksi ' + expand_turkish_number(-number)

    def under_thousand(num: int) -> str:
        tokens = []
        h = num // 100
        rem = num % 100
        if h > 1:
            tokens.append(TURKISH_DIGITS_ONES[h] + ' yüz')
        elif h == 1:
            tokens.append('yüz')
        t = rem // 10
        o = rem % 10
        if t > 0:
            tokens.append(TURKISH_DIGITS_TENS[t])
        if o > 0:
            tokens.append(TURKISH_DIGITS_ONES[o])
        return ' '.join(tokens)

    parts = []
    if number >= 1_000_000_000:
        b = number // 1_000_000_000
        parts.append(under_thousand(b) + ' milyar')
        number %= 1_000_000_000
    if number >= 1_000_000:
        m = number // 1_000_000
        parts.append(under_thousand(m) + ' milyon')
        number %= 1_000_000
    if number >= 1_000:
        k = number // 1_000
        if k == 1:
            parts.append('bin')
        else:
            parts.append(under_thousand(k) + ' bin')
        number %= 1_000
    if number > 0:
        parts.append(under_thousand(number))
    return ' '.join(parts)


def normalize_turkish_text(text: str) -> str:
    """Normalizes Turkish text respecting dotted/dotless I, circumflexes and numbers."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Nonempty text required')
    text = unicodedata.normalize('NFKC', text)
    # Correct Turkish case-folding: I -> ı, İ -> i
    # Standard Python .lower() erroneously converts I to i without locale.
    char_map = {
        'İ': 'i', 'I': 'ı', 'â': 'a', 'Â': 'a',
        'î': 'i', 'Î': 'i', 'û': 'u', 'Û': 'u',
        '’': "'", '‘': "'", 'ʼ': "'", '…': '.'
    }
    for orig, repl in char_map.items():
        text = text.replace(orig, repl)
    text = text.lower()

    # Expand digits into Turkish words
    def replace_num(match):
        num_str = match.group(0)
        try:
            return ' ' + expand_turkish_number(int(num_str)) + ' '
        except ValueError:
            return num_str

    text = re.sub(r'\d+', replace_num, text)
    return text


def turkish_word_to_phones(word: str) -> list[str]:
    """Converts a normalized Turkish word into phonetic symbols.

    Applies soft-g ('ğ') vowel lengthening and glide rules, as well as
    consonant gemination.
    """
    chars = list(word)
    phones = []
    for i, ch in enumerate(chars):
        if ch == 'ğ':
            # Yumuşak G: lengthens previous vowel or functions as palatal glide
            prev_vowel = None
            for j in range(i - 1, -1, -1):
                if chars[j] in ALL_VOWELS:
                    prev_vowel = chars[j]
                    break
            next_vowel = None
            for j in range(i + 1, len(chars)):
                if chars[j] in ALL_VOWELS:
                    next_vowel = chars[j]
                    break
            if prev_vowel in FRONT_VOWELS:
                if next_vowel in FRONT_VOWELS:
                    phones.append('Y')
                else:
                    phones.append(TURKISH_LETTER_MAP[prev_vowel])
            elif prev_vowel in BACK_VOWELS:
                phones.append(TURKISH_LETTER_MAP[prev_vowel])
            else:
                phones.append('Y')
        elif ch in TURKISH_LETTER_MAP:
            phones.append(TURKISH_LETTER_MAP[ch])
    return phones


class TurkishPhonemes:
    """Phonetic encoder for the Turkish language with 100% symbol-inventory compatibility."""

    def __init__(self):
        self.symbols = ['<pad>', '<sil>'] + (ROOT / 'cmudict.symbols').read_text().split()
        self.indices = {s: i for i, s in enumerate(self.symbols)}
        self.dictionary_sha256 = hashlib.sha256((ROOT / 'cmudict.dict').read_bytes()).hexdigest()

    def encode_with_metadata(self, text: str):
        normalized = normalize_turkish_text(text)
        words = re.findall(r"[abcçdefgğhıijklmnoöprsştuüvyz]+|[.,!?;:]", normalized)
        units = ['<sil>']
        unknown = []
        for token in words:
            if re.fullmatch(r'[.,!?;:]', token):
                if units[-1] != '<sil>':
                    units.append('<sil>')
                continue
            phones = turkish_word_to_phones(token)
            if not phones:
                unknown.append(token)
            else:
                units.extend(phones)
        if units[-1] != '<sil>':
            units.append('<sil>')
        if len(units) < 3:
            raise ValueError('No pronounceable Turkish words')
        return [self.indices[x] for x in units], {'unknown_words': unknown, 'phones': units, 'lang': 'tr'}

    def encode(self, text: str):
        return self.encode_with_metadata(text)[0]
