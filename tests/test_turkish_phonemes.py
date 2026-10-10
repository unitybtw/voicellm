import pytest
from src.text.turkish_phonemes import (
    TurkishPhonemes, normalize_turkish_text, expand_turkish_number, turkish_word_to_phones
)
from src.text.phonemes import EnglishPhonemes

def test_turkish_number_expansion():
    assert expand_turkish_number(0) == 'sıfır'
    assert expand_turkish_number(1) == 'bir'
    assert expand_turkish_number(15) == 'on beş'
    assert expand_turkish_number(100) == 'yüz'
    assert expand_turkish_number(2026) == 'iki bin yirmi altı'
    assert expand_turkish_number(1000000) == 'bir milyon'

def test_turkish_normalization_preserves_dotted_and_dotless_i():
    norm = normalize_turkish_text('Isparta ve İstanbul')
    assert 'ısparta' in norm
    assert 'istanbul' in norm
    assert 'I' not in norm and 'İ' not in norm

def test_turkish_soft_g_rules():
    # Back vowel lengthening
    dag = turkish_word_to_phones('dağ')
    assert dag == ['D', 'AA1', 'AA1']
    # Front vowel glide
    deger = turkish_word_to_phones('değer')
    assert deger == ['D', 'EH1', 'Y', 'EH1', 'R']

def test_turkish_phoneme_encoder_validity():
    tp = TurkishPhonemes()
    ids, meta = tp.encode_with_metadata('Merhaba, nasılsın?')
    assert meta['lang'] == 'tr'
    assert 'M' in meta['phones']
    assert 'EH1' in meta['phones']
    assert 'N' in meta['phones']
    assert 'AA1' in meta['phones']
    assert meta['phones'][0] == '<sil>' and meta['phones'][-1] == '<sil>'
    assert len(ids) == len(meta['phones'])

def test_turkish_phoneme_symbols_match_cmu_inventory():
    tp = TurkishPhonemes()
    ep = EnglishPhonemes()
    assert tp.symbols == ep.symbols
    assert tp.dictionary_sha256 == ep.dictionary_sha256

def test_auto_routing_in_english_phonemes():
    ep = EnglishPhonemes()
    # English routes to CMUdict
    _, meta_en = ep.encode_with_metadata('Hello world')
    assert 'HH' in meta_en['phones'] and 'W' in meta_en['phones']
    # Turkish routes to TurkishPhonemes
    _, meta_tr = ep.encode_with_metadata('Merhaba dünya')
    assert meta_tr.get('lang') == 'tr'
    assert 'M' in meta_tr['phones'] and 'D' in meta_tr['phones']

