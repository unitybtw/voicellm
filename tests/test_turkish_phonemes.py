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
    assert deger == ['D', 'EH0', 'Y', 'EH1', 'R']

def test_turkish_phoneme_encoder_validity():
    tp = TurkishPhonemes()
    ids, meta = tp.encode_with_metadata('Merhaba, nasılsın?')
    assert meta['lang'] == 'tr'
    assert 'M' in meta['phones']
    assert 'EH0' in meta['phones']
    assert 'N' in meta['phones']
    assert 'AA1' in meta['phones']
    assert meta['phones'][0] == '<sil>' and meta['phones'][-1] == '<sil>'
    assert len(ids) == len(meta['phones'])

def test_turkish_syllable_stress():
    # Polysyllabic stress: unstressed penult, stressed ultima
    phones = turkish_word_to_phones('merhaba')
    assert phones == ['M', 'EH0', 'R', 'HH', 'AA0', 'B', 'AA1']
    # Monosyllabic stress: single vowel has stress 1
    ben_phones = turkish_word_to_phones('ben')
    assert ben_phones == ['B', 'EH1', 'N']

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
    # ASCII Turkish sentence without special letters
    _, meta_ascii = ep.encode_with_metadata('seni seviyorum')
    assert meta_ascii.get('lang') == 'tr'
    assert 'S' in meta_ascii['phones'] and 'V' in meta_ascii['phones']


def test_turkish_ordinal_expansion():
    from src.text.turkish_phonemes import expand_turkish_ordinal
    assert expand_turkish_ordinal(1) == 'birinci'
    assert expand_turkish_ordinal(2) == 'ikinci'
    assert expand_turkish_ordinal(3) == 'üçüncü'
    assert expand_turkish_ordinal(4) == 'dördüncü'
    assert expand_turkish_ordinal(5) == 'beşinci'
    assert expand_turkish_ordinal(6) == 'altıncı'
    assert expand_turkish_ordinal(7) == 'yedinci'
    assert expand_turkish_ordinal(8) == 'sekizinci'
    assert expand_turkish_ordinal(9) == 'dokuzuncu'
    assert expand_turkish_ordinal(10) == 'onuncu'
    assert expand_turkish_ordinal(20) == 'yirminci'
    assert expand_turkish_ordinal(21) == 'yirmi birinci'
    assert expand_turkish_ordinal(100) == 'yüzüncü'


def test_turkish_expanded_normalization():
    # Ordinals in sentences
    norm_ord = normalize_turkish_text('1. sınıfta 2’nci sırada oturdu.')
    assert 'birinci' in norm_ord
    assert 'ikinci' in norm_ord

    # Percentages
    norm_pct = normalize_turkish_text('Fiyatlarda %25 indirim var.')
    assert 'yüzde' in norm_pct and 'yirmi beş' in norm_pct

    # Currencies
    norm_cur = normalize_turkish_text('Kitap 50 TL tuttu.')
    assert 'elli türk lirası' in norm_cur

    # Abbreviations
    norm_abbr = normalize_turkish_text('vb. durumlar dr. ahmet tarafından incelendi.')
    assert 've benzeri' in norm_abbr
    assert 'doktor' in norm_abbr

    # Clause juncture dashes
    norm_dash = normalize_turkish_text('Hava güzel - dışarı çıkalım.')
    assert ', ' in norm_dash

def test_front_rounded_vowel_palatal_rules():
    """Validates that 'ü' palatal glide applies only to velar consonants (gün, kül) and not others (üç, bütün)."""
    assert 'Y' in turkish_word_to_phones('gün')
    assert 'Y' in turkish_word_to_phones('kül')
    assert turkish_word_to_phones('üç')[0] == 'UW1'
    assert 'Y' not in turkish_word_to_phones('bütün')
    assert 'Y' not in turkish_word_to_phones('türkçe')



