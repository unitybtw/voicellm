# VoiceLLM — yerel konuşma modeli geliştirme

Experimental local speech synthesis research in PyTorch. Trained from scratch using public human recordings, without pretrained model weights or cloud inference. **Intelligible speech is not yet validated.** This is a research code release, not a finished voice assistant.

## Açık kaynak sürümü ve kurulum

Kaynak kod MIT lisanslıdır; CMU telaffuz sözlüğünün ayrı lisansı `data/pronunciation/LICENSE` içinde korunur. Kişisel ses/video kayıtları, eğitim verileri, model/optimizer ağırlıkları ve yerel deney raporları bu repoda dağıtılmaz. Aşağıdaki geliştirme geçmişindeki `reports/` bağlantıları yerel deney dosyalarıdır; temiz klonda bulunmaz. Henüz indirilebilir, doğrulanmış bir checkpoint yayımlanmamıştır.

Python 3.11+ önerilir. Komutları repo kökünde çalıştırın:

```bash
git clone https://github.com/unitybtw/voicellm.git
cd voicellm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Testler indirilen eğitim verisi veya hazır checkpoint gerektirmez. Eğitim için resmi LJSpeech arşivini ayrıca indirin; birkaç GB disk alanı ve ciddi hesaplama süresi gerekir:

```bash
python scripts/download_public.py
python scripts/prepare_public.py
mkdir -p models/checkpoints reports
python scripts/train_public.py --device cpu
```

Apple Silicon üzerinde uygun PyTorch kurulumu ile `--device mps` kullanılabilir. İlk eğitim kalite koşullarını geçemezse aktif checkpoint oluşturulmaz. `train_spectral.py` ve `train_aligned.py` doğrulanmış yerel checkpoint ve ona ait metadata gerektirir; temiz klonda doğrudan çalışmaz. Eğitim komutlarının tamamlanması anlaşılır konuşmayı garanti etmez.

Katkı için [CONTRIBUTING.md](CONTRIBUTING.md), veri/lisans bilgileri için [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) dosyalarına bakın.

Dış model ağırlığı, distilasyon veya bulut modeli kullanılmaz. Kullanıcının videosu/sesi artık eğitime dahil edilmez. İnsan kayıtları ve eşleşmiş İngilizce metinler [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) resmi kaynağından alınır.

## Kalite durumu

**Kullanıcı son spektral örnekte de gürültü duyduğunu bildirdi. Anlaşılır konuşma başarısı doğrulanmadı.** Çıktının sessiz olmaması, düşen spektrum hatası, daha fazla periyodik kare veya başarılı kod testleri kalite kanıtı değildir. `speech_ready=false`.

Kanonik aktif dosya `models/checkpoints/voice_unified.pt`. Henüz geçmeyen geliştirme adayı `models/checkpoints/voice_aligned_candidate.pt` içinde tutulur. Bir aday, küçük eğitim kümesini öğrenmesiyle aktif model yapılmaz; ayrı doğrulama kayıtlarında sabit ortalama tahminini de geçmelidir. Aktif dosyanın türü/sayacı/hash'i `.training.json` yanında kayıtlıdır.

## Son değişiklik

Önceki ağ yalnızca toplam süreyi öğreniyordu; gerçek kayıtlardaki hızlı ses değişimlerinin yaklaşık %1–2'sini üretiyordu. Yeni tek ağ fonem bağlamını, her fonemin süresini, tam FFT spektrumunu, perdeyi ve sesli/sessiz olasılığını birlikte öğrenir. Çıktı önceki ses karesini tekrar tekrar geri beslemez.

İngilizce fonemler [CMU telaffuz sözlüğü](https://github.com/cmusphinx/cmudict) ile belirlenir. Bu insan telaffuz verisidir, model ağırlığı değildir. İndirilen commit, dosya hash'leri ve [lisans](https://github.com/cmusphinx/cmudict/blob/master/LICENSE) `data/pronunciation/` içindedir; sözlük dışı kelimeler açıkça işaretlenen harf okuma yöntemine düşer. Native checkpoint sözlük hash'ini ve fonem listesini doğrular.

Kendi dinamik programlama kodumuz gerçek hedef spektrum ve fonem özellikleri arasında sırayı koruyan eşleşme bulur. İnsan kelime zaman damgaları varmış gibi davranılmaz. Yöntem [monoton eşleştirme araştırmasından](https://arxiv.org/abs/2005.11129) yararlanır; başka modelin ağırlıkları veya dış hizalayıcı kullanılmaz. Eşleşmede kaba frekans özellikleri kullanılır; kısa ses birimlerinin atlanması engellenir.

Tam FFT büyüklüklerinden DSP yeniden yapım, dört gerçek kayıtta yaklaşık %0,17–0,35 büyüklük hatası verdi; önceki mel ters dönüşümü yaklaşık %12–15 hata verdi (`reports/decoder_reconstruction_comparison.json`). Bu yalnızca gerçek kaydın dönüştürülme testidir; TTS başarısı değildir.

Tek ağın perde ve sesli/sessiz tahminleri yerel harmonik kaynak/filtre sentezini sürer; aperiyodik kol filtreli rastgele faz gürültüsüdür. Öğrenilmiş ayrı bir vocoder yoktur. Kaynak waveform sentezlenir; eğitim WAV'ı kopyalanmaz veya karıştırılmaz. Başlangıç/bitişte 10 ms geçiş uygulanır ve Nyquist üstü harmonikler kesilir. Akış blok sentezidir; gerçek canlı gecikme iddiası yoktur.

Gürültü kolundaki fazların örtüşmesi enerji kaybettiriyordu; eski karışım düşük sesli olma olasılığında bile uğultu üretiyordu. Gürültü kolu tahmin edilen kare RMS'iyle kalibre edilir ve iki kol eşit güç geçişiyle karıştırılır. Harmonik sayısı artık Nyquist ve minimum perdeden hesaplanır; düşük perdede üst frekansları silen sabit 64 harmonik sınırı kaldırıldı. Bunlar kontrollü DSP regresyonlarıyla sınanır, anlaşılır konuşma kanıtı sayılmaz.

`periodic_prior_mode=confidence` ağın düşük veya kararsız sesli tahminlerinde spektruma harmonik filtre dayatmasını engeller. Önceki checkpoint'ler `probability` modunda aynı spektrum davranışıyla yüklenir. Yeni mod yeniden eğitimle sınanır; seçim başlangıcı kaynak checkpoint'in kendi modundaki çıktısıdır.

Yalnızca gerçek eşleşmeyle verilen süreye dayanmanın serbest üretimde hata yarattığı görülünce, modelin kendi süre tahminini kullanan diferansiyellenebilir üretim de ses kaybıyla eğitilir. Küçük küme tanı testlerinde, kendi eşleştirmelerimiz sabit tutularak değişen hedeflerin süre öğrenimini bozması ayrıca sınanır. `clock_mode=soft` bu davranışı checkpoint'e kaydeder; eski native modeller yüklenebilir.

Önceki geliştirme turlarında **6.400 gradyan adımı** denendi ve **5.600** sayaçlı aday seçildi. İlk 16 kayıt denemesi, 128 kayıt genişletmesi, serbest zamanlama eğitimi ve iki küçük küme tanı turu ayrı raporlanır. Ağ **14.670.213 parametre / yaklaşık 28,02 MiB**. Güncel adım, hash, test ve kalite durumu `reports/noise_development_latest.json` içinde tutulur. Genel konuşma eşiği geçmeyen aday aktif dosyaya taşınmaz.

Perde maskesinin gradyanı perde başlığını neredeyse sabit 200 Hz civarında tutuyordu. Bu yol ayrıldı; perde/sesli-sessiz başlıkları gerçek waveform etiketlerinden eğitilir. Aynı cümlede perde standart sapması yaklaşık **0,22 Hz → 47,56 Hz** oldu (`reports/pitch_lock_regression.json`). Etiket çıkarımında aralık sınırındaki monoton korelasyon, periyot zirvesi sayılmaz. Bu düzeltmeler telaffuz veya doğallık kanıtı değildir.

## Eğitim ve denetim

Önce 16 kısa kamu kaydında açıkça bir ezberleme/sanity testi yapıldı; sonra 128 ve 256 ayrı eğitim kaydına genişletildi. Son turlar 32 doğrulama ve 32 test kaydı kullanır. Doğrulama ve test kayıtları gradyan eğitiminde kullanılmaz; test sonuçları aday seçiminde kullanılmaz. Eğitim alt kümedir; tüm 24 saatin eğitildiği iddia edilmez. Her tur kaynak checkpoint hash'ini, kullanılan kayıtları, denenmiş/seçilmiş adımları ve gerçek ölçümleri raporlar. Başarısız adaylar aktif sayılmaz.

Spektrum hatasına ek olarak gerçek kaydın kare enerjisi de eğitim kaybına dahildir; aşırı sessiz ortalama spektrum çözümünü önlemeyi amaçlar. Aday seçimi ayrı doğrulamada perde, sesli/sessiz ayrımı, süre, enerji ve kare değişimini birlikte ölçer. Başlangıç adayından daha kötü bir turun onu ezmesine izin verilmez. Süre eğitimindeki minimum kare sayısı üretimle aynıdır; sabit uzunluk isteyen üretim tam kare sayısını korur. Tipografik kesme işaretleri İngilizce kelimeleri yanlış bölmez.

```bash
python3 scripts/train_aligned.py --steps 1600 --clips 16 --device mps
# Doğrulanmış kendi geliştirme adayından ve hash/adımı eşleşen optimizer'dan devam:
python3 scripts/train_aligned.py --checkpoint models/checkpoints/voice_aligned_candidate.pt --steps 600 --clips 256 --min-seconds 1.5 --max-seconds 6 --soft-clock --confident-periodic-prior --device mps --lr .00005
python3 scripts/review_aligned.py --checkpoint models/checkpoints/voice_aligned_candidate.pt
python3 scripts/voice.py speak --checkpoint models/checkpoints/voice_aligned_candidate.pt --text "Hello world." --temperature 0
python3 -m pytest -q
python3 -m compileall -q src scripts tests
```

Son raporlar `reports/aligned_training_latest.json`, `reports/generation_aligned_review.json` ve `reports/noise_development_latest.json`. Kod testleri dinamik programlamayı tam tarama optimumuyla, gerçek gradyanları, perde geri oluşturmayı, ses birimlerinin atlanmamasını, checkpoint kimliğini ve serbest zamanlama gradyanını doğrular. Türkçe konuşma, doğal telaffuz, frontier kalite veya rakiplere üstünlük henüz doğrulanmadı. Önceki deneyler `reports/SPECTRAL_POOLING_HISTORY.md` ve `reports/RAW_CODEC_HISTORY.md` içinde korunur.
