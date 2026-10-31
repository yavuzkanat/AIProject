# Dinamik Çoklu İHA Görev Dağıtımı için Derin Pekiştirmeli Öğrenme

Yapay Zekâ dersi için hazırlanmış bir araştırma projesi.

## Konu

Bir drone (İHA) sürüsü, harita üzerindeki hedefleri gözlemekle görevlidir: afet bölgesindeki
hasarlı binalar, izlenmesi gereken noktalar, keşif bölgeleri gibi. Bir drone işini bitirdiğinde
**"şimdi hangi hedefe gideyim?"** sorusuna cevap verilmelidir. Bu karar görev sırasında sürekli
tekrarlanır ve ortam değişir:

* drone düşebilir, sensörü bozulabilir,
* hava bozulup gözlem kalitesi düşebilir,
* yeni hedefler ortaya çıkabilir, hedefler yer değiştirebilir veya önemini yitirebilir.

Hedefe ne kadar geç gidilirse gözlemden elde edilen bilgi o kadar eskir. Uzun uçuşlar da riski
artırır. Bu yüzden sıralama kararları görevin sonucunu doğrudan belirler. Bu problem **dinamik
görev dağıtımı** olarak bilinir.

Proje, problemin matematiksel modelini şu makaleden alır:

> G. Li, Y. Liu, Q. Zheng, K. Liu, "A Multi-Swarm Dynamic Crow Search Algorithm for
> Multi-UAV Dynamic Task Allocation", *Drones* 2026, 10(6), 467.
> <https://www.mdpi.com/2504-446X/10/6/467>

Makale bu problemi bir evrimsel optimizasyon algoritmasıyla (MDCSATA) çözer: her olaydan sonra
bütün plan yeniden aranır.

## Amaç ve araştırma sorusu

Projenin amacı, aynı problemde farklı bir yaklaşımı **deneysel olarak** sınamaktır. Her olayda plan
aramak yerine, kararı veren bir politikayı **pekiştirmeli öğrenmeyle** (PPO) önceden eğitmek ve
görev sırasında yalnızca bu politikayı çalıştırmak.

**Araştırma sorusu:** Rastgele üretilmiş görevlerde eğitilen bir PPO politikası, daha önce hiç
görmediği görevlerde klasik karar kuralları ve yeniden planlayıcı bir yöntem kadar (veya daha fazla)
değer toplayabilir mi? Bunu çok daha kısa karar süresiyle yapabilir mi? Bu fark olay türüne,
olay sıklığına ve sürü büyüklüğüne göre nasıl değişir?

Sonuç önceden kabul edilmedi. Hipotezler deneylerden önce yazıldı ve desteklenmeyen hipotezler
de raporlandı (`docs/report.md`).

**Kapsam sınırları:**

* MDCSATA algoritması yeniden üretilmedi. Makalenin modeli (Denklem 4–11) ve üç senaryosu
  (Tablo 3–5) olay güdümlü bir simülatör olarak yeniden yazıldı; çözücü olarak farklı bir yöntem
  incelendi.
* Makalenin MHV/MIGD ölçütleri raporlanmadı. Tek bir karar politikası bir Pareto cephesi üretmez
  (`docs/phase1_design.md` §I.3).
* Bütün sonuçlar simülasyona aittir; gerçek dronelar hakkında kanıt değildir.

## Karşılaştırılan yöntemler

| Yöntem | Nasıl karar verir? |
|---|---|
| **Yapay zekâ (hedef-odaklı ağ)** | MaskablePPO. Her hedefe aynı küçük sinir ağı uygulanır ve hedefe bir puan verir; boşa çıkan drone en yüksek puanlı geçerli hedefe gider. Kural verilmedi; ödülden öğrendi. |
| Yapay zekâ + ileriye bakma | Modelin en iyi iki önerisi, gelecekteki olaylar bilinmeden simülatörün bir kopyasında görev sonuna kadar oynatılır; daha iyisi seçilir. |
| Klasik: en yakın hedef | Boşa çıkan drone en yakın geçerli hedefe gider. |
| Klasik: değer ÷ süre | Beklenen değer ÷ (uçuş + gözlem süresi) en yüksek hedef. |
| Klasik: planlayıcı (RH-RK) | Her kararda bütün sürü için tam plan arar (makalenin rastgele-anahtar kodlaması ve karga arama algoritması) ve planın ilk adımını uygular. MDCSATA değildir. |
| Rastgele seçim | Alt sınır. |

Bütün yöntemler aynı simülatörde, aynı görevlerde ve aynı kurallarla (pil, sensör, son tarih)
çalışır. Geçersiz bir hedef hiçbir yöntem tarafından seçilemez.

## İki ortam

| | v1: makale ortamı | v2: genişletilmiş ortam |
|---|---|---|
| Amaç | Makaleye sadık model; makaleyle karşılaştırılabilir | Önceliğe göre kendini yeniden düzenleyen gözetleme sürüsü |
| Sürü / hedef | 3–5 drone, 8–18 hedef | 4–7 drone, 16–28 hedef (en çok 40) |
| Öncelik | yok (her hedefin bir değeri var) | hedeflerin %25'i **kritik**: değeri yüksek, gözlem 200–700 s içinde başlamazsa değeri sıfır |
| Sensör | her drone her hedefi gözleyebilir | hedeflerin bir kısmı yalnızca kızılötesi veya yalnızca radarla gözlenebilir |
| Yeniden görevlendirme | yok: atanan görev değişmez | yeni hedef çıktığında veya drone düştüğünde, yoldaki dronelar hedef değiştirebilir |
| Skor | J = 0,5 × toplanan değer oranı − 0,5 × görev süresi / 2400 s | toplanan değer oranı |
| Yapılandırma | `configs/train/ppo_entity.yaml` | `configs/train/ppo_entity_hard.yaml` |
| Sonuçlar | `results/tables/`, `docs/report.md` | `results/tables/v2/` |

**v2 neden var?** Makale ortamında bir hedefin değeri neredeyse yalnızca yol süresine bağlıdır.
Bu yüzden "en yakın hedefe git" kuralı zaten en iyiye çok yakındır. Denenen bütün makul kurallar
birbirinin %1–2'si içinde kalmıştır. Böyle bir ortamda hiçbir yöntem belirgin fark açamaz. v2,
karar vermenin gerçekten önemli olduğu iki unsuru ekler: **son tarihli öncelikler** ve
**sensör yetenekleri**. Her kuralın gerekçesi `docs/assumptions.md` §10'dadır. v1 değişmedi:
v2'nin bütün seçenekleri varsayılan olarak kapalıdır.

## Sonuçlar (özet)

Bütün sayılar, eğitimde ve model seçiminde hiç kullanılmamış **500 test görevinin** ortalamasıdır.
Farklar aynı görevler üzerinde eşleştirilmiş Wilcoxon testi ve Holm düzeltmesiyle sınandı.

**v1 (makale ortamı, 5 eğitim tohumu):** Yapay zekâ bütün klasik yöntemlerden anlamlı ölçüde
iyidir (planlayıcıya göre J farkı +0,010, p < 1e-10). Ancak fark küçüktür: toplanan değer %23,0'a
karşı %22,1 (en yakın hedef). Karar süresi 0,19 ms, planlayıcıdan yaklaşık 70 kat hızlıdır.
Makalenin üç senaryosundan yalnızca S1'de planlayıcıya yetişebilmiştir. Ayrıntılar:
`docs/report.md`.

**v2 (genişletilmiş ortam, ilk 3 eğitim tohumu):**

| Yöntem | Toplanan değer | Kritik hedef zamanında | Karar süresi |
|---|---|---|---|
| **Yapay zekâ + ileriye bakma** | **%38,6** | **%70** | ~23 ms |
| Yapay zekâ | %36,5 | %68 | < 1 ms |
| Klasik: değer ÷ süre | %36,5 | %68 | < 0,1 ms |
| Klasik: planlayıcı (RH-RK) | %36,2 | %66 | ~32 ms |
| Klasik: en yakın hedef | %32,8 | %53 | < 0,1 ms |
| Rastgele seçim | %10,4 | %17 | – |

* Yapay zekâ, önceliği görmeyen **en yakın hedef kuralından açıkça iyidir**: +3,7 puan (göreli
  +%11, p = 2e-45). İleriye bakmayla birlikte fark +5,9 puandır (göreli +%18, p = 1e-75). Kritik
  hedeflerin zamanında gözlenme oranı da %53'ten %70'e çıkar.
* Değeri hesaba katan **değer ÷ süre kuralıyla aynı düzeydedir** (fark +0,01 puan, anlamlı değil).
  İleriye bakmayla birlikte bu kuralı da geçer (+2,2 puan, p = 1e-41).
* Daha uzun eğitim farkı büyütmedi. Doğrulama skoru yaklaşık 300–400 bin adımdan sonra düzleşti.
  Bu sınır karar yapısından gelir: bir sonraki hedefi seçme kararında basit değer kuralları zaten
  güçlüdür.
* Planlanan 5 tohumdan 3'ü tamamlandı; 3. ve 4. tohumların eğitimi yarıda bırakıldı. Bu yüzden v2
  sonuçları ön sonuçtur.

## Kurulum (macOS Apple Silicon; Linux'ta CPU ile de çalışır)

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[rl,analysis,dev]"
```

`requirements-lock.txt`, doğrulanmış ortamın tam sürümlerini içerir: Python 3.12, torch 2.14.1,
gymnasium 1.4.0, stable-baselines3 / sb3-contrib 2.9.0.

Sinir ağı eğitimi varsayılan olarak **CPU**'da yapılır. Bu boyuttaki küçük ağlar için Apple
GPU'su (MPS) yaklaşık 15 kat daha yavaş ölçüldü. Ayrıca SB3'ün `device="auto"` seçeneği MPS'i
hiçbir zaman seçmez. Bir eğitim yapılandırmasında `device: mps` yazılabilir; kullanılan cihaz her
koşunun `versions.json` dosyasına kaydedilir. Simülatör her zaman CPU'da çalışır.

## Kullanım

```bash
pytest                                   # birim ve entegrasyon testleri (279 test)
python scripts/check_env.py              # Gymnasium + SB3 ortam kontrolleri, bir makale görevi

# Panel: http://127.0.0.1:8765/
# Ortam seçimi (v1 / v2), özet, yapay zekâ ile klasik yöntemlerin yan yana oynatılması,
# test sonuçları ve istatistikler, öğrenme eğrileri, eğitim koşuları ve PPO göstergeleri.
# Belirli bir ana bağlantı: /?env=v1&scenario=paper_S3&t=420
python scripts/dashboard.py

# v1: eğitim (M3 işlemcide 2M adım, tohum başına ~10–45 dk), değerlendirme ve analiz
python scripts/train.py --config configs/train/ppo_entity.yaml --seed 0
./scripts/run_all_training.sh            # bütün v1 koşuları; yarıda kalırsa kaldığı yerden sürer
python scripts/evaluate.py --spec configs/eval/main.yaml
python scripts/evaluate.py --spec configs/eval/generalization.yaml
python scripts/analyze.py
python scripts/make_figures.py

# v2: 5 tohum eğitim (tohum başına ~1,5–2 saat), değerlendirme ve analiz
PARALLEL=3 ./scripts/run_v2_training.sh
python scripts/evaluate.py --spec configs/eval/v2_main.yaml
python scripts/evaluate.py --spec configs/eval/v2_generalization.yaml
python scripts/analyze.py --main results/eval/v2_main.csv --gen results/eval/v2_generalization.csv \
    --ppo ppo_entity_hard ppo_entity_hard_lookahead --out results/tables/v2
```

Herhangi bir eğitim seçeneği kod değiştirmeden ezilebilir:
`--set '{"env": {"reward": {"w_value": 0.75, "w_time": 0.25}}, "ppo": {"ent_coef": 0.0}}'`.

## Tekrarlanabilirlik

* Bütün rastgele değerler tohumlanmış `numpy.random.Generator` nesnelerinden gelir. Senaryo
  yerleşimi ve olay takvimi, senaryo tohumunun birbirinden bağımsız alt akışlarını kullanır
  (`utils/seeding.py`).
* Rastgele olayların hangi drone'u veya hedefi etkileyeceği önceden çekilir. Böylece aynı tohumda
  değerlendirilen bütün yöntemler aynı olay dizisiyle karşılaşır.
* Tohum aralıkları ayrıktır: eğitim < 1.000.000 ≤ doğrulama < 2.000.000 ≤ test < 3.000.000 ≤
  dağılım dışı testler. En iyi model yalnızca doğrulama görevlerine göre seçilir (`best_model.zip`).
* Her koşu klasörü `config.yaml`, `versions.json`, `progress.csv`, `validation.csv`,
  `training_summary.json`, `best_model.zip` ve `final_model.zip` dosyalarını saklar.
* Aynı tohum ve aynı karar dizisi her zaman bit düzeyinde aynı simülasyon çıktısını verir (testle
  doğrulanmıştır).

## Dizin yapısı

```text
configs/scenarios/paper.yaml   makalenin Tablo 3–5'i (S1/S2/S3)
configs/train/                 PPO yapılandırmaları (v1: düz ağ, hedef-odaklı ağ, dikkat ağı; v2)
configs/eval/                  değerlendirme koşulları (test, makale, genelleme, ablasyon, v2)
src/uavta/entities/            drone ve hedef veri modelleri
src/uavta/dynamics/            Denklem 4–11 için saf fonksiyonlar
src/uavta/events/              olay türleri
src/uavta/scenarios/           YAML şeması ve yükleyici, rastgele senaryo üreteci
src/uavta/environment/         simülatör, gözlem, eylem maskesi, ödül, Gymnasium ortamı
src/uavta/baselines/           rastgele, en yakın hedef, değer ÷ süre, RH-RK planlayıcı
src/uavta/agents/              MaskablePPO, hedef-odaklı politika ağı, ileriye bakma
src/uavta/training/            eğitim
src/uavta/evaluation/          görev çalıştırıcı, ölçütler, istatistik, deney yöneticisi
src/uavta/visualization/       panel (dashboard), grafikler
tests/                         birim ve entegrasyon testleri
scripts/                       komut satırı betikleri
results/                       değerlendirme CSV'leri, tablolar, şekiller (eğitim koşuları git'e girmez)
docs/                          tasarım belgesi, varsayımlar, sonuç raporu
```

## Ortam tek paragrafta

`MultiUAVDynamicEnv` (Gymnasium kimliği `MultiUAVDynamic-v0`) olay güdümlü bir yarı-Markov karar
sürecidir. Her adımda *karar veren drone*'a bir hedef atanır. Ardından sürekli zaman, bir sonraki
karar anına (bir drone'un boşa çıkması veya bir olay) kadar ilerler.

Gözlem sabit uzunlukta bir sayı dizisidir. v1'de 527 sayıdan oluşur: 6 drone ve 24 hedef yuvası,
boş yuvalar sıfırla doldurulur. Yeni hedefler sıradaki boş yuvaya girer.

Geçersiz eylemler bir eylem maskesiyle kaldırılır ve `sb3_contrib.MaskablePPO` bu maskeyi kullanır.
Geçersiz eylemler şunlardır: tamamlanmış, başka drone'a atanmış, pilin yetmediği veya (v2'de)
gerekli sensörün ya da son tarihin tutmadığı hedefler.

Altı olay türü desteklenir: yeni hedef, drone düşmesi, sensör arızası, hava, hedef taşınması ve
değer düşüşü. Olaylar ya makaledeki gibi önceden zamanlanır (S1–S3) ya da tohumlanmış Poisson
süreçleriyle üretilir.

## Belgeler

| Belge | İçerik |
|---|---|
| `docs/phase1_design.md` | makale analizi, MDP formülasyonu, tasarım seçenekleri |
| `docs/assumptions.md` | makalenin her yorumu, her uygulama kararı, v2'nin gerekçesi (§10) |
| `docs/report.md` | v1 deney sonuçları, hipotezler, sınırlılıklar |
| `results/tables/summary.md`, `results/tables/v2/summary.md` | bütün sonuç tabloları (betikle üretilir) |

## Bilinen sınırlılıklar

* Dronelar nokta kütle olarak düz çizgide uçar; enerji modeli yalnızca zamana bağlıdır (makaledeki
  gibi). Rüzgâr, engel, çarpışma, iletişim ve üsse dönüş yoktur.
* Risk, beklenen değer üzerinden basitleştirilmiş bir modeldir.
* Kanıt yalnızca simülasyondan gelir.
* İki amaç tek bir skorda birleştirilmiştir. İndirim karar başınadır, saniye başına değildir.
* v2 ortamı makalede yoktur; bu projenin tasarımıdır. v2 sonuçları 5 yerine 3 tohuma dayanır.
