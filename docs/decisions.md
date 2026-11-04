# Tasarım kararları (Faz 0)

Bu dosya, sonradan değişirse projenin büyük kısmını etkileyen kararları ve gerekçelerini tutar.
Bir karar değişirse buraya tarihli bir not eklenir, eskisi silinmez.

## K1 — Churn tanımı: N günlük sessizlik

**Sorun.** Sözleşmesiz perakendede müşteri "abonelikten çıkmaz"; terk anı hiçbir zaman gözlenmez.
BG/NBD bu yüzden terki gizli bir değişken olarak modeller. Survival modelleri (DeepHit, CoxTime)
ise gözlenen bir olay ister.

**Karar.** Art arda N gün hiç alışveriş yapmayan müşteri terk etmiş sayılır.
Olay günü = son alışveriş + N (sessizliğin eşiği aştığı gün).

Snapshot günü T'de:
- Yalnızca aktif müşteriler (recency < N) modele girer.
- `duration` = olay günü − T, `event` = olay veri ufkunda gözlendiyse 1, değilse 0 (sağdan sansür).
- `churn_w` = W gün içinde olay (sınıflandırma etiketi).

**Sonuç.** "Kaç gün sonra terk eder?" sorusunun ölçülebilir karşılığı "N günlük sessizlik eşiğine
kaç günde ulaşır?" olur. Model bir müşterinin geri dönüp dönmeyeceğini değil, ritminin kopup
kopmayacağını tahmin eder. Raporlarda bu ayrım açıkça yazılmalı.

**Açık nokta.** Varsayılan N = 30, W = 90. Dunnhumby'deki 2.500 hane "sık alışveriş yapanlar"
arasından seçildiği için churn oranı düşük çıkabilir. Gerçek veride
`scripts/suggest_threshold.py` çalıştırılıp N bu tabloya göre sabitlenecek.

### Güncelleme 2026-11-04 — gerçek veri: hedef "terk" değil "kopma (lapse)", N = 60

`suggest_threshold.py` gerçek Dunnhumby verisinde (boşluk medyanı 3 gün, 95. yüzdelik 22, 99. yüzdelik 61):

| N | Boşlukların N'den uzun payı | Son snapshot'ta oran | Olaydan sonra 180 günde geri dönüş |
|---|---|---|---|
| 14 | %9,3 | %63,2 | %97,9 |
| 30 | %3,1 | %32,9 | %94,7 |
| 45 | %1,6 | %21,2 | %91,5 |
| 60 | %1,0 | %13,9 | %88,6 |
| 90 | %0,5 | %7,4 | %82,8 |

**Bulgu.** Hiçbir N'de kalıcı terk yok: 90 günlük sessizlikten sonra bile müşterilerin %83'ü geri dönüyor.
Muhtemel sebep panelin kendisi: haneler iki yıl boyunca düzenli alışveriş yapanlar arasından
seçilmiş, yani gerçekten giden müşteriler baştan veriye alınmamış olabilir.

**Karar.** Hedefin anlamı "kopma (lapse)": müşteri ritmini kaybedip uzun süre sessiz kalıyor, ama
büyük ihtimalle geri geliyor. CRM'deki karşılığı geri kazanma (win-back) kampanyasının tetikleyicisi.
Kodda `churn_*` adları korunuyor; raporlarda ve sunumda "kopma" denecek.

**N = 60 gerekçesi.** 60 günlük boşluk tüm alışveriş aralıklarının yalnızca %1'i, yani popülasyon
ritmine göre gerçekten olağan dışı. Oran (%13,9) dengesiz ama modellenebilir. N = 30'da ise
seyrek alışveriş yapan bir müşterinin normal ritmi "kopma" sayılıyordu; model davranış değişikliği
yerine müşteri tipini öğreniyordu.

**Sonraki fazlara etkisi.** Müşteriler aslında gitmediği için 6 aylık CLTV'yi tutundurma değil,
**sıklık ve sepet tutarı** belirliyor. Faz 3–4'teki "indirim → CLTV" simülatörü indirimin sıklık ve
harcama üzerindeki etkisini modellemeli; "kaç müşteriyi tutar" sorusu ikincil kalır.

**Erken uyarı metriği.** Recency eşiğe yaklaştıkça kopma neredeyse kesinleşir (recency'si 55 olan
biri 5 gün içinde gelmezse zaten kopmuş sayılır), bu da genel AUC'yi şişirir. Bu yüzden raporda
ayrıca `recency ≤ 14` alt kümesindeki AUC veriliyor (`early_warning`). Derin modellerin asıl
değer katması gereken yer burası: sessizlik başlamadan önce tahmin.

## K2 — Zaman bazlı, sızıntısız bölme

AnalyzeWise-CRM'de rolling snapshot'lar rastgele bölündüğünde metrikler şişmişti
(sinyalsiz veride AUC ~0.65). Burada:
1. Eğitim snapshot'ları T ≤ test_günü − W.
2. Eğitim etiketleri `horizon = test_günü` ile hesaplanır (test gününden sonrası görünmez).
3. Test, test gününde aktif müşterilerin tamamı; etiketleri gerçek ufukla.

`tests/test_labels_and_splits.py::test_fold_train_is_invariant_to_data_after_test_day`
bunu garanti eder: test gününden sonraki veriyi silmek ya da değiştirmek eğitim tablosunu
değiştirmemeli.

Not: test kümesinde takip süresi en fazla W gün (Dunnhumby'de 711 − 621 = 90). Survival
metrikleri bu yüzden 90 günlük ufukla sınırlı.

## K3 — SAP tarzı veri katmanı

Modelleme hattı Dunnhumby'yi doğrudan okumaz. Ham veri önce SAP tarzı tablolara çevrilir,
hat bunlardan okur:

| Tablo | İçerik | Dunnhumby kaynağı |
|---|---|---|
| KNA1 / ZKNA1_DEMO | Müşteri ana verisi / demografi (custom) | household_key, hh_demographic |
| MARA, T023T, TSPAT | Malzeme, mal grubu, bölüm | product |
| VBRK / VBRP | Fatura başlığı / kalemi (SD) | transaction_data (sepet = fatura) |
| PRCD_ELEMENTS | Fiyat koşulları: PR00, ZRDS, ZCMD, ZMCP | sales_value + indirim sütunları |
| BKPF / BSEG | Muhasebe kayıtları (FI) | fatura toplamlarından türetilir |

Gerçek bir SAP sisteminden aynı alanlarla export alınırsa (SE16N, CDS view, OData),
`to_sap` adımı atlanır ve hattın geri kalanı değişmeden çalışır.
Dunnhumby'de gerçek tarih yok; `SapConfig.base_date` (2023-01-01) temsilidir.

## K4 — Tutar tanımları (gerçek veride doğrulanacak)

- `net_value` = SALES_VALUE (perakendecinin aldığı)
- `gross_value` = net + |RETAIL_DISC| + |COUPON_MATCH_DISC| (indirimsiz raf değeri)
- `coupon_disc` = |COUPON_DISC|: üretici kuponu; perakendeci için istatistiksel
  (ZMCP, KSTAT = 'X'), net değeri azaltmaz.

Dunnhumby User Guide'daki fiyat formülleriyle karşılaştırılıp gerekirse düzeltilecek.
Faz 3'te (indirim etkisi) bu tanım doğrudan "tedavi" (treatment) değişkenini belirlediği için önemli.

## K5 — Baseline'lar

| Görev | Baseline | Metrik |
|---|---|---|
| W günde churn | recency tek başına, XGBoost | ROC AUC, PR AUC, Brier |
| Terke kadar süre | CoxPH | C-index, 1−S(W) ile churn AUC |
| Sonraki sepet | global popülerlik, kişisel tekrar sıklığı | Recall@10, NDCG@10, Hit@10 |

Market verisinde kişisel tekrar baseline'ı genelde çok güçlüdür (insanlar aynı sütü, ekmeği alır).
Faz 1'deki sekans modelinin hedefi bunu geçmek. Geçemezse bu dürüstçe raporlanır.

### Gerçek veride çıta (2026-11-04, N = 60, W = 90, test günü 621, 2.247 aktif müşteri)

| Görev | Basit | Güçlü |
|---|---|---|
| 90 günde kopma, ROC AUC | recency: 0.834 | XGBoost: **0.893** (PR AUC 0.646, oran %13,9) |
| Kopmaya kadar süre | — | CoxPH C-index: **0.878**, 1−S(90) ile AUC 0.889 |
| Sonraki sepet, Recall@10 | popülerlik: 0.088 | kişisel tekrar: **0.157** (NDCG@10 0.192) |

**Erken uyarı** (`recency ≤ 14`: 1.766 müşteri, kopma oranı %5,7 ≈ 100 olay):

| Model | ROC AUC | PR AUC |
|---|---|---|
| recency | 0.649 | 0.102 |
| XGBoost | **0.822** | **0.274** |
| CoxPH 1−S(90) | 0.815 | — |

Derin modellerin asıl çıtası bu tablo. Pozitif sınıf ~100 olay olduğu için tek test gününde AUC
gürültülü (kabaca ±0.03–0.04). Faz 1'den itibaren karşılaştırmalar bootstrap güven aralığıyla ve
birden fazla test gününde (rolling-origin) yapılacak.


## K6 — Faz 1 model tasarımı (2026-11-04)

**Token = sepet.** Her sepet; ürün embedding'lerinin ortalaması + sürekli özellikler (önceki sepetten
bu yana gün, net tutar, indirim payı, ürün sayısı, kupon) + haftanın günü vekili (`day % 7`) ile
temsil edilir. Dizi kodlayıcı: GRU4Rec tarzı GRU ya da SASRec tarzı nedensel Transformer.
Vocab: test gününe kadar en az 5 sepette geçen ürünler, geri kalanı `<unk>`.

**Sonraki sepette tekrar bonusu.** Sentetik veride ilk denemede sinir ağı popüler ürünlere takıldı ve
kişisel tekrar baseline'ının belirgin altında kaldı (Recall@10 0.72 vs 0.87). Bu, sonraki sepet
literatüründe bilinen bir durum: tekrar alımlar baskın, saf embedding benzerliği bunu öğrenmekte zorlanıyor.
Çözüm: skor = h·e_i + g(h)·log(1 + ürünü içeren geçmiş sepet sayısı) + b·[ürün daha önce alındı].
g(h) ≥ 0 öğrenilir, yani model müşterinin durumuna göre "tekrar mı keşif mi" ağırlığını ayarlar.
Sentetik veride bu ekleme farkı kapattı (0.865 vs 0.870, anlamlı değil). `--no-repeat` ile kapatılabilir.
Not: geçmiş sayıları dizi penceresiyle (son 50 sepet) sınırlı; baseline tüm geçmişi kullanıyor.

**Kopma başlığı Faz 0 özelliklerini de görür.** Sadece diziyi gören model, tenure ve toplam harcama
gibi uzun dönem bilgisini kaçırıyordu. Başlık girdisi = [dizi durumu, recency, ölçeklenmiş Faz 0
özellikleri]. Asıl soru "dizi, özet özelliklerin üstüne bilgi ekliyor mu?" olduğu için
`--no-summary` ablasyonu da raporlanacak.

**İki başlık, iki checkpoint.** Kopma başlığı 2–7 epoch'ta doyuyor ve aşırı öğrenmeye başlıyor; sonraki
sepet daha yavaş öğreniyor. Tek bir birleşik erken durdurma skoru sonraki sepeti erken kesiyordu.
Her başlığın doğrulama skoruna göre ayrı en iyi checkpoint'i tutulur, testte o kullanılır.

**Doğrulama.** Müşterilerin %10'u eğitimden tamamen ayrılır. Sonraki sepet doğrulaması son eğitim
snapshot'ında, kopma doğrulaması bu müşterilerin tüm eğitim snapshot'larında yapılır; hepsi test
gününden önceki veriyle.

**Bellek.** Sampled softmax'ta aday küme = partideki hedef ürünler + 2.048 rastgele negatif, üst
sınır 8.192. Üst sınır aşılırsa hedeflerin rastgele alt kümesi kullanılır. En kötü durumda
(her hedefin farklı olduğu yapay parti) bir adım CPU'da ~2 GB tepe bellek kullandı.

**Karşılaştırma.** Aynı test müşterileri üzerinde eşli bootstrap (1.000 tekrar), fark için %95 aralık.
Aralık 0'ı içeriyorsa "model daha iyi" denmez. Erken uyarı alt kümesi ayrıca raporlanır.

### Faz 1 ilk gerçek veri sonuçları (2026-11-04, test günü 621, RTX 2070 Super)

Vocab 34.334 ürün, 238.524 sepet. GRU 2,26 M, SASRec 2,32 M parametre (çoğu ürün embedding'i).
Süre: GRU ~20 s/epoch, SASRec ~14 s/epoch.

| | GRU | SASRec | Baseline |
|---|---|---|---|
| Kopma ROC AUC (tümü) | 0.895 | 0.893 | XGBoost 0.893 — fark anlamlı değil |
| Kopma ROC AUC (erken uyarı) | 0.826 | 0.822 | XGBoost 0.822 — fark anlamlı değil |
| Sonraki sepet Recall@10 | **0.167** | **0.166** | kişisel tekrar 0.157 — **model anlamlı olarak daha iyi** (+0.009/+0.010, %95 aralık 0'ın üstünde) |

Gözlemler:
1. **Kopma:** Model XGBoost'un özet özelliklerini de gördüğü halde onu geçemiyor. Dizinin ek
   bilgi taşıyıp taşımadığı `--no-summary` ablasyonuyla ölçülecek.
2. **Sonraki sepet:** Kişisel tekrarı geçen ilk model (~%6 göreli). Ama her iki kodlayıcıda en iyi
   doğrulama epoch 1; sonrasında eğitim kaybı düşerken doğrulama NDCG'si geriliyor. Kazancın büyük
   kısmı muhtemelen tekrar bonusundan ve hızlı öğrenilen popülerlikten geliyor; dizi modellemesinin
   katkısı belirsiz (`--no-repeat` ile ölçülecek).
   Gerileme için hipotez: in-batch negatifler popüler ürünleri orantısız sık negatif yapıyor ve
   model zamanla popüler ürünleri fazla bastırıyor. Test: `--logq` (logQ düzeltmesi), `--lr 3e-4`.
3. GRU ve SASRec arasında fark yok; SASRec daha hızlı olduğu için ablasyonlar SASRec ile yapılacak.

### Faz 1.1 ablasyonları (2026-11-04, SASRec, test günü 621)

| Çalıştırma | Kopma AUC (tümü / erken uyarı) | Sonraki sepet R@10 | En iyi epoch (sepet / kopma) |
|---|---|---|---|
| Tam model (lr 1e-3) | 0.893 / 0.822 | 0.166 | 1 / 4 |
| `--no-summary` | **0.878 / 0.785** (XGBoost'tan anlamlı olarak kötü) | 0.167 | 1 / 4 |
| `--no-repeat` | 0.893 / 0.823 | **0.093** (kişisel tekrardan anlamlı olarak kötü) | 4 / 5 |
| `--logq --lr 3e-4` | 0.892 / 0.815 | **0.169** | **5 / 10** |
| XGBoost / kişisel tekrar | 0.893 / 0.822 | 0.157 | — |

**Kopma: dizi, özet özelliklere bilgi eklemiyor.** Yalnızca diziyi gören model XGBoost'un anlamlı
olarak altında (erken uyarıda −0.038); özet özellikler eklenince XGBoost'a eşitleniyor ama geçemiyor.
Kopma bu veride sıklık ve recency ile belirleniyor, bunları özet özellikler zaten yakalıyor.
Karar: kopma için üretimde XGBoost/CoxPH yeterli; derin modelin kopma başlığı ancak çok görevli
öğrenmeye katkı için tutulur, ayrı bir iddiası yok.

**Sonraki sepet: kazancın kaynağı tekrar terimi.** Tekrar terimi olmadan sinir ağı popülerlik
seviyesine düşüyor (0.093; popülerlik 0.088). Tekrar terimiyle kişisel tekrarı ~+0.010 geçiyor.
Bu +0.010'un ne kadarının öğrenmeden, ne kadarının sadece "son alınanlara daha çok ağırlık
vermekten" geldiğini ayırmak için zaman ağırlıklı tekrar baseline'ı eklendi
(skor = Σ decay^(sepet yaşı), decay son eğitim snapshot'ında seçilir).

**logQ + lr 3e-4: gerileme durdu.** En iyi sonraki sepet epoch'u 1'den 5'e çıktı, doğrulama artık
geriye gitmiyor; test kazancı küçük (0.166 → 0.169, gürültü içinde). Eğitim daha kararlı olduğu için
**varsayılan yapıldı** (`--no-logq` ile kapatılabilir). İkisinin etkisi ayrıştırılmadı.
