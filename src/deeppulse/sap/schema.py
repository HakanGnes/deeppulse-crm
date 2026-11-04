"""SAP tarzı tabloların sütun sözlüğü.

Bu tablolar SAP ECC / S/4HANA'daki gerçek tablo ve alan adlarını izler; ancak
sadeleştirilmiştir (her tabloda modellemenin ihtiyaç duyduğu alt küme var).
"Z" ile başlayanlar SAP geleneğindeki müşteriye özel (custom) alanlar/tablolardır.

Amaç: modelleme hattı (pipeline) Dunnhumby'yi doğrudan okumaz, her zaman bu
tablolardan okur (`adapter.from_sap`). Böylece gerçek bir SAP export'u
(SE16N, CDS view ya da OData ile alınmış aynı alanlar) aynı hatta takılabilir.
"""

TABLES: dict[str, dict[str, str]] = {
    "KNA1": {  # Müşteri ana verisi (genel)
        "KUNNR": "Müşteri numarası (10 karakter, sıfır dolgulu)",
        "NAME1": "Ad",
        "LAND1": "Ülke anahtarı",
        "KTOKD": "Müşteri hesap grubu",
    },
    "ZKNA1_DEMO": {  # Custom: hane demografisi (SAP standardında yok)
        "KUNNR": "Müşteri numarası",
        "ZZAGE_DESC": "Yaş aralığı",
        "ZZINCOME_DESC": "Gelir aralığı",
        "ZZHH_COMP_DESC": "Hane kompozisyonu",
        "ZZHH_SIZE_DESC": "Hane büyüklüğü",
        "ZZKID_CAT_DESC": "Çocuk sayısı kategorisi",
        "ZZHOMEOWNER_DESC": "Ev sahipliği",
        "ZZMARITAL_CODE": "Medeni durum kodu",
    },
    "MARA": {  # Malzeme ana verisi
        "MATNR": "Malzeme numarası (18 karakter, sıfır dolgulu)",
        "MATKL": "Mal grubu (commodity)",
        "SPART": "Bölüm (department)",
        "MFRNR": "Üretici",
        "ZZBRAND": "Marka tipi (Private/National)",
        "ZZSUBCOMM": "Alt mal grubu metni",
    },
    "T023T": {  # Mal grubu metinleri
        "MATKL": "Mal grubu",
        "WGBEZ": "Mal grubu tanımı",
    },
    "TSPAT": {  # Bölüm metinleri
        "SPART": "Bölüm",
        "VTEXT": "Bölüm tanımı",
    },
    "VBRK": {  # Fatura başlığı (SD)
        "VBELN": "Fatura belge numarası",
        "FKART": "Fatura türü (F2)",
        "FKDAT": "Fatura tarihi",
        "ERZET": "Kayıt saati (HHMMSS)",
        "VKORG": "Satış organizasyonu",
        "BUKRS": "Şirket kodu",
        "KUNAG": "Sipariş veren müşteri",
        "NETWR": "Net değer (belge para birimi)",
        "WAERK": "Belge para birimi",
        "KNUMV": "Fiyatlandırma koşul belge numarası",
        "XBLNR": "Referans belge (Dunnhumby BASKET_ID)",
    },
    "VBRP": {  # Fatura kalemi (SD)
        "VBELN": "Fatura belge numarası",
        "POSNR": "Kalem numarası",
        "MATNR": "Malzeme",
        "FKIMG": "Faturalanan miktar",
        "VRKME": "Satış ölçü birimi",
        "NETWR": "Kalem net değeri",
        "WERKS": "Üretim yeri / mağaza",
    },
    "PRCD_ELEMENTS": {  # Fiyatlandırma koşulları (S/4HANA; ECC'de KONV)
        "KNUMV": "Koşul belge numarası",
        "KPOSN": "Koşul kalem numarası (= POSNR)",
        "KSCHL": "Koşul türü: PR00 brüt fiyat, ZRDS perakende/sadakat indirimi, "
                 "ZCMD kupon eşleştirme indirimi, ZMCP üretici kuponu (istatistiksel)",
        "KWERT": "Koşul değeri (indirimler negatif)",
        "KSTAT": "İstatistiksel koşul ('X' ise net değeri etkilemez)",
    },
    "BKPF": {  # Muhasebe belge başlığı (FI)
        "BUKRS": "Şirket kodu",
        "BELNR": "Muhasebe belge numarası",
        "GJAHR": "Mali yıl",
        "BLART": "Belge türü (RV = fatura aktarımı)",
        "BUDAT": "Kayıt tarihi",
        "AWTYP": "Referans işlem (VBRK)",
        "AWKEY": "Referans anahtarı (= VBELN)",
    },
    "BSEG": {  # Muhasebe belge kalemi (FI)
        "BUKRS": "Şirket kodu",
        "BELNR": "Muhasebe belge numarası",
        "GJAHR": "Mali yıl",
        "BUZEI": "Kalem",
        "KOART": "Hesap türü (D müşteri, S ana hesap)",
        "SHKZG": "Borç/alacak göstergesi (S borç, H alacak)",
        "KUNNR": "Müşteri (KOART=D ise)",
        "HKONT": "Ana hesap",
        "DMBTR": "Tutar (yerel para birimi, her zaman pozitif)",
    },
}

CONDITION_GROSS = "PR00"
CONDITION_RETAIL_DISC = "ZRDS"
CONDITION_COUPON_MATCH = "ZCMD"
CONDITION_MANUFACTURER_COUPON = "ZMCP"
