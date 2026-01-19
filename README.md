# # 🚦 Akıllı Trafik Işığı Yönetimi (PPO & SUMO)

![Development Status](https://img.shields.io/badge/Status-Geliştirme%20Aşamasında-yellow)
![Python](https://img.shields.io/badge/Python-3.12.7-blue)
![SUMO](https://img.shields.io/badge/Simulation-SUMO-orange)
![Algorithm](https://img.shields.io/badge/RL-PPO-green)

Bu proje, kentsel trafik akışını optimize etmek ve bekleme sürelerini en aza indirmek amacıyla **Pekiştirmeli Öğrenme (Reinforcement Learning)** yöntemlerini kullanan bir simülasyon çalışmasıdır. Trafik ışıklarının kontrolü için **Proximal Policy Optimization (PPO)** algoritması kullanılmakta ve ortam simülasyonu **SUMO (Simulation of Urban MObility)** üzerinde gerçekleştirilmektedir.

> ⚠️ **Not:** Bu proje şu anda aktif geliştirme aşamasındadır. Kod yapısı, hiperparametreler ve model performansı düzenli olarak güncellenmektedir.

## 🎯 Projenin Amacı

Geleneksel zaman ayarlı trafik ışıkları, değişken trafik yoğunluklarına adapte olmakta yetersiz kalabilmektedir. Bu projenin temel hedefleri:
- Kavşaklardaki araç kuyruk uzunluklarını (queue length) azaltmak.
- Ortalama bekleme süresini (waiting time) minimize etmek.
- PPO ajanı ile dinamik ve adaptif bir trafik yönetim sistemi oluşturmak.

## 🛠️ Kullanılan Teknolojiler

* **Dil:** Python
* **Simülasyon:** SUMO (Simulation of Urban MObility) / TraCI
* **Algoritma:** PPO (Proximal Policy Optimization)
* **Kütüphaneler:**
    * `gymnasium` veya `gym` (Ortam yönetimi için)
    * `stable-baselines3` (RL algoritmaları için)
    * `torch` (Ekran Kartı için)
    * `sumolib` & `traci` & `numpy`

## ✨ Proje Özellikleri ve Yetenekler

Bu proje, bir trafik simülasyonunu sadece izlemek yerine, onu **Gymnasium** arayüzü üzerinden programatik olarak kontrol edilebilir bir öğrenme ortamına dönüştürmektedir.

* **🔌 Özel Gym Ortamı (Custom Environment):**
    * SUMO simülasyonu, standart OpenAI Gym/Gymnasium formatına (`step`, `reset`, `render`) uygun hale getirilmiştir.
    * Bu sayede Stable-Baselines3 gibi popüler RL kütüphaneleri ile doğrudan entegrasyon sağlanmıştır.

* **🔄 TraCI ile Anlık Kontrol:**
    * Python script'i, **TraCI (Traffic Control Interface)** üzerinden simülasyondaki trafik ışıklarına saniye bazında müdahale edebilir.
    * Araç verileri (konum, hız, bekleme süresi) simülasyon durdurulmadan gerçek zamanlı olarak çekilir.

* **🧠 Dinamik Durum Gözlemi (State Observation):**
    * Ajan, sadece ışığın rengini değil; şeritlerdeki araç yoğunluğunu ve kümülatif bekleme sürelerini matris formatında algılar.

* **🤖 PPO Entegrasyonu:**
    * Trafik akışını optimize etmek için **Proximal Policy Optimization (PPO)** algoritması aktif olarak çalışmaktadır.
    * Model, belirlenen ödül fonksiyonuna (reward function) göre aksiyon alarak trafiği rahatlatmayı öğrenir.

## 🔬 Aktif Geliştirme Odakları

Projenin teknik altyapısı ve simülasyon bağlantısı tamamlanmış olup, şu anda **ajanın (agent) performansını maksimize etmeye** odaklanılmaktadır. Güncel çalışmalar şu başlıklar altında devam etmektedir:

* **🧪 Ödül Fonksiyonu (Reward Function) Tasarımı:**
    * Ajanın en doğru stratejiyi öğrenmesi için kritik olan "ödül mekanizması" geliştiriliyor.
    * Sadece *bekleme süresini* değil, aynı zamanda *kuyruk uzunluğunu* ve *araçların dur-kalk sayısını* da hesaba katan karmaşık bir ödül fonksiyonu üzerinde deneyler yapılıyor.

* **📈 Hiperparametre Optimizasyonu:**
    * PPO algoritmasının daha kararlı (stable) öğrenmesi için `learning_rate`, `batch_size` ve `gamma` gibi hiperparametreler üzerinde ince ayarlar yapılıyor.
    * Modelin öğrenme eğrisi (learning curve) Tensorboard üzerinden izlenerek aşırı öğrenme (overfitting) engellenmeye çalışılıyor.
      
* **🔄 Exploration (Keşfetme) Dengesinin İyileştirilmesi:**
    * Modelin belirli senaryolarda "sabit karar verme" (policy collapse) eğilimi analiz edilmektedir.
    * Ajanın farklı stratejileri denemeye devam etmesi için *Entropy Coefficient* katsayısı üzerinde çalışılmakta ve *Epsilon-Greedy* benzeri keşif mekanizmaları test edilmektedir.

* **🌐 Çoklu Kavşak (Multi-Intersection) Entegrasyonu:**
    * Şu anki tekil kavşak (single-agent) başarısı referans alınarak, sistemin çoklu kavşak senaryolarına taşınması planlanmaktadır.
    * Komşu kavşakların birbirini etkilediği bu senaryoda, merkezi veya dağıtık (decentralized) öğrenme yaklaşımları araştırılmaktadır.
 
## 🔮 Gelecek Vizyonu (Future Works)

Bu proje, temel trafik kontrolünün ötesine geçerek tam otonom bir şehir yönetim sistemi olmayı hedeflemektedir. Henüz AR-GE aşamasında olan uzun vadeli hedefler şunlardır:

* **🚑 Acil Durum Önceliklendirmesi (EVP - Emergency Vehicle Preemption):**
    * Ambulans, itfaiye ve polis araçlarının simülasyonda tespiti.
    * Bu araçlar kavşağa yaklaştığında, RL modelinin ödül fonksiyonunu (reward function) override ederek onlara anında "Yeşil Dalga" (Green Wave) oluşturulması.

* **👁️ Bilgisayarlı Görü (Computer Vision) Entegrasyonu:**
    * Simülasyondaki yapay sensör verileri yerine, gerçek dünya senaryosuna hazırlık olarak kamera görüntülerinin kullanılması.
    * **YOLO (You Only Look Once)** gibi nesne tespit algoritmalarıyla araçların sayılması, sınıflarının (kamyon, otomobil, otobüs) ayrıştırılması ve bu verilerin PPO modeline input olarak verilmesi.

* **🧠 Öngörüye Dayalı Ağ Yönlendirmesi (Predictive Routing):**
    * Sistemin sadece anlık durumu değil, trafik yoğunluğunun artacağı saatleri tahmin etmesi.
    * Sıkışıklık henüz oluşmadan, araçların navigasyon sistemleriyle haberleşerek onları alternatif ve daha boş kavşaklara yönlendiren (Load Balancing) makro bir strateji geliştirilmesi.

## 📂 Kurulum ve Çalıştırma

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1.  Repoyu klonlayın:
    ```bash
    git clone https://github.com/ehehehe-cmd/Traffic_Management.git
    cd Traffic_Management
    ```

2.  Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install -r requirements.txt
    ```

3.  Eğitimi başlatmak için:
    ```bash
    python train.py
    ```

## 📂 Proje Dosya Yapısı

Bu proje, kodun modülerliğini sağlamak ve farklı deneyleri yönetmek amacıyla aşağıdaki dizin yapısına sahiptir. Bu dizin yapısı projenin genelini temsil etmektedir basit değişikliler olablir:

```text
Traffic_Management/
├── adaptor.py              # SUMO ile model arasındaki bağlantı adaptörü
├── egitim.py               # Eğitimi başlatan ana dosya
├── test_model.py           # Eğitilmiş modeli test etme kodu
├── requirements.txt        # Gerekli kütüphaneler
├── logs/                   # Eğitim logları (Tensorboard)
│   ├── PPO_1/
│   └── PPO_2/
├── modeller/               # Kaydedilmiş modeller
│   ├── 4 ışık/             # 4 ışıklı kavşak denemeleri
│   │   ├── modelv1/
│   │   └── modelv2/
│   └── solo/               # Tekil kavşak denemeleri
│       └── solov1/
│           ├── egitim/     # Ara kayıtlar (Checkpoints 100k, 200k...)
│           └── ppo_kavsak_model_solov1_final.zip
└── SUMO/                   # Simülasyon harita ve rota dosyaları
    ├── grid_cmd_code.txt
    ├── map/                # Grid şehir haritası (4 Işıklı)
    │   ├── grid_proje.sumocfg
    │   ├── grid_sehir.net.xml
    │   └── traffic.rou.xml
    └── map_solo/           # Tekil kavşak haritası
        ├── simulasyon.sumocfg
        └── solo.net.xml
```
## 📊 Örnek Görseller
# Tek Kavşak Modeli(solov1) Demo

![Kayt2026-01-19112834-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/a8a31ba6-6900-48f8-958b-ccf267d0bcb0)


---
Her türlü geri bildirim ve katkıya açığım! İletişim için [LinkedIn Profilim](https://www.linkedin.com/in/erg%C3%BCn-enes-yaz%C4%B1rl%C4%B1o%C4%9Flu-282829292/) üzerinden ulaşabilirsiniz.
