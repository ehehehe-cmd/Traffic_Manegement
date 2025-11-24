import traci
import os
import sys

# DÜZELTME 1: Dosya yolunu düzgün tanımlama
# Windows'ta sorun yaşamamak için ya r"..." kullanırız ya da "/" işareti.
# Eğer python dosyan ile .sumocfg aynı klasördeyse direkt ismini yazman yeterli.
# Eğer SUMO klasörünün içindeyse: "SUMO/test.sumocfg"
config_dosyasi = "SUMO/test.sumocfg" 

# DÜZELTME 2: Komutu bir LİSTE olarak hazırlama
# sumo-gui: Programın adı
# -c: Config dosyasını yükle komutu
sumoCmd = ["sumo-gui", "-c", config_dosyasi]

# Kendi haritandaki ŞERİT (Lane) ID'lerini buraya yazmalısın.
# Genelde EdgeID_0 şeklindedir.
yollar = {
    "Kuzey": "E0", 
    "Guney": "-E3",
    "Dogu":  "-E1",
    "Bati":  "-E2"
}

TLS_ID = "J1"
# Hangi yön hangi FAZ'da yeşil yanıyor? (Genelde 0 ve 2'dir)
# Faz 0: Kuzey-Güney Yeşil
# Faz 2: Doğu-Batı Yeşil
PHASE_NS_GREEN = 0 
PHASE_EW_GREEN = 2

# Zamanlayıcılar
MIN_YESIL_SURE = 20  # Işık en az 10 saniye yeşil kalsın (Zırt pırt değişmesin)

# Araç Puanları
PUANLAR = {
    "car": 1,
    "bus": 2,
    "ambulance": 1000  # Ambulans görünce sistem çıldırmalı :)
}

def akilli_yogunluk_hesapla():
    # Skorları tutacak sözlük: {'NS': 10, 'EW': 5}
    # NS: North-South (Kuzey-Güney), EW: East-West (Doğu-Batı)
    skorlar = {"NS": 0, "EW": 0}

    for yon, edge_id in yollar.items():
        try:
            serit_sayisi = traci.edge.getLaneNumber(edge_id)
            for i in range(serit_sayisi):
                serit_id = f"{edge_id}_{i}"
                araclar = traci.lane.getLastStepVehicleIDs(serit_id)
                
                for arac_id in araclar:
                    arac_tipi = traci.vehicle.getTypeID(arac_id)
                    puan = PUANLAR.get(arac_tipi, 1)
                    
                    # Puanları Grupla (Kuzey+Güney bir takım, Doğu+Batı bir takım)
                    if yon in ["Kuzey", "Guney"]:
                        skorlar["NS"] += puan
                    else:
                        skorlar["EW"] += puan
                    
                    if puan >= 1000:
                        print(f"🚨 ACİL DURUM: {yon} yönünde araç tespit edildi! 🚨")

        except Exception as e:
            # Hata olursa (örn yol boşsa) devam et
            pass
            
    return skorlar

# --- SİMÜLASYON ---
traci.start(sumoCmd)
print("Akıllı Trafik Işığı Sistemi Başlatıldı...")

last_switch_step = 0
current_phase_group = "NS" # Başlangıçta NS yeşil varsayalım

step = 0
while step < 3600:
    traci.simulationStep()
    
    # 1. Verileri Topla
    skorlar = akilli_yogunluk_hesapla()
    ns_score = skorlar["NS"]
    ew_score = skorlar["EW"]

    # 2. Şu an geçen süre
    gecen_sure = step - last_switch_step

    # 3. KARAR MEKANİZMASI
    
    # Durum A: Ambulans Varsa (Acil Müdahale)
    if ns_score >= 1000 and current_phase_group != "NS":
        print("🚑 AMBULANS GEÇİŞİ İÇİN KUZEY-GÜNEY AÇILIYOR!")
        traci.trafficlight.setPhase(TLS_ID, PHASE_NS_GREEN)
        current_phase_group = "NS"
        last_switch_step = step # Süreyi sıfırla

    elif ew_score >= 1000 and current_phase_group != "EW":
        print("🚑 AMBULANS GEÇİŞİ İÇİN DOĞU-BATI AÇILIYOR!")
        traci.trafficlight.setPhase(TLS_ID, PHASE_EW_GREEN)
        current_phase_group = "EW"
        last_switch_step = step

    # Durum B: Normal Trafik (En az 10 saniye geçmişse kontrol et)
    elif gecen_sure > MIN_YESIL_SURE:
        
        # Eğer Doğu-Batı çok daha kalabalıksa ve şu an NS yanıyorsa -> DEĞİŞTİR
        if ew_score > ns_score and current_phase_group == "NS":
            print(f"🔄 Trafik Yönü Değişiyor: DOĞU-BATI (Skor: {ew_score} vs {ns_score})")
            traci.trafficlight.setPhase(TLS_ID, PHASE_EW_GREEN)
            current_phase_group = "EW"
            last_switch_step = step
            
        # Eğer Kuzey-Güney çok daha kalabalıksa ve şu an EW yanıyorsa -> DEĞİŞTİR
        elif ns_score > ew_score and current_phase_group == "EW":
            print(f"🔄 Trafik Yönü Değişiyor: KUZEY-GÜNEY (Skor: {ns_score} vs {ew_score})")
            traci.trafficlight.setPhase(TLS_ID, PHASE_NS_GREEN)
            current_phase_group = "NS"
            last_switch_step = step

    # Debug için her 10 adımda bir yaz
    if step % 10 == 0:
        sys.stdout.write(f"\rStep {step} | NS Skor: {ns_score} | EW Skor: {ew_score} | Aktif: {current_phase_group}")
        sys.stdout.flush()

    step += 1

traci.close()