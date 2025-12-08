import gymnasium as gym
from stable_baselines3 import PPO
import os
import time

# Senin ortam dosyan
from adaptor import SUMOTrafikOrtami 

# --- AYARLAR ---
# Eğitimde kullandığın yolların aynısı
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
NET_DOSYASI = r"SUMO\map\grid_sehir.net.xml"  # Kendi dosya yolun
ROUTE_DOSYASI = r"SUMO\map\traffic.rou.xml" # Kendi dosya yolun

# Modelin olduğu klasör (Eğitimde değiştirdiğimiz C:/ yolu)
MODEL_YOLU = "ppo_kavsak_modelv2_final.zip" 
# VEYA ara kayıtları denemek istersen:
# MODEL_YOLU = "C:/Trafik_Yapay_Zeka/modeller/ppo_multi_model_50000_steps.zip"

def testi_baslat():
    print("--- GÖRSEL TEST BAŞLIYOR ---")
    
    # 1. Ortamı Oluştur
    # Not: egitim_deneme.py içinde "sumo-gui" yazdığından emin ol!
    env = SUMOTrafikOrtami(NET_DOSYASI, ROUTE_DOSYASI)

    # 2. Eğitilmiş Modeli Yükle
    # İster "final_model"i, ister en iyi checkpoint'i yükleyebilirsin.
    # Örn: model_yolu = os.path.join(KAYIT_KLASORU, "ppo_multi_final_model")
    
    model_adi = "ppo_multi_final_model" # Uzantısız yaz (.zip gerekmez)
    model_yolu = os.path.join(MODEL_YOLU)
    
    print(f"Model yükleniyor: {model_yolu}")
    
    try:
        model = PPO.load(model_yolu)
    except:
        print("HATA: Model dosyası bulunamadı! İsmi veya klasörü kontrol et.")
        return

    # 3. Simülasyon Döngüsü
    obs, info = env.reset()
    bitti = False
    toplam_odul = 0
    
    print("\nSimülasyon penceresi açıldı!")
    print("İzlemek için SUMO penceresindeki 'Play' (Yeşil Üçgen) tuşuna bas.")
    
    while not bitti:
        # Modelden tahmin al (Deterministic=True, rastgelelik yapma en iyi bildiğini yap demek)
        action, _states = model.predict(obs, deterministic=True)
        
        # Ortamda uygula
        obs, reward, terminated, truncated, info = env.step(action)
        
        toplam_odul += reward
        bitti = terminated or truncated
        
        # Biraz yavaşlat ki gözle takip edebilelim (Opsiyonel)
        # time.sleep(0.05) 

    print(f"--- TEST BİTTİ ---")
    print(f"Toplam Puan (Reward): {toplam_odul}")
    
    # Hemen kapanmasın diye bekle
    input("Kapatmak için Enter'a bas...")
    env.close()

if __name__ == "__main__":
    testi_baslat()