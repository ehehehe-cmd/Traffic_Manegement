import gymnasium as gym
from stable_baselines3 import PPO
import os
import time
import traci

# Senin adaptör dosyan (adaptor.py dosyasının yanında olmalı bu kod)
from adaptor import SUMOTrafikOrtami 

# --- AYARLAR ---
# Dosya yollarını kendi bilgisayarına göre kontrol et
NET_DOSYASI = r"SUMO\map_solo\solo.net.xml"

# En son kaydedilen modelin tam adı (Uzantısı .zip olsun veya olmasın fark etmez)
MODEL_YOLU = "modeller\solo\solov4\ppo_kavsak_model_solov4_final" 

def testi_baslat():
    print("--- 🚦 GÖRSEL TEST BAŞLIYOR 🚦 ---")
    
    # 1. ORTAMI HAZIRLA
    # 'use_gui=True' parametresini ekledim. Eğer adaptor.py'ni güncellemediysen
    # hata verebilir, aşağıda try-except ile hallediyoruz.
    
    try:
        env = SUMOTrafikOrtami(NET_DOSYASI, use_gui=True)
    except TypeError:
        # Eğer adaptor.py eski halindeyse (parametre almıyorsa):
        print("Uyarı: Adaptör eski sürüm, manuel GUI yaması yapılıyor...")
        env = SUMOTrafikOrtami(NET_DOSYASI, use_gui=True)
        # Manuel olarak komutu sumo-gui'ye çeviriyoruz
        if env.sumo_cmd[0] == "sumo":
            env.sumo_cmd[0] = "sumo-gui"
            # Otomatik başlatma ve çıkış komutlarını ekleyelim
            env.sumo_cmd.extend(["--start", "true", "--quit-on-end", "true"])

    # 2. MODELİ YÜKLE
    print(f"Model yükleniyor: {MODEL_YOLU}...")
    try:
        model = PPO.load(MODEL_YOLU)
        print("✅ Model başarıyla yüklendi!")
    except FileNotFoundError:
        print(f"❌ HATA: '{MODEL_YOLU}.zip' dosyası bulunamadı!")
        return

    # 3. SİMÜLASYON DÖNGÜSÜ
    obs, info = env.reset()
    done = False
    toplam_odul = 0
    adim_sayisi = 0
    
    print("\n📺 Simülasyon penceresi açılıyor...")
    print("Eğer otomatik başlamazsa sol üstteki 'Play' (Yeşil Üçgen) tuşuna bas.")
    
    try:
        while not done:
            # deterministic=True : Ajan macera aramaz, öğrendiği EN İYİ hamleyi yapar.
            action, _states = model.predict (obs, deterministic=True)
            
            # Aksiyonu uygula
            obs, reward, terminated, truncated, info = env.step(action)
            
            toplam_odul += reward
            adim_sayisi += 1
            done = terminated or truncated
            
            # Konsola anlık bilgi bas (Opsiyonel)
            if adim_sayisi % 10 == 0:
                print(f"Adım: {adim_sayisi} | Anlık Ödül: {reward:.2f} | Aksiyon: {action}")

            # Gözle takip edebilmek için simülasyonu biraz yavaşlatıyoruz
            # Bilgisayarın çok hızlıysa bu sayıyı 0.1 yapabilirsin
            time.sleep(0.05) 

    except KeyboardInterrupt:
        print("\nTest kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\nBeklenmedik bir hata oluştu: {e}")
    finally:
        print(f"\n--- TEST SONUCU ---")
        print(f"Toplam Adım: {adim_sayisi}")
        print(f"Toplam Puan: {toplam_odul:.2f}")
        print("Simülasyon kapatılıyor...")
        env.close()

if __name__ == "__main__":
    testi_baslat()