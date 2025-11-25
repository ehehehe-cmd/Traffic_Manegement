import os
import gymnasium as gym
from stable_baselines3 import PPO
import sumo_rl

# --- AYARLAR ---
HARITA_DOSYASI = "SUMO\map\grid_sehir.net.xml"
TRAFIK_DOSYASI = "SUMO\map\\traffic.rou.xml"
# Oluşan zip dosyasının tam adı (uzantısız yazabilirsin)
MODEL_DOSYASI = "trafik_yonetici_ppo_final" 

def main():
    print("🎬 GÖSTERİ BAŞLIYOR...")
    print("SUMO Penceresi açıldığında 'Play' (Yeşil Oynat) tuşuna basmayı unutma!")

    # 1. ORTAMI OLUŞTUR (BU SEFER GUI AÇIK!)
    env = sumo_rl.SumoEnvironment(
        net_file=HARITA_DOSYASI,
        route_file=TRAFIK_DOSYASI,
        use_gui=True,              # <--- İŞTE BÜYÜ BURADA: True yaptık!
        num_seconds=3600,          # 1 saatlik simülasyonu izleyelim
        min_green=5,
        delta_time=5,
        reward_fn='diff-waiting-time',
        single_agent=True          # Eğitimdeki ayarın aynısı olmalı
    )

    # 2. EĞİTİLMİŞ BEYNİ YÜKLE
    # Eğer dosya bulunamadı hatası alırsan ismini kontrol et
    try:
        model = PPO.load(MODEL_DOSYASI)
        print("✅ Yapay Zeka Modeli Başarıyla Yüklendi.")
    except FileNotFoundError:
        print(f"❌ HATA: '{MODEL_DOSYASI}.zip' bulunamadı! Dosya ismini kontrol et.")
        return

    # 3. SİMÜLASYONU BAŞLAT
    obs, info = env.reset()
    done = False
    
    while not done:
        # Modelden bir hamle iste (Deterministic=True: En iyi bildiği hamleyi yapsın, macera aramasın)
        action, _states = model.predict(obs, deterministic=True)
        
        # Hamleyi uygula
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    print("Gösteri bitti.")
    env.close()

if __name__ == "__main__":
    main()