import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import sumo_rl

# --- AYARLAR ---
HARITA_DOSYASI = "SUMO\map\grid_sehir.net.xml"
TRAFIK_DOSYASI = "SUMO\map\\traffic.rou.xml"
MODEL_ADI = "trafik_yonetici_ppo"
SIMULASYON_SURESI = 5000 # Adım sayısı (Saniye)

def main():
    print("🤖 Trafik Yapay Zekası Eğitimi Başlıyor...")
    print(f"Harita: {HARITA_DOSYASI}")
    print("Not: Pencere AÇILMAYACAK (Hız için). Sabırlı olun...")

    # 1. ORTAMI OLUŞTUR
    # single_agent=True: PPO'nun hata vermemesi için tek bir ajanı yönetir.
    env = sumo_rl.SumoEnvironment(
        net_file=HARITA_DOSYASI,
        route_file=TRAFIK_DOSYASI,
        use_gui=False,             # Eğitimde grafik arayüzü kapatıyoruz
        num_seconds=SIMULASYON_SURESI,
        min_green=5,
        delta_time=5,
        reward_fn='diff-waiting-time',
        single_agent=True          # <--- KRİTİK AYAR (Hata almamak için)
    )

    # Ortamı loglama için Monitor ile, uyumluluk için DummyVecEnv ile sarıyoruz
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # 2. MODELİ OLUŞTUR (PPO)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003,
        batch_size=256
    )

    # 3. EĞİTİMİ BAŞLAT
    # 50.000 adım yaklaşık 10-15 dakika sürebilir (bilgisayar hızına göre)
    EGITIM_ADIM = 50000 
    print(f"Hedeflenen Adım Sayısı: {EGITIM_ADIM}. Başlıyor...")
    
    model.learn(total_timesteps=EGITIM_ADIM)

    # 4. KAYDET
    model.save(MODEL_ADI)
    print(f"\n✅ Eğitim tamamlandı! Model '{MODEL_ADI}.zip' olarak kaydedildi.")
    
    env.close()

if __name__ == "__main__":
    main()