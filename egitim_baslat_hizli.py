import os
import sys

# --- HIZLANDIRICI 1: LIBSUMO ---
# Bu, Python ile SUMO'nun ram üzerinden konuşmasını sağlar (Çok daha hızlıdır)
# Eğer hata alırsan bu satırı sil.
if os.name != 'nt': # Windows dışındaysa kesin çalışır, Windows'ta dener.
    os.environ['LIBSUMO_AS_TRACI'] = '1'

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv # Tekli ama güvenli
from stable_baselines3.common.monitor import Monitor
import sumo_rl

# --- AYARLAR ---
HARITA_DOSYASI = "SUMO\map\grid_sehir.net.xml"
TRAFIK_DOSYASI = "SUMO\map\\traffic.rou.xml"
MODEL_ADI = "trafik_yonetici_ppo_final"

# --- HIZLANDIRICI 2: KISA BÖLÜMLER ---
# Trafiğin birikmesine izin vermeden reset atacağız.
SIMULASYON_SURESI = 3000  # 5000 yerine 3000. Daha sık reset = Daha az kilitlenme.

def main():
    print("🚀 Stabil ve Hızlı Eğitim Başlatılıyor...")
    print("Trafik yoğunluğu düşürüldü ve süre optimize edildi.")

    # Ortamı oluştur
    env = sumo_rl.SumoEnvironment(
        net_file=HARITA_DOSYASI,
        route_file=TRAFIK_DOSYASI,
        use_gui=False,
        num_seconds=SIMULASYON_SURESI,
        min_green=5,
        delta_time=5,
        reward_fn='diff-waiting-time',
        single_agent=True
    )
    
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # Modeli oluştur
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,
        batch_size=512,
        n_steps=2048 
    )

    EGITIM_ADIM = 100000 
    print(f"Hedef: {EGITIM_ADIM} adım. Başlıyor...")
    
    model.learn(total_timesteps=EGITIM_ADIM)

    model.save(MODEL_ADI)
    print(f"\n✅ Eğitim Tamamlandı! Kaydedildi.")
    env.close()

if __name__ == "__main__":
    main()