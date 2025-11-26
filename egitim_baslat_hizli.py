import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import sumo_rl
import supersuit as ss

# --- AYARLAR ---
HARITA_DOSYASI = "SUMO\map\grid_sehir.net.xml"
TRAFIK_DOSYASI = "SUMO\map\\traffic.rou.xml"
MODEL_ADI = "trafik_yonetici_4_kavsak_final"

# Windows Libsumo ayarı (Hata alırsan sil)
if os.name != 'nt':
    os.environ['LIBSUMO_AS_TRACI'] = '1'

# --- MANTIK AYARLARI ---
KARAR_SURESI = 15  
MIN_YESIL = 10
SIMULASYON_SURESI = 4500 

def main():
    print("🚦 MULTI-AGENT EĞİTİM (RENDER_MODE TAMİRLİ)...")
    
    # 1. ORTAMI OLUŞTUR
    env = sumo_rl.parallel_env(
        net_file=HARITA_DOSYASI,
        route_file=TRAFIK_DOSYASI,
        use_gui=False,
        num_seconds=SIMULASYON_SURESI,
        min_green=MIN_YESIL,
        delta_time=KARAR_SURESI,
        reward_fn='pressure' 
    )

    # --- HATA ÇÖZÜCÜ YAMA (MONKEY PATCH) ---
    # SuperSuit'in aradığı 'render_mode' özelliğini elle ekliyoruz.
    # use_gui=False olduğu için modu 'rgb_array' veya None diyebiliriz.
    env.unwrapped.render_mode = "rgb_array"
    # ----------------------------------------

    # 2. SARMALAMA (WRAPPING)
    # Artık hata vermeyecek çünkü render_mode özelliğini ekledik.
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    # 3. İŞLEMCİLERİ BİRLEŞTİR
    # Windows hatası olmaması için num_cpus=0 (Ana işlemcide çalıştır)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=0, base_class='stable_baselines3')

    # 4. MODELİ OLUŞTUR
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,
        batch_size=512,
        n_steps=1024
    )

    # 5. EĞİTİMİ BAŞLAT
    EGITIM_ADIM = 100000 
    print(f"Hedef: {EGITIM_ADIM} adım. Başlıyor...")
    
    model.learn(total_timesteps=EGITIM_ADIM)

    model.save(MODEL_ADI)
    print(f"\n✅ 4 Kavşaklı Model Başarıyla Eğitildi! '{MODEL_ADI}.zip'")
    env.close()

if __name__ == "__main__":
    main()