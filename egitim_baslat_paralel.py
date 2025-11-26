import os
import sys

# Windows Hızlandırması
if os.name != 'nt':
    os.environ['LIBSUMO_AS_TRACI'] = '1'

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import sumo_rl
import supersuit as ss

# --- AYARLAR ---
HARITA_DOSYASI = "SUMO\map\grid_sehir.net.xml"
TRAFIK_DOSYASI = "SUMO\map\\traffic.rou.xml"
MODEL_ADI = "trafik_yonetici_4_kavsak_hizli"
KARAR_SURESI = 15  
MIN_YESIL = 10
SIMULASYON_SURESI = 4000
ISLEM_SAYISI = 4   # Çekirdek sayısı

def main():
    print(f"🚀 MULTI-AGENT PARALEL EĞİTİM BAŞLIYOR ({ISLEM_SAYISI} ÇEKİRDEK)...")
    
    # 1. ORTAMI OLUŞTUR
    env = sumo_rl.parallel_env(
        net_file=HARITA_DOSYASI,
        route_file=TRAFIK_DOSYASI,
        use_gui=False,
        num_seconds=SIMULASYON_SURESI,
        min_green=MIN_YESIL,
        delta_time=KARAR_SURESI,
        reward_fn='pressure', 
    )

    # --- HATA DÜZELTME YAMASI (PATCH) ---
    # SuperSuit kütüphanesi 'render_mode' arıyor ama bulamıyor.
    # Biz de "var gibi" davranıyoruz.
    try:
        env.unwrapped.render_mode = "rgb_array"
    except AttributeError:
        # Bazı versiyonlarda direkt env üzerine yazmak gerekir
        env.render_mode = "rgb_array"
    # ------------------------------------

    # 2. VEKTÖRİZASYON (SuperSuit)
    # Artık hata vermemesi lazım çünkü render_mode'u elle ekledik.
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    # 3. PARALELLEŞTİRME (Multiprocessing)
    # concat_vec_envs_v1: İşlemcilere dağıtır.
    # num_vec_envs=1 diyoruz çünkü concat zaten kopyalayacak. 
    # Ama SuperSuit mantığında eldeki env'i çoğaltmak için num_vec_envs'i toplam sayı yapıyoruz.
    env = ss.concat_vec_envs_v1(env, num_vec_envs=ISLEM_SAYISI, num_cpus=ISLEM_SAYISI, base_class='stable_baselines3')

    # Monitor ekle (Loglama için)
    env = Monitor(env)

    # 4. MODEL OLUŞTUR
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,
        batch_size=512,  
        n_steps=1024,
        device='auto'
    )

    # 5. EĞİTİMİ BAŞLAT
    EGITIM_ADIM = 200000 
    print(f"Hedef: {EGITIM_ADIM} adım. RAM kullanımı artabilir...")
    
    model.learn(total_timesteps=EGITIM_ADIM)

    model.save(MODEL_ADI)
    print(f"\n✅ Hızlı Multi-Agent Eğitim Tamamlandı! '{MODEL_ADI}.zip'")
    
    env.close()

if __name__ == "__main__":
    main()