import os
import sys
import gymnasium as gym

# Windows Hızlandırması
if os.name != 'nt':
    os.environ['LIBSUMO_AS_TRACI'] = '1'

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
import sumo_rl
import supersuit as ss

# --- AYARLAR ---
# DÜZELTME: Dosya yollarının başına 'r' koyduk (Raw String).
# Böylece \ işaretleri sorun çıkarmaz.
HARITA_DOSYASI = r"SUMO\mapV2\duz_yol.net.xml"
TRAFIK_DOSYASI = r"SUMO\mapV2\duz_map.rou.xml"
MODEL_ADI = "trafik_yonetici_bagimsiz"

# MANTIK AYARLARI
KARAR_SURESI = 10 
MIN_YESIL = 5
SIMULASYON_SURESI = 4000
ISLEM_SAYISI = 4   # Çekirdek Sayısı

def main():
    print(f"🚀 BAĞIMSIZ MULTI-AGENT EĞİTİM ({ISLEM_SAYISI} Çekirdek)...")
    print("Not: Bu sefer her kavşak KENDİ kararını verecek.")

    # 1. ORTAMI OLUŞTUR (Sınıf miras alma YOK)
    # Direkt fonksiyonu çağırıyoruz.
    env = sumo_rl.parallel_env(
        net_file=HARITA_DOSYASI,
        route_file=TRAFIK_DOSYASI,
        use_gui=False,
        num_seconds=SIMULASYON_SURESI,
        min_green=MIN_YESIL,
        delta_time=KARAR_SURESI,
        reward_fn='pressure', 
    )

    # --- HATA DÜZELTME YAMASI (INSTANCE PATCHING) ---
    # Sınıf oluşturmak yerine, oluşturulmuş nesneye (env)
    # eksik olan özelliği elle yapıştırıyoruz.
    try:
        env.unwrapped.render_mode = "rgb_array"
    except AttributeError:
        env.render_mode = "rgb_array"
    # ------------------------------------------------

    # 2. BAĞIMSIZLAŞTIRMA VE HIZLANDIRMA (SuperSuit)
    
    # Adım A: PettingZoo -> Vektör Ortamı
    # Bu aşamada ortam PPO uyumlu hale gelir.
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    # Adım B: İşlemcilere Dağıt (Paralelleştirme)
    # concat_vec_envs_v1 fonksiyonu bizim için 4 tane işlemci açar.
    # num_vec_envs=ISLEM_SAYISI: Toplam kaç simülasyon dönecek?
    # num_cpus=ISLEM_SAYISI: Kaç çekirdek kullanacak?
    env = ss.concat_vec_envs_v1(env, num_vec_envs=ISLEM_SAYISI, num_cpus=ISLEM_SAYISI, base_class='stable_baselines3')

    # 3. LOGLAMA
    env = VecMonitor(env)

    # 4. MODEL (Meraklı PPO)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,
        batch_size=1024,      
        n_steps=512,
        
        # --- KRİTİK AYAR: ENTROPİ ---
        # 0.05 yaparak modelin "farklı şeyler denemesini" sağlıyoruz.
        # Bu sayede ışıklar senkronize (aynı anda) hareket etmez.
        ent_coef=0.05,        
        
        gamma=0.995,
        device='auto'
    )

    # 5. EĞİTİM
    EGITIM_ADIM = 1000000 
    print(f"Hedef: {EGITIM_ADIM} adım. Başlıyor...")
    
    model.learn(total_timesteps=EGITIM_ADIM)

    model.save(MODEL_ADI)
    print(f"\n✅ Bağımsız Model Eğitildi! '{MODEL_ADI}.zip'")
    env.close()

if __name__ == "__main__":
    main()