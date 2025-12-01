import os
import sys
import gymnasium as gym
import numpy as np  # <--- Hata çözümü için eklendi

# Windows Hızlandırması
if os.name != 'nt':
    os.environ['LIBSUMO_AS_TRACI'] = '1'

from stable_baselines3 import PPO
import sumo_rl
import supersuit as ss

# --- AYARLAR ---
# Dosya yollarının başına 'r' koyuyoruz ki Windows hatası vermesin
HARITA_DOSYASI = r"SUMO\map\grid_sehir.net.xml"
TRAFIK_DOSYASI = r"SUMO\map\traffic.rou.xml"
MODEL_ADI = "trafik_yonetici_bagimsiz"

# MANTIK AYARLARI
KARAR_SURESI = 10 
MIN_YESIL = 5
SIMULASYON_SURESI = 3600

def main():
    print("🎬 TEST MODU BAŞLIYOR...")
    print("Not: SUMO açıldığında 'Play' tuşuna bas ve 'Delay'i 100ms yap.")

    # 1. ORTAMI OLUŞTUR (GUI AÇIK!)
    env = sumo_rl.parallel_env(
        net_file=HARITA_DOSYASI,
        route_file=TRAFIK_DOSYASI,
        use_gui=True,              # <--- İZLEMEK İÇİN TRUE
        num_seconds=SIMULASYON_SURESI,
        min_green=MIN_YESIL,
        delta_time=KARAR_SURESI,
        reward_fn='pressure', 
    )

    # --- YAMA (Eğitimdekiyle aynı yama şart) ---
    try:
        env.unwrapped.render_mode = "rgb_array"
    except AttributeError:
        env.render_mode = "rgb_array"

    # 2. ORTAMI PAKETLE (SuperSuit)
    # Model eğitimi sırasında verileri bu formatta gördü.
    
    # Adım A: Vektörize Et
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    # Adım B: Tek İşlemcide Çalıştır
    # Test yaparken num_vec_envs=1 ve num_cpus=1 yapıyoruz.
    # Böylece tek bir pencere açılır ve bilgisayar kasmaz.
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

    # 3. MODELİ YÜKLE
    try:
        model = PPO.load(MODEL_ADI)
        print(f"✅ Model Yüklendi: {MODEL_ADI}")
    except FileNotFoundError:
        print(f"❌ HATA: '{MODEL_ADI}.zip' bulunamadı! Dosya ismini kontrol et.")
        return

    # 4. SİMÜLASYONU BAŞLAT
    obs = env.reset()
    
    # HATA ÇÖZÜMÜ: while döngüsü yerine for döngüsü kullanıyoruz.
    # VecEnv ortamlarında 'done' bir liste olduğu için while not done hata verir.
    # Biz 10.000 adım boyunca (veya simülasyon bitene kadar) izleyeceğiz.
    
    print("Simülasyon döngüsü başlıyor...")
    
    for step in range(10000):
        # Modelden karar iste
        action, _states = model.predict(obs, deterministic=True)
        
        # Konsola Yazdır: 
        # Eğer [1 0 0 1] gibi karışık sayılar görüyorsan BAĞIMSIZ karar veriyordur!
        print(f"Adım {step} -> Kararlar: {action}") 
        
        # Kararı uygula
        obs, rewards, done, info = env.step(action)
        
        # 'done' bir dizi (array) olarak döner: [False, False, False, False]
        # Eğer herhangi biri True ise (np.any), o simülasyon bitmiş demektir.
        if np.any(done):
            print("--- Bir bölüm tamamlandı, ortam otomatik resetlendi ---")
            # İstersen break diyip çıkabilirsin, ama izlemeye devam edelim.
            # break 

    print("Test bitti.")
    env.close()

if __name__ == "__main__":
    main()