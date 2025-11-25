import os
import gymnasium as gym
from stable_baselines3 import PPO
import sumo_rl

# --- AYARLAR ---
HARITA_DOSYASI = "SUMO\map\grid_sehir.net.xml"
TRAFIK_DOSYASI = "SUMO\map\\traffic.rou.xml"
MODEL_DOSYASI = "trafik_yonetici_ppo_final" 

def main():
    print("🎬 TÜM KAVŞAKLAR İÇİN GÖSTERİ BAŞLIYOR...")
    print("Not: 'single_agent=False' yaptık, artık herkesi yöneteceksin.")

    # 1. ORTAMI OLUŞTUR (MULTI-AGENT MODU)
    env = sumo_rl.SumoEnvironment(
        net_file=HARITA_DOSYASI,
        route_file=TRAFIK_DOSYASI,
        use_gui=True,              
        num_seconds=3600,          
        min_green=5,
        delta_time=5,
        reward_fn='diff-waiting-time',
        single_agent=False          # Çoklu Ajan Modu
    )

    # 2. MODELİ YÜKLE
    try:
        model = PPO.load(MODEL_DOSYASI)
        print("✅ Beyin Yüklendi.")
    except FileNotFoundError:
        print(f"❌ '{MODEL_DOSYASI}' bulunamadı.")
        return

    # 3. SİMÜLASYONU BAŞLAT (HATA ÇÖZÜMÜ BURADA)
    # env.reset() bazen tek (obs), bazen çift (obs, info) döner.
    # Bunu kontrol altına alıyoruz:
    reset_return = env.reset()
    
    if isinstance(reset_return, tuple):
        # Eğer (obs, info) döndüyse:
        obs = reset_return[0]
    else:
        # Eğer sadece obs döndüyse:
        obs = reset_return

    # Done (Bitti) kontrolü için
    done = {'__all__': False}
    
    while not done['__all__']:
        actions = {}
        
        # --- PARAMETRE PAYLAŞIMI ---
        # Haritadaki her kavşak için aynı beyni kullanıyoruz
        for agent_id in obs.keys():
            agent_obs = obs[agent_id]
            action, _states = model.predict(agent_obs, deterministic=True)
            actions[agent_id] = action
        
        # Adım at (Step)
        step_return = env.step(actions)
        
        # Step dönüşü de versiyona göre değişebilir (4'lü veya 5'li olabilir)
        if len(step_return) == 5:
            obs, rewards, terminations, truncations, info = step_return
            done = terminations # Yeni versiyonlarda 'terminations' kullanılır
        else:
            obs, rewards, done, info = step_return # Eski versiyon
            
            # Eğer done bir sözlük değilse (tek ajan gibi döndüyse) düzelt
            if not isinstance(done, dict):
                done = {'__all__': done}

    print("Gösteri bitti.")
    env.close()

if __name__ == "__main__":
    main()