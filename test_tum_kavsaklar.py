import os
import gymnasium as gym
from stable_baselines3 import PPO
import sumo_rl

# --- AYARLAR ---
HARITA_DOSYASI = "SUMO\map\grid_sehir.net.xml"
TRAFIK_DOSYASI = "SUMO\map\\traffic.rou.xml"
MODEL_DOSYASI = "trafik_yonetici_4_kavsak_final" 

# --- ÇOK ÖNEMLİ AYAR ---
# Eğitimde 15 yaptıysan, burada da 15 OLMAK ZORUNDA!
KARAR_SURESI = 15 

def main():
    print("🕵️‍♂️ DETEKTİF MODU: Modelin ne düşündüğünü izliyoruz...")

    env = sumo_rl.SumoEnvironment(
        net_file=HARITA_DOSYASI,
        route_file=TRAFIK_DOSYASI,
        use_gui=True,              
        num_seconds=3600,          
        min_green=5,
        delta_time=KARAR_SURESI,    # <--- BURASI 15 OLMALI
        reward_fn='pressure',
        single_agent=False          # Tüm kavşaklar
    )

    try:
        model = PPO.load(MODEL_DOSYASI)
        print("✅ Beyin Yüklendi.")
    except FileNotFoundError:
        print(f"❌ '{MODEL_DOSYASI}' bulunamadı.")
        return

    # Resetleme Mantığı (Hata önleyici)
    reset_return = env.reset()
    if isinstance(reset_return, tuple):
        obs = reset_return[0]
    else:
        obs = reset_return

    done = {'__all__': False}
    
    step_sayaci = 0
    while not done['__all__']:
        actions = {}
        
        print(f"\n--- Adım {step_sayaci} ---")
        
        for agent_id in obs.keys():
            agent_obs = obs[agent_id]
            
            # Deterministic=False yapalım ki bazen risk alabilsin (Test amaçlı)
            action, _states = model.predict(agent_obs, deterministic=True)
            
            actions[agent_id] = action
            
            # KONSOLA YAZDIR: Hangi kavşak ne yapmak istiyor?
            # Action 0 veya 1 genelde "Koru", 2 veya 3 "Değiştir" olabilir (Faz yapısına göre)
            print(f"🚦 {agent_id} -> Karar: {action}")
        
        step_return = env.step(actions)
        
        if len(step_return) == 5:
            obs, rewards, terminations, truncations, info = step_return
            done = terminations
        else:
            obs, rewards, done, info = step_return
            if not isinstance(done, dict): done = {'__all__': done}
            
        step_sayaci += 1

    env.close()

if __name__ == "__main__":
    main()