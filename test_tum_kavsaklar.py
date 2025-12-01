import os
import sys
import gymnasium as gym
import numpy as np

# Windows Libsumo ayarı (Hız için)
if os.name != 'nt':
    os.environ['LIBSUMO_AS_TRACI'] = '1'

from stable_baselines3 import PPO
# --- KRİTİK İMPORTLAR ---
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import sumo_rl

# --- AYARLAR ---
# 2 Şeritli harita ile eğittiysen burayı ona göre güncelle!
HARITA_DOSYASI = "SUMO\map\grid_sehir.net.xml"
TRAFIK_DOSYASI = "SUMO\map\\traffic.rou.xml"

# Eğittiğin son modelin tam adı (uzantısız)
MODEL_DOSYASI = "trafik_yonetici_hafizali" 

# Eğitimdeki ayar neyse o olmalı (10 veya 15)
KARAR_SURESI = 10 

# --- EĞİTİMDE KULLANDIĞIMIZ ADAPTER SINIFI ---
# Bu sınıf olmadan model çalışmaz, çünkü model bu sınıftan gelen veriye göre eğitildi.
class PettingZooToGymAdapter(gym.Env):
    def __init__(self, pz_env):
        self.pz_env = pz_env
        self.possible_agents = pz_env.possible_agents
        self.observation_space = pz_env.observation_space(self.possible_agents[0])
        self.action_space = pz_env.action_space(self.possible_agents[0])
        self.render_mode = "rgb_array"
        self.metadata = {"render_modes": ["rgb_array"]}
        self.last_action = None 

    def reset(self, seed=None, options=None):
        self.last_action = None
        obs_dict, info_dict = self.pz_env.reset(seed=seed, options=options)
        return obs_dict[self.possible_agents[0]], info_dict[self.possible_agents[0]]

    def step(self, action):
        # Tek aksiyonu tüm ajanlara yay (Parameter Sharing)
        actions = {agent: action for agent in self.possible_agents}
        obs_dict, rewards, terminations, truncations, infos = self.pz_env.step(actions)
        
        # Sadece izliyoruz, ödül hesaplamaya gerek yok ama format bozulmasın
        total_reward = sum(rewards.values())
        self.last_action = action
        
        obs = obs_dict[self.possible_agents[0]]
        done = any(terminations.values()) or any(truncations.values())
        info = infos[self.possible_agents[0]]
        return obs, total_reward, done, False, info

def main():
    print("🧠 HAFIZALI MODEL TEST EDİLİYOR...")
    print(f"Harita: {HARITA_DOSYASI}")

    # 1. TEMEL ORTAMI OLUŞTUR (GUI AÇIK)
    # Burada direkt sumo_rl.SumoEnvironment değil, parallel_env kullanıyoruz
    # çünkü Adapter sınıfımız parallel_env bekliyor.
    env = sumo_rl.parallel_env(
        net_file=HARITA_DOSYASI,
        route_file=TRAFIK_DOSYASI,
        use_gui=True,              # <--- İZLEMEK İÇİN AÇIK
        num_seconds=3600,
        min_green=5,
        delta_time=KARAR_SURESI,
        reward_fn='pressure',
    )

    # 2. ADAPTER İLE SAR
    env = PettingZooToGymAdapter(env)

    # 3. VEKTÖR ORTAMI YAP (SB3 Uyumu için)
    # VecFrameStack kullanabilmek için ortamın DummyVecEnv olması şarttır.
    env = DummyVecEnv([lambda: env])

    # 4. HAFIZA EKLE (VecFrameStack)
    # --- EN ÖNEMLİ KISIM BURASI ---
    # Model 4 kare hafızalı eğitildiği için testte de 4 kare vermeliyiz.
    env = VecFrameStack(env, n_stack=4)

    # 5. MODELİ YÜKLE
    try:
        model = PPO.load(MODEL_DOSYASI)
        print("✅ Hafızalı Beyin Yüklendi.")
    except FileNotFoundError:
        print(f"❌ HATA: '{MODEL_DOSYASI}.zip' bulunamadı.")
        return

    # 6. SİMÜLASYONU BAŞLAT
    obs = env.reset() # VecEnv olduğu için direkt obs döner (info dönmez)
    
    done = False
    step = 0
    
    while not done:
        # Modelden karar iste
        action, _states = model.predict(obs, deterministic=True)
        
        # Konsola yaz
        print(f"Adım {step} -> Karar: {action[0]}") # VecEnv olduğu için action bir liste gelir
        
        # Kararı uygula
        obs, rewards, done, info = env.step(action)
        
        step += 1

    print("Test bitti.")
    env.close()

if __name__ == "__main__":
    main()