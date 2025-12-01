import os
import sys
import gymnasium as gym
import numpy as np

# Windows Hızlandırması
if os.name != 'nt':
    os.environ['LIBSUMO_AS_TRACI'] = '1'

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack # <--- YENİ: HAFIZA
from stable_baselines3.common.monitor import Monitor
import sumo_rl

# --- AYARLAR ---
HARITA_DOSYASI = "SUMO\map\grid_sehir.net.xml"
TRAFIK_DOSYASI = "SUMO\map\\traffic.rou.xml"
MODEL_ADI = "trafik_yonetici_hafizali"

KARAR_SURESI = 10  # Biraz daha sık baksın (Hafızası var artık)
MIN_YESIL = 5
SIMULASYON_SURESI = 5000
ISLEM_SAYISI = 4   

# --- ADAPTER (Aynı Kalıyor) ---
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
        actions = {agent: action for agent in self.possible_agents}
        obs_dict, rewards, terminations, truncations, infos = self.pz_env.step(actions)
        
        # --- ÖDÜL AYARI (Daha Agresif) ---
        # Sabit sistemi yenmek için modelin kuyrukları eritmesi lazım.
        # Değişim cezasını çok azalttık (Sadece gereksiz titreşimi önlesin diye)
        
        switch_penalty = 0
        if self.last_action is not None and action != self.last_action:
            switch_penalty = 2 # Çok küçük bir ceza (Sadece gürültü önleyici)
            
        # Pressure ödülünü normalize et (Model daha iyi anlasın diye)
        # Genelde pressure -1000 ile -5000 arasıdır. Bunu küçültüyoruz.
        total_reward = (sum(rewards.values()) * 0.01) - switch_penalty
        
        self.last_action = action
        
        obs = obs_dict[self.possible_agents[0]]
        done = any(terminations.values()) or any(truncations.values())
        info = infos[self.possible_agents[0]]
        return obs, total_reward, done, False, info

def make_env(rank):
    def _init():
        env = sumo_rl.parallel_env(
            net_file=HARITA_DOSYASI,
            route_file=TRAFIK_DOSYASI,
            use_gui=False,
            num_seconds=SIMULASYON_SURESI,
            min_green=MIN_YESIL,
            delta_time=KARAR_SURESI,
            reward_fn='pressure', 
        )
        env = PettingZooToGymAdapter(env)
        env = Monitor(env)
        return env
    return _init

def main():
    print(f"🧠 HAFIZALI (FRAME STACK) EĞİTİM BAŞLIYOR...")
    
    # 1. Ortamları Oluştur
    env = SubprocVecEnv([make_env(i) for i in range(ISLEM_SAYISI)])

    # 2. HAFIZA EKLE (VecFrameStack)
    # Model artık son 4 kareyi üst üste koyup görür.
    # Böylece araçların HIZINI ve AKIŞ YÖNÜNÜ anlar.
    env = VecFrameStack(env, n_stack=4) 

    # 3. Model (Beyni biraz büyüttük: [256, 256])
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,
        batch_size=1024, # Daha büyük batch = Daha stabil öğrenme
        n_steps=2048,
        # Beyin kapasitesini artırıyoruz (Daha karmaşık stratejiler için)
        policy_kwargs=dict(net_arch=[256, 256]) 
    )

    # 4. EĞİTİM (UZUN SOLUKLU)
    # Sabit sistemi yenmek istiyorsan en az 500k - 1M lazım.
    EGITIM_ADIM = 500000 
    print(f"Hedef: {EGITIM_ADIM} adım. (Bu uzun sürebilir, sabırlı ol)")
    
    model.learn(total_timesteps=EGITIM_ADIM)

    model.save(MODEL_ADI)
    print(f"\n✅ Hafızalı Model Eğitildi! '{MODEL_ADI}.zip'")
    env.close()

if __name__ == "__main__":
    main()