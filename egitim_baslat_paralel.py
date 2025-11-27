import os
import sys
import gymnasium as gym
import numpy as np

# Windows Hızlandırması
if os.name != 'nt':
    os.environ['LIBSUMO_AS_TRACI'] = '1'

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import sumo_rl

# --- AYARLAR ---
HARITA_DOSYASI = "SUMO\map\grid_sehir.net.xml"
TRAFIK_DOSYASI = "SUMO\map\\traffic.rou.xml"
MODEL_ADI = "trafik_yonetici_4_kavsak_ceza01"
KARAR_SURESI = 15  
MIN_YESIL = 10
SIMULASYON_SURESI = 4000
ISLEM_SAYISI = 15   # Çekirdek Sayısı

# --- ÖZEL ADAPTER SINIFI (Wrapper Değil!) ---
# gym.Wrapper yerine direkt gym.Env kullanıyoruz.
# Böylece "AssertionError" hatasını baypas ediyoruz.
class PettingZooToGymAdapter(gym.Env):
    def __init__(self, pz_env):
        self.pz_env = pz_env
        self.possible_agents = pz_env.possible_agents
        
        # İlk ajanın özelliklerini alıp Gym standardı yapıyoruz
        self.observation_space = pz_env.observation_space(self.possible_agents[0])
        self.action_space = pz_env.action_space(self.possible_agents[0])
        
        # Render mode
        self.render_mode = "rgb_array"
        self.metadata = {"render_modes": ["rgb_array"]}

        # --- EKLEMEN GEREKEN YER (1) ---
        # Sınıf ilk yaratıldığında "Daha önce hiçbir şey yapmadım" diyoruz.
        self.last_action = None 
        # -------------------------------

    def reset(self, seed=None, options=None):
        # --- EKLEMEN GEREKEN YER (2) ---
        # Oyun sıfırlandığında hafızayı da sıfırlayalım
        self.last_action = None
        # -------------------------------

        obs_dict, info_dict = self.pz_env.reset(seed=seed, options=options)
        agent_id = self.possible_agents[0]
        return obs_dict[agent_id], info_dict[agent_id]

    def step(self, action):
        # Aksiyonları dağıt
        actions = {agent: action for agent in self.possible_agents}
        obs_dict, rewards, terminations, truncations, infos = self.pz_env.step(actions)
        
        # --- YENİ ÖDÜL MANTIĞI ---
        # 1. Ham Baskı Puanını Al (Negatif bir sayıdır)
        raw_pressure = sum(rewards.values())
        
        # 2. Değişim Cezası (Switch Penalty)
        switch_penalty = 0
        if self.last_action is not None and action != self.last_action:
            switch_penalty = 10  # Cezayı 10 yaptık
        
        # 3. Nihai Ödül:
        # Baskıyı biraz küçültüyoruz (0.05 ile çarpıp) ki değişim cezasını yutmasın.
        # Böylece model hem trafiği hem de değişimi dengeli görür.
        total_reward = (raw_pressure * 0.05) - switch_penalty
        
        self.last_action = action
        # -------------------------

        obs = obs_dict[self.possible_agents[0]]
        done = any(terminations.values()) or any(truncations.values())
        info = infos[self.possible_agents[0]]
        
        return obs, total_reward, done, False, info

def make_env(rank):
    def _init():
        # 1. SUMO Parallel Env oluştur
        env = sumo_rl.parallel_env(
            net_file=HARITA_DOSYASI,
            route_file=TRAFIK_DOSYASI,
            use_gui=False,
            num_seconds=SIMULASYON_SURESI,
            min_green=MIN_YESIL,
            delta_time=KARAR_SURESI,
            reward_fn='pressure', 
        )
        
        # 2. Bizim yazdığımız özel Adapter ile sar
        # Bu sınıf Gym ortamı gibi davranır ama arkada SUMO'yu yönetir
        env = PettingZooToGymAdapter(env)
        
        # 3. Monitor ekle
        env = Monitor(env)
        return env
    return _init

def main():
    print(f"🚀 GARANTİ MULTI-AGENT EĞİTİM BAŞLIYOR ({ISLEM_SAYISI} ÇEKİRDEK)...")
    
    # Çoklu İşlemci Ortamı
    env = SubprocVecEnv([make_env(i) for i in range(ISLEM_SAYISI)])

    # Model
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,
        batch_size=512,
        n_steps=1024
    )

    # Eğitim
    EGITIM_ADIM = 200000 
    print(f"Hedef: {EGITIM_ADIM} adım. Bekleyin...")
    
    model.learn(total_timesteps=EGITIM_ADIM)

    model.save(MODEL_ADI)
    print(f"\n✅ Eğitim Tamamlandı! '{MODEL_ADI}.zip'")
    env.close()

if __name__ == "__main__":
    main()