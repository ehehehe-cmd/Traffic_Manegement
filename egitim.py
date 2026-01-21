import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import torch
#import intel_extension_for_pytorch as ipex

from adaptor import SUMOTrafikOrtami

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
ANA_KAYIT_YERİ = os.path.join(BASE_PATH, "logs")

NET_DOSYASI = os.path.join(BASE_PATH, "SUMO/map_solo/solo.net.xml")  # Kendi dosya yolun
ROUTE_DOSYASI = os.path.join(BASE_PATH, "SUMO/map_solo/traffic.rou.xml") # Kendi dosya yolun
KAYIT_KLASORU = os.path.join(ANA_KAYIT_YERİ, "modeller")
LOG_KLASORU = os.path.join(ANA_KAYIT_YERİ, "logs")
model_adi= "ppo_kavsak_model_solov6"
CPU_SAYISI = 10 # Bilgisayarının çekirdek sayısına göre ayarla (Örn: 4, 8, 12)

# Klasörleri oluştur (Yoksa yaratır, varsa dokunmaz)
os.makedirs(KAYIT_KLASORU, exist_ok=True)
os.makedirs(LOG_KLASORU, exist_ok=True)

def egitim_baslat():
    print("------Eğitim Başlıyor------")

    # GPU Kullanımı Kontrolü (Opsiyonel Bilgi)
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"Eğitim Cihazı: {device}")

    env_kwargs = {
        "net_dosyasi": NET_DOSYASI, 
        "use_gui":False
    }

    env = make_vec_env(
        SUMOTrafikOrtami, 
        n_envs=CPU_SAYISI, 
        seed=0, 
        vec_env_cls=SubprocVecEnv, 
        env_kwargs=env_kwargs
    )

    env = VecMonitor(env, LOG_KLASORU)

    # Modeli oluşturma
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate= 0.0003,
        n_steps= 1024,
        batch_size=64,
        gamma=0.99,
        tensorboard_log=LOG_KLASORU,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[512, 512])
    )

    # Her 10.000 adımda bir kaydeder
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=KAYIT_KLASORU,
        name_prefix= model_adi + "_test"
    )

    # Eğitimi başlat
    print("----Model Eğitimi Başlıyor----")
    model.learn(total_timesteps=1000000, callback=checkpoint_callback)

    # Modeli kaydet
    model.save(model_adi + "_final")

if __name__ == "__main__":
    egitim_baslat()
