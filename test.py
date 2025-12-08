import numpy as np
from adaptor import SUMOTrafikOrtami  # Sınıfını yazdığın dosyayı import et

# Dosya yollarını tanımla
NET_DOSYASI = r"SUMO\map\grid_sehir.net.xml"  # Kendi dosya yolun
ROUTE_DOSYASI = r"SUMO\map\traffic.rou.xml" # Kendi dosya yolun

def manuel_test():
    print("--- MANUEL TEST BAŞLIYOR ---")
    
    # 1. Ortamı Oluştur
    try:
        env = SUMOTrafikOrtami(NET_DOSYASI, ROUTE_DOSYASI)
        print("✅ Ortam başarıyla oluşturuldu.")
    except Exception as e:
        print(f"❌ Ortam oluşturulurken hata: {e}")
        return

    # 2. Reset Testi
    try:
        obs, info = env.reset()
        print("✅ Reset başarılı.")
        print(f"   Gözlem (Obs) Boyutu: {obs.shape}")
        print(f"   Gözlem Space Beklentisi: {env.observation_space.shape}")
        
        # Boyut Kontrolü
        assert obs.shape == env.observation_space.shape, "HATA: Gelen veri boyutu ile tanımlanan space uyuşmuyor!"
        
    except Exception as e:
        print(f"❌ Reset sırasında hata: {e}")
        return

    # 3. Adım (Step) Testi
    print("\n--- 10 Adımlık Simülasyon Testi ---")
    for i in range(10):
        # Rastgele bir aksiyon üret (Space yapısına uygun)
        action = env.action_space.sample()
        
        # Adım at
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Adım {i+1}:")
        print(f"   Aksiyon: {action}") # Örn: [0, 1, 0, 1] görmelisin
        print(f"   Ödül: {reward}")     # Negatif bir sayı görmelisin (örn: -12.5)
        print(f"   Obs Tipi: {type(obs)}") # <class 'numpy.ndarray'> olmalı
        
        # Veri tipi kontrolü (PPO float32 sever)
        if obs.dtype != np.float32:
            print("   UYARI: Obs verisi float32 değil! PPO hata verebilir.")

    print("\n✅ Test başarıyla tamamlandı. Kodun çalışıyor!")
    
    # Kapat
    env.close()

if __name__ == "__main__":
    manuel_test()