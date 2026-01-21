import gymnasium as gym
import numpy as np
from gymnasium import spaces
import traci
import sumolib
from sumolib.net import Net
import xml.etree.ElementTree as ET
import random
import os
import sys

class SUMOTrafikOrtami(gym.Env):
    def __init__(self, net_dosyasi, use_gui=False):
        super().__init__()

        # Ayarlar
        self.net_dosyasi = net_dosyasi # .net uzaltılı dosyanın yolu
        self.maks_kuyruk = 50
        self.sim_step_per_action = 5  # Her aksiyonda ilerlenecek adım
        self.max_steps = 3600  # Bir bölüm kaç adım sürecek
        self.step_counter = 0
        self.kavsak_id = "kavsak_id" # SUMO'daki Junction ID buraya yazılmalı
        #self.min_yesil_suresi = 15  # Bir ışık en az 15 saniye yanmak ZORUNDA

        # 2. GUI SEÇİMİ (Bilgisayarındaki tam yolu kullanıyoruz ki kesin açılsın)
        if use_gui:
            #Eğitim ve test kımında değiştirmek için
            self.sumo_binary = "sumo-gui"
            self.extra_params = ["--start", "false", "--quit-on-end", "true"]
            # Rota dosyalarının listesi
            self.route_files = [
                r"SUMO\map_solo\test.rou.xml"
            ]
        else:
            self.sumo_binary = "sumo"
            self.extra_params = []
            # Rota dosyalarının listesi
            self.route_files = [
                "SUMO\map_solo\low.rou.xml",
                "SUMO\map_solo\medium.rou.xml",
                "SUMO\map_solo\high.rou.xml"
            ]

        # Haritdaki tüm kavşakları bulma

        self._ag_analiz_yap()

        self.son_degisim_zamani = {}
        for tls in self.tls_verileri:
            self.son_degisim_zamani[tls['id']] = 0

        # Action space tanımı
        # Verilen örnek haritada kaç tane fazı vars ona göre tanımlar
        # Modelde buna göre çıktı vericek

        print("Şerit sayılarını netleştirmek için simülasyon taranıyor...")

        aksiyon_boyutlari = [tls["num_actions"] for tls in self.tls_verileri]
        self.action_space = spaces.MultiDiscrete(aksiyon_boyutlari)

        try:
            traci_port = sumolib.miscutils.getFreeSocketPort()

            gecici_route = self.route_files[0]

            # GUI olmadan ('sumo') hızlıca başlat
            sumo_cmd = ["sumo", "-n", self.net_dosyasi, "-r", gecici_route, "--no-step-log", "true"]
            traci.start(sumo_cmd, port= traci_port)
            
            # Traci'ye şeritleri sor ve kaydet
            for tls in self.tls_verileri:
                controlled = traci.trafficlight.getControlledLanes(tls['id'])
                tls['lanes'] = sorted(list(set(controlled)))
                
            traci.close() # İşimizi bitirip kapatıyoruz
            print("✅ Şerit taraması tamamlandı.")
            
        except Exception as e:
            print(f"BAŞLANGIÇ HATASI: Simülasyon taranırken sorun oldu: {e}")
            try: traci.close()
            except: pass

        # Observation space
        # Tüm kavşakşarın şerit sayısı toplamı
        toplam_serit_sayisi = sum([len(tls["lanes"]) for tls in self.tls_verileri])
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(toplam_serit_sayisi,),
            dtype=np.float32
        )

        print(f"Toplam tespit edilen kavşak: {len(self.tls_verileri)}")
        print(f"Telpit edilen şerit sayisi{toplam_serit_sayisi}")

    def _ag_analiz_yap(self):
        print("Harita taraniyor...")

        self.tls_verileri = []

        # Bendeki .net uzantılı dosyayı okurken sorun çıktığı için direkt koddan okuma
        try:
            tree = ET.parse(self.net_dosyasi)
            root = tree.getroot()
        except Exception as e:
            raise Exception(f"Dosya okunamadı: {e}")
        
        # Ggerekli kısımları bulma
        tl_logics = root.findall("tlLogic")
        print(f"XML içinde bulunan ışık programı sayısı: {len(tl_logics)}")

        for tl in tl_logics:
            tls_id = tl.get("id")
            
            # Sadece 'static' veya 'actuated' olanları al (Gereksizleri ele)
            tl_type = tl.get("type", "static") 
            
            # Fazları (phase) bul
            phases = tl.findall("phase")
            
            yesil_fazlar = []
            
            # Fazları analiz et
            for i, phase in enumerate(phases):
                state = phase.get("state")
                try:
                    duration = float(phase.get("duration"))
                except:
                    duration = 0
                
                # Kural: İçinde 'G' veya 'g' (Yeşil) harfi varsa VE süresi 5sn'den uzunsa
                if ('G' in state or 'g' in state) and duration >= 5:
                    yesil_fazlar.append(i)
            
            # Sonuçları yazdır (Kontrol için)
            print(f"  > Kavşak: {tls_id} | Tip: {tl_type} | Toplam Yeşil Faz: {len(yesil_fazlar)}")
            
            # Eğer yönetilebilir bir ışıksa (en az 2 seçeneği varsa) listeye ekle
            if len(yesil_fazlar) > 1:
                self.tls_verileri.append({
                    'id': tls_id,
                    'green_phases': yesil_fazlar,     
                    'num_actions': len(yesil_fazlar), 
                    'lanes': [], # Reset fonksiyonunda traci ile dolacak
                    'last_action_idx': 0
                })

        # HATA KONTROLÜ
        if len(self.tls_verileri) == 0:
            print("\n!!! HATA !!!: XML okundu ama uygun 'Yeşil Faz' bulunamadı.")
            print("Lütfen NetEdit'te faz sürelerinin 5 saniyeden uzun olduğundan emin olun.")
        

    def reset(self, seed=None):

        # 1. Rastgele bir senaryo seç
        selected_route = random.choice(self.route_files)
        print(f"--- Yeni Eğitim Başlıyor: Seçilen Senaryo {selected_route} ---")

        # Eğer sumo açıksa kapatıcak
        try:
            traci.close()
        except:
            pass

        self.sumo_cmd = [
            self.sumo_binary,              # Görsel arayüz
            "-n", self.net_dosyasi,  # Sınıfa gönderdiğin .net.xml dosyası
            "-r", selected_route,# Sınıfa gönderdiğin .rou.xml dosyası
            "--no-step-log", "true", 
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1" # Sıkışan araçlar ışınlanmasın (gerçekçi olsun)
        ]

        # GUI ise start komutlarını ekle
        self.sumo_cmd.extend(self.extra_params)

        # Simülasyonu başa sar
        traci.start(self.sumo_cmd)

        self.sim_step = 0

        # Simülasyon başladıktan sonra hata olmaması için şeritlerin tam listesini çekiyoruz

        for tls in self.tls_verileri:
            # Bu ışığın yönettiği tüm şeritleri al
            controlled = traci.trafficlight.getControlledLanes(tls["id"])

            # Tekrarları temizle ve sırala
            controlled = sorted(list(set(controlled)))
            tls["lanes"] = controlled
            tls["last_action_idx"] = 0 # Sıfırla

        return self._get_observation(), {}

    def _get_observation(self):
        
        # Tüm kavşakların verilerini tek bir uzun dizide bilrleştiriyoruz
        tum_gozlem = []
        
        for tls in self.tls_verileri:
            for lane in tls["lanes"]:
                try:
                    kuyruk = traci.lane.getLastStepHaltingNumber(lane)
                    tum_gozlem.append(min(1.0, kuyruk / self.maks_kuyruk))
                except:
                    tum_gozlem.append(0.0)
        return np.array(tum_gozlem, dtype= np.float32)
        
    def _get_reward(self):
        # Bekleyen araç sayısına göre ceza
        return -1* sum(self._get_observation())

    def step(self, action):
        toplam_ceza = 0
        toplam_cikis_odulu = 0 # Outflow (Havuç)
        
        # 1. HER KAVŞAK İÇİN AKSİYONU UYGULA
        for i, tls in enumerate(self.tls_verileri):
            kavsak_id = tls['id']
            
            # Ajanın seçtiği hedef yeşil faz (Örn: 0 veya 2)
            istenen_aksiyon_idx = action[i] 
            hedef_faz = tls['green_phases'][istenen_aksiyon_idx]
            
            # Mevcut fazı öğren
            suanki_faz_index = traci.trafficlight.getPhase(kavsak_id)
            
            # --- KRİTİK DEĞİŞİKLİK: ZAMAN KİLİDİ YOK, SARI IŞIK VAR ---
            
            # Eğer ajan farklı bir faz seçtiyse (Değişim istiyorsa)
            if suanki_faz_index != hedef_faz:
                # A. Önce Sarı Işığı Yak
                # SUMO'da genelde mevcut yeşilin bir faz sonrası sarıdır.
                # 1. Toplam faz sayısını öğren (Dinamik olması için)
                # Bu kavşak için tüm mantığı çeker
                logic = traci.trafficlight.getAllProgramLogics(kavsak_id)[0]
                toplam_faz_sayisi = len(logic.phases)

                # 2. Modulo (%) kullanarak bir sonraki fazı hesapla
                # Eğer faz 3 ise ve toplam 4 ise: (3+1) % 4 = 0 olur (Başa döner)
                sari_faz = (suanki_faz_index + 1) % toplam_faz_sayisi

                traci.trafficlight.setPhase(kavsak_id, sari_faz)
                                
                # B. Sarı ışık süresi kadar simülasyonu ilerlet (Örn: 3 saniye)
                # Not: Bu döngüde de ceza hesaplamayı unutma!
                for _ in range(5): 
                    traci.simulationStep()
                    self.sim_step += 1
                    toplam_ceza += self._hesapla_anlik_ceza() # Yardımcı fonksiyon
                
                # C. Şimdi Hedef Yeşile Geç
                traci.trafficlight.setPhase(kavsak_id, hedef_faz)
                
                # Değişim olduğu için sayacı sıfırlayabilirsin (opsiyonel, istatistik için)
                self.son_degisim_zamani[kavsak_id] = traci.simulation.getTime()
                
            else:
                # Ajan "Aynı fazda kal" dediyse hiçbir şey yapma, yeşil yanmaya devam etsin.
                pass

        # 2. ANA SİMÜLASYON ADIMI (ACTION DURATION)
        # Ajan kararını verdi, şimdi sonucunu görmek için zamanı akıtıyoruz.
        # Örn: 5 saniye boyunca bu kararın sonuçlarını izle.
        sim_step_per_action = 5
        
        for _ in range(sim_step_per_action):
            traci.simulationStep()
            self.sim_step += 1
            toplam_ceza += self._hesapla_anlik_ceza()   

            toplam_cikis_odulu += self._hesapla_anlik_odul()

        # 3. GÖZLEM VE ÖDÜL
        obs = self._get_observation()
        
        # ÖDÜL FONKSİYONU REVİZYONU
        # WaitingTime yerine HaltingNumber kullanmak eğitimi hızlandırır.
        # Artık hem ödül hem ceza veriyor.
        reward = ((toplam_cikis_odulu * 0.5) - (toplam_ceza * 1.0)) / 100.0
        # Loglama
        '''
        if self.sim_step % 100 == 0:
            print(f"Adım: {self.sim_step}, Ödül: {reward:.2f}, Eylem: {action}")
        '''
        terminated = self.sim_step >= self.max_steps

        return obs, reward, terminated, False, {}

    # --- YARDIMCI FONKSİYON ---
    # Kod tekrarını önlemek için bunu sınıfına ekle
    def _hesapla_anlik_ceza(self):
        anlik_ceza = 0
        for tls in self.tls_verileri:
            for lane in tls["lanes"]:
                # Kuyruk uzunluğunu (duran araç sayısı) ceza olarak al
                anlik_ceza += traci.lane.getLastStepHaltingNumber(lane)
        return anlik_ceza
    
    def _hesapla_anlik_odul(self):
        anlik_hareketlilik = 0
        
        for tls in self.tls_verileri:
            for lane in tls["lanes"]:
                # 1. O şeritteki toplam araç sayısı (Duran + Giden)
                toplam_arac = traci.lane.getLastStepVehicleNumber(lane)
                # 2. O şeritte duran araç sayısı
                duran_arac = traci.lane.getLastStepHaltingNumber(lane)
                # 3. Hareket edenleri bul
                hareket_eden = toplam_arac - duran_arac
                # Eğer hareket eden varsa puana ekle
                if hareket_eden > 0:
                    anlik_hareketlilik += hareket_eden
                    
        return anlik_hareketlilik