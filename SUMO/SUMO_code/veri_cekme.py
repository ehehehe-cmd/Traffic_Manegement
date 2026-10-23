import os
import sys
import traceback
import xml.etree.ElementTree as ET
import random
import sumolib
import time


if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = '/usr/share/sumo'
    

USE_GUI = True

pending = {} # tls_id -> {"state": hedef ışık string'i, "remaining": kalan adım}

if USE_GUI:
    import traci
else:
    import libsumo as traci

def load_net(cfg_path):
    """
    .sumocfg'den net-file yolunu bulup sumolib ile okur 
    """

    root = ET.parse(cfg_path).getroot()
    net_file = root.find(".//net-file").get("value")
    net_path = os.path.join(os.path.dirname(cfg_path), net_file) 

    return sumolib.net.readNet(net_path, withPrograms=True)

def set_agent_action(tls_id, action_id, action_spaces, hold_steps=None):
    """
    RL ajanının seçtiği aksionları SUMO'daki karşılığna uyarlar
    """
    act = next((a for a in action_spaces[tls_id]["available_actions"]
                if a["action_id"] == action_id), None)
    if act is None:
        print(f"Hata {tls_id} için geçersiz aksiyon ID: {action_id}")
        return
    
    if tls_id in pending:
        return

    cur_satate = traci.trafficlight.getRedYellowGreenState(tls_id)
    target_state = act["state"]

    #Zaten aynısı seçildiyse süreyi uzat
    if cur_satate == target_state:
        return

    yellow_state = "".join(
        "y" if c in "Gg" and t == "r" else c
        for c, t in zip(cur_satate, target_state)
    )
    traci.trafficlight.setRedYellowGreenState(tls_id, yellow_state)
    pending[tls_id] = {"state": target_state, 
                       "remaining": YELLOW_TIME}


def update_pending_transitions():
    """
    Her sim_step'ten sonra çağır sarı süresi bitenleri yeşile çevirir
    """

    for tls_id in list(pending):
        p = pending[tls_id]
        p["remaining"] -= 1
        if p["remaining"] <= 0:
            traci.trafficlight.setRedYellowGreenState(tls_id, p["state"])
            del pending[tls_id]

    
def get_agent_state(tls_id, messages = None):
    """
    RL ajanı için gerekli verileri çeker(araç sayısı, hız, kuyruk vb.)
    """
    #TODO - İLERDE BURAYI MODELE GÖRE DÜZENLE
    #Girilen kavşaktaki geçerli şeritlerin ID'lerini al
    incoming_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))

    lane_data = {}
    total_vehicels = 0
    total_halting_vehicles = 0 #Kuyrukta bekleyen (duran) araç sayısı ASIL METRİK

    

    for lane in incoming_lanes:
        #Şeritteki total araç sayısı
        veh_count = traci.lane.getLastStepVehicleNumber(lane)

        #Şeritteki suran araç sayısı
        halting_count = traci.lane.getLastStepHaltingNumber(lane)

        #Şeritteki araçların ort hızı
        mean_speed = traci.lane.getLastStepMeanSpeed(lane)

        total_vehicels += veh_count
        total_halting_vehicles += halting_count

        lane_data[lane] = {
            "vehicle_count": veh_count,
            "halting_count": halting_count,
            "mean_speed": mean_speed
        }

    return {
        "total_vehicles": total_vehicels,
        "total_halting_vehicles": total_halting_vehicles,
        "lane_details": lane_data,
        "neighbor_messages": messages or {},
    }

def build_messages(states, neighbors):
    """Her ajanın komşularından alacağı mesajı, önceden çekilmiş state'lerden türetir"""
    return {
        t: {f"from_{n}": states[n]["total_halting_vehicles"] for n in neighbors[t]}
        for t in neighbors
    }

def get_tls_neighbors(tls_ids, net):
    """
    Kavşaklar arası komşuluk ilişkilerini otomatik çıkartıcak
    Hangi kavşağın hangisiyle bağlantılı olduğunu tespit eder
    """
    tls_set = set(tls_ids)

    neighbors = {t: set() for t in tls_ids}

    #Her kavşağın kontrolündeki yolları incele
    for t in tls_ids:
        for in_lane, out_lane, _ in net.getTLS(t).getConnections():
            up = in_lane.getEdge().getFromNode().getID()    # yukarı akış kavşağı
            down = out_lane.getEdge().getToNode().getID()   # aşağı akış kavşağı
            for n in (up, down):
                if n in tls_set and n != t:
                    neighbors[t].add(n)
                    neighbors[n].add(t)   # karşılıklı olsun

    return {t: sorted(v) for t, v in neighbors.items()}

def exchange_messages(tls_ids, neighbors):
    """
    Her kavşağın komşulara göndericeği mesajları hesaplar
    Komşu kavşakların uzayıne eklemek üzere hazılar.
    """

    agent_messages= {}

    #Her ajan gönderilecek verileri özetler
    for tls_id in tls_ids:
        incoming_lanes =list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
        total_halting = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in incoming_lanes)

        #Komşulara gidecek mesaj
        agent_messages[tls_id] = {
            "congestion_score": total_halting, #Yoğunluk katsayısı
        }
    
    incoming_messages_for_agents = {tls_id:{} for tls_id in tls_ids}

    for tls_id, comm_neighbors in neighbors.items():
        for neighbor_id in comm_neighbors:
            #Komşunun gönderdiği yoğunluğu ajanın gelen kutusuna yaz
            incoming_messages_for_agents[tls_id][f"from_{neighbor_id}"] = agent_messages[neighbor_id]["congestion_score"]

    return incoming_messages_for_agents

def get_tls_action_spaces (sumo_cfg_path):
    """
    SUMO cfg dosyasını açar ardından yeşil faz sayılarını alır ve bunları aksiyom uzayı sayısı olarak çeker
    Bunu yaparken her kavşak için ayrı ayrı yapar. böylece her kavşaya ayrı ajan koyabiliriz.
    """

    #Yapılabilecek aksiyonları almak için SUMO dosyasını açıyoruz.
    sumo_cmd = ["sumo", "-c", sumo_cfg_path, "--no-step-log", "true", "--no-warnings", "true"]


    #SUMO'yu başlatır
    traci.start(sumo_cmd)

    #Kavşakların id'sini çeker
    tls_ids = traci.trafficlight.getIDList()
    print(f"Haritada toplamda: {len(tls_ids)}\n")

    agent_action_spaces = {}

    for tls_id in tls_ids:
        #id id her kavşağın yön bilgilerini çeker
        #Burda gelecek liste SUMO'daki karşılığı GGrrGr gibi karekterlerdir
        #Bu gelen karekterler hangi şeritten hangi şeride geçiş olduğunu söyler
        controlled_links = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
        num_signals = len(controlled_links)

        #Mevcut kavşağın sinyal programını ve fazlarını çek
        logics = traci.trafficlight.getAllProgramLogics(tls_id)
        current_logic = logics[0] #İlk varsayılan fazları al
        all_phases = current_logic.phases

        #Sadece yeşil ışık olan fazları filtreliycez. Bunlar ana fazlar
        #Ajana sarı ışık yak komutu vermiyorum. sarı ışığı faz geçişlerinde zorunl olarak uygulıycak
        green_actions = []

        for phases_idx, phase in enumerate(all_phases):
            #"G" öncelikli yeşil "g" ise normal yeşil olan fazlar var
            #Ayrıca "y" ve "Y" olan sarı olmayan fazlar var

            if ("G" in phase.state or "g" in phase.state) and ("y" not in phase.state.lower()):
                green_actions.append({
                    "action_id": len(green_actions), #Ajanın seçebileceği aksiyon (0, 1, 2...)
                    "sumo_phase_index": phases_idx, #SUMO'da karşılık gelen gerçek faz indeksi
                    "state": phase.state, #Işıkların durumu Örn: GGrrGGrr gibi
                    "default_duration": phase.duration
                })

        #Bilgileri ilgili kavşak ajanı için kaydet
        agent_action_spaces[tls_id] = {
            "num_controlled_signals": num_signals,
            "total_sumo_phases": len(all_phases),
            "action_space_size": len(green_actions), #Disreate(N) için N değeri
            "available_actions": green_actions,
            "controlled_links": controlled_links
        }

        #Konsola log yazdırma
        print(f"=== Ajan (Kavşak) ID: {tls_id} ===")
        print(f"Kontrol Edilen Işık/Yön Sayısı (State Uzunluğu): {num_signals}")
        print(f"Toplam SUMO Fazı (Sarılar dahil): {len(all_phases)}")
        print(f"RL Ajanı Aksiyon Uzayı Boyutu (Sadece Yeşil Fazlar): {len(green_actions)}")
        print("Seçilebilir Aksiyonlar (Fazlar):")
        for act in green_actions:
            print(f"  -> Aksiyon {act['action_id']} (SUMO Faz İndeksi: {act['sumo_phase_index']}): Durum = '{act['state']}'")
        print("-" * 50)

        
    #SUMO'yu kapatma
    traci.close()   
    return agent_action_spaces


if __name__ == "__main__":
    #NOTE - Zaman ölçümü başlangıç
    baslangic_zamani = time.perf_counter()

    #SUMO'nun .sumocfg uzantısı buraya yazılmalı
    CONFIG_PATH = "SUMO/SUMO_map/test.sumocfg"

    DESICION_INTERVAL = 20
    HOLD = 20

    YELLOW_TIME = 3 #Işığın kaç saniye sarı yanıcağı
    


    #TODO -  MAX_NEIGHBOR = max(len(v) for v in neighbors.values())

    try:
        action_spaces = get_tls_action_spaces(CONFIG_PATH)
        tls_ids = list(action_spaces.keys())
        net = load_net(CONFIG_PATH)


        #TEST AMAÇLI OLAN KISIM sumo-gui kullanabilirsin ui için
        sumo_cmd = ["sumo", "-c", CONFIG_PATH, "--no-step-log", "true"]
        traci.start(sumo_cmd)

        #Komşulukları belirleme
        neighbors = get_tls_neighbors(tls_ids, net)
        print(f"Komşuluk matrisi {neighbors}\n")
        
        #Simülasyonu X adım çalıştırıp test et
        for step in range(3600):
            traci.simulationStep()

            #NOTE - Önce  bunu çağırman lazım
            update_pending_transitions()
            
            #NOTE - TEST
            states = {t: get_agent_state(t) for t in tls_ids}
            messages = build_messages(states, neighbors)
            for t in tls_ids:
                states[t]["neighbor_messages"] = messages[t]
            
            print(f"******Adım: {step}******")
            #print(states)
            #NOTE - TEST SON

            """
            #Komşular arası mesajlaşmayı gerçekleştir
            messages = exchange_messages(tls_ids, neighbors)
            #print(messages)
            #Her ajan için [kendi + komşu verileri] birleştir
            for tls_id in tls_ids:
                #Kendi verileri örn toğlam kuyruk
                my_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
                my_queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in my_lanes)

                #Komşudan gelen mesaj 
                neighbor_msgs = messages[tls_id]

                #print(f"Adım {step} | Ajan {tls_id} -> Kendi Kuyruğum: {my_queue} | Komşudan Gelen Mesaj: {neighbor_msgs}")
                

            for tls_id in action_spaces.keys():
                #Ajan gözlemlerini çek
                state = get_agent_state(tls_id, messages[tls_id])
                #print(f"Adım {step} | Kavşak {tls_id} -> Araç Sayısı: {state['total_vehicles']}, Kuyruk: {state['total_halting_vehicles']}")

                # Örnek: Ajan rastgele veya sırayla bir aksiyon seçsin (Örn: Aksiyon 1)
                # set_agent_action(tls_id, action_id=1, action_spaces=action_spaces)
            """
            if step % DESICION_INTERVAL == 0:
                for tls_id, space in action_spaces.items():
                    n = space["action_space_size"]
                    action_id = random.randint(0, n -   1)#(step // DESICION_INTERVAL) % n #Sırayla dön
                    set_agent_action(tls_id, action_id, action_spaces, HOLD)

                    cur = traci.trafficlight.getPhase(tls_id)
                    state = traci.trafficlight.getRedYellowGreenState(tls_id)
                    print(f"Adım {step} | {tls_id} | aksiyon={action_id} " f"| faz={cur} | ışık={state}")

    

    except Exception as e:
        print(f"\nKod çalışırken bir hata oluştu: {e}")
        traceback.print_exc()

    finally:
        traci.close()
        #NOTE - Zaman ölçümü bitiş True False 120step :farkı 2.13541 -> 0.02750 77x, 3600step: fark 5.24526 -> 1.20300 4x fark
        bitis_zamani = time.perf_counter()

        calisma_suresi = bitis_zamani - baslangic_zamani
    print(f"Kodun çalışma süresi: {calisma_suresi:.5f} saniye")
        
