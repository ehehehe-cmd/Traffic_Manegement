import sumolib
import os

# BURAYA EKRAN GÖRÜNTÜSÜNDEKİ DOSYA ADINI YAZ:
DOSYA_ADI = r"SUMO\map\grid_sehir.net.xml" 

if not os.path.exists(DOSYA_ADI):
    print(f"HATA: '{DOSYA_ADI}' dosyası bulunamadı! İsmi veya klasörü kontrol et.")
else:
    print(f"--- {DOSYA_ADI} Analizi Başlıyor ---")
    net = sumolib.net.readNet(DOSYA_ADI)
    lights = net.getTrafficLights()
    
    print(f"Haritada toplam {len(lights)} adet ışık bulundu.")
    
    for tls in lights:
        print(f"\nIşık ID: {tls.getID()}")
        programs = tls.getPrograms()
        print(f"  Program Sayısı: {len(programs)}")
        
        if not programs:
            print("  -> BU IŞIK BOŞ! (Program dictionary boş)")
        else:
            for prog_id, program in programs.items():
                print(f"  -> Program ID: '{prog_id}'")
                phases = program.getPhases()
                print(f"     Faz Sayısı: {len(phases)}")
                # Fazların sürelerini yazdıralım
                durations = [p.duration for p in phases]
                print(f"     Süreler: {durations}")