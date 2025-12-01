import os
import sys
import sumolib # SUMO'nun harita okuma kütüphanesi

# --- AYARLAR ---
HARITA_DOSYASI = "SUMO\map\grid_sehir.net.xml"  # Kullandığın harita
CIKTI_DOSYASI = "traffic_dengesiz.rou.xml"    # Oluşacak yeni trafik dosyası

# TRAFİK YOĞUNLUKLARI
KUZEY_GUNEY_SIKLIGI = 2.0   # Çok Yoğun
DOGU_BATI_SIKLIGI = 25.0    # Çok Seyrek

def main():
    print(f"Harita taranıyor: {HARITA_DOSYASI}...")
    
    try:
        net = sumolib.net.readNet(HARITA_DOSYASI)
    except Exception as e:
        print(f"HATA: Harita bulunamadı! {e}")
        return

    # 1. HARİTANIN SINIRLARINI BUL (Bounding Box)
    # xMin, yMin, xMax, yMax değerlerini alır.
    bbox = net.getBBoxXY()
    x_min, y_min = bbox[0]
    x_max, y_max = bbox[1]
    
    print(f"Harita Sınırları: X({x_min}-{x_max}), Y({y_min}-{y_max})")

    with open(CIKTI_DOSYASI, "w") as f:
        f.write('<routes>\n')
        f.write('    <vType id="car" accel="1.0" decel="4.5" length="5.0" minGap="2.5" maxSpeed="50" sigma="0.5" />\n')

        count = 0
        
        for edge in net.getEdges():
            # İç yolları atla
            if edge.getFunction() != "": continue

            # Yolun başlangıç noktasının koordinatını al
            from_node = edge.getFromNode()
            x, y = from_node.getCoord()

            # --- YENİ MANTIK: SINIR KONTROLÜ ---
            # Eğer noktanın X veya Y koordinatı haritanın sınırına eşitse (veya çok yakınsa),
            # orası bir giriş kapısıdır.
            is_fringe = False
            
            # 1 metre hata payı bırakıyoruz (float hassasiyeti için)
            if abs(x - x_min) < 1.0 or abs(x - x_max) < 1.0 or \
               abs(y - y_min) < 1.0 or abs(y - y_max) < 1.0:
                is_fringe = True
            
            # Sadece sınırdaki yollardan araç başlat
            if not is_fringe:
                continue
            # -----------------------------------

            # Yönü ve Yoğunluğu Belirle
            from_coord = edge.getFromNode().getCoord()
            to_coord = edge.getToNode().getCoord()
            dx = abs(from_coord[0] - to_coord[0])
            dy = abs(from_coord[1] - to_coord[1])
            
            if dy > dx: # DİKEY (Kuzey-Güney)
                period = KUZEY_GUNEY_SIKLIGI
                prefix = "NS"
            else:       # YATAY (Doğu-Batı)
                period = DOGU_BATI_SIKLIGI
                prefix = "EW"
            
            f.write(f'    <flow id="{prefix}_{count}" type="car" begin="0" end="3600" period="{period}" from="{edge.getID()}"/>\n')
            count += 1

        f.write('</routes>\n')
    
    if count == 0:
        print("❌ HATA: Hiçbir yol bulunamadı! Harita sınırlarını kontrol et.")
    else:
        print(f"✅ Başarılı! Toplam {count} giriş noktasından trafik akışı oluşturuldu.")
        print(f"Dosya: {CIKTI_DOSYASI}")

if __name__ == "__main__":
    main()