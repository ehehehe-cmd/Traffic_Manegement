import pyautogui
import time

print("Koordinat bulucu çalışıyor... (Çıkmak için Ctrl+C)")
print("Mouse'u istediğin yolun sol üst köşesine ve sağ alt köşesine götür.")

try:
    while True:
        x, y = pyautogui.position()
        print(f"X: {x}, Y: {y}", end="\r") # Sürekli aynı satıra yazar
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nÇıkış yapıldı.")
    