import os
import requests
import sys

print("🔧 Đang tải dữ liệu từ Google Drive...")

FILES = {
    "movies_final.csv": "1y6iKc5yL1Tx44iPY7i-syE8_3D8LKR8C",
    "ratings.csv": "1h66CNA3tPNTPilakGNT6NTVlPv98MTb4",
    "cosine_sim.pkl": "1Uf9GOG8_7QCrKaC_6UQ_OCrI2PUpb9Md",
    "session_state.pkl": "17G5ORDR5fYcm3Oo-eV_TqL-n-2WFx6vL",
    "model_content_based.pkl": "1-MNWTdSICuOBfJKhxJ5X-ZrYY5oAN7tk",
}

def download_if_needed():
    os.makedirs("data", exist_ok=True)
    
    for filename, file_id in FILES.items():
        filepath = f"data/{filename}"
        if not os.path.exists(filepath):
            print(f"📥 Đang tải {filename}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            
            response = requests.get(url, stream=True)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            size_mb = os.path.getsize(filepath) / (1024*1024)
            print(f"✅ Đã tải {filename} ({size_mb:.1f} MB)")
    
    print("🎯 Tải dữ liệu hoàn tất!")

if __name__ == "__main__":
    download_if_needed()