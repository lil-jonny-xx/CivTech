import os
import urllib.request

# Folder where models will be saved
models_dir = os.path.join(os.getcwd(), "models")
os.makedirs(models_dir, exist_ok=True)

# Model URLs
models = {
    "yolov8l.pt": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt",
    "yolov8l-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-seg.pt",
}

for name, url in models.items():
    save_path = os.path.join(models_dir, name)
    if not os.path.exists(save_path):
        print(f"⬇️  Downloading {name}...")
        urllib.request.urlretrieve(url, save_path)
        print(f"✅ Saved to {save_path}")
    else:
        print(f"🟢 {name} already exists at {save_path}")
