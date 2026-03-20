import os
import json
import shutil
import paramiko
import yaml
from ultralytics import YOLO, settings
from dotenv import load_dotenv
from ultralytics.utils.downloads import download

load_dotenv()

#yolo settings datasets_dir="/path-to-DAFYOLO/datasets"

# --- SSH Configuration ---
SERVER_IP = os.getenv("SERVER_IP")
SSH_PORT = 22
SSH_USER = os.getenv("USERNAME")
SSH_PASSWORD = os.getenv("PASSWORD")
SERVER_UPLOAD_DIR = "/datadrive/DAFYOLO/uploads" # (Path on the SSH server)
SERVER_DOWNLOAD_DIR = "/datadrive/DAFYOLO/global_model"

def setup_local_dataset(target_class_name):
    # 1. Ensure the base YOLO model exists
    model_name = "yolo26n.pt"
    if not os.path.exists(model_name):
        print(f"Downloading base {model_name}...")
        download(f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_name}")

    # 2. Trigger the download via the YOLO engine directly
    print("\nChecking/Downloading the Pascal VOC dataset (~2GB). This may take a few minutes...")
    # This safely triggers the download and format conversion if VOC.yaml is missing locally
    YOLO(model_name).train(data="VOC.yaml", epochs=0, imgsz=640)
    
    # 3. Locate the downloaded dataset
    datasets_root = settings['datasets_dir']
    base_dir = os.path.join(datasets_root, 'VOC')
    
    labels_dir = os.path.join(base_dir, 'labels', 'train') # sometimes it's train2012 or train2007
    images_dir = os.path.join(base_dir, 'images', 'train')
    
    # Fallback checks if the folder structure is slightly different
    if not os.path.exists(labels_dir):
        # Check if it split them into VOC2007 and VOC2012
        alt_labels_dir = os.path.join(base_dir, 'images', 'train2012')
        if os.path.exists(alt_labels_dir):
            labels_dir = os.path.join(base_dir, 'labels', 'train2012')
            images_dir = os.path.join(base_dir, 'images', 'train2012')
        else:
            # Check just the base images folder
            labels_dir = os.path.join(base_dir, 'labels')
            images_dir = os.path.join(base_dir, 'images')

    if not os.path.exists(labels_dir) or not os.path.exists(images_dir):
        raise FileNotFoundError(f"Could not find VOC dataset folders. Looked in: {base_dir}")

    # 4. Read VOC original class ID
    voc_classes = {
        0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle',
        5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow',
        10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
        15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'
    }
    
    original_id = None
    for k, v in voc_classes.items():
        if v.lower() == target_class_name.lower():
            original_id = k
            break
            
    if original_id is None:
        valid_classes = ", ".join(voc_classes.values())
        raise ValueError(f"Class '{target_class_name}' not found in Pascal VOC. Valid options: {valid_classes}")

    # 5. Setup local client directories
    client_id = f"client_{target_class_name}"
    out_dir = os.path.abspath(f"./{client_id}_data")
    out_img_dir = os.path.join(out_dir, "images", "train")
    out_lbl_dir = os.path.join(out_dir, "labels", "train")
    
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    
    print(f"\nFiltering VOC images strictly for: '{target_class_name}' (Original ID: {original_id})...")
    
    images_copied = 0
    # Process every label file
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'): continue
        
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            lines = f.readlines()
            
        filtered = []
        for line in lines:
            if not line.strip(): continue
            try:
                if int(line.split()[0]) == original_id:
                    parts = line.split()
                    parts[0] = "0" # Force ID to 0 locally
                    filtered.append(" ".join(parts) + "\n")
            except ValueError:
                pass
                
        if filtered:
            img_filename = label_file.replace('.txt', '.jpg')
            src_img = os.path.join(images_dir, img_filename)
            
            if os.path.exists(src_img):
                with open(os.path.join(out_lbl_dir, label_file), 'w') as f:
                    f.writelines(filtered)
                shutil.copy(src_img, os.path.join(out_img_dir, img_filename))
                images_copied += 1
            
    print(f"Extraction complete! Found and copied {images_copied} valid images containing '{target_class_name}'.")
    
    if images_copied == 0:
        raise ValueError(f"No images found. Ensure the VOC dataset downloaded correctly to {datasets_root}.")

    # 6. Create YAML
    yaml_path = f"{client_id}.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump({
            "path": out_dir, 
            "train": "images/train", 
            "val": "images/train", 
            "nc": 1, 
            "names": {0: target_class_name}
        }, f, sort_keys=False)
                   
    return yaml_path, client_id

def ssh_transfer(client_id, weights_path):
    """Transfers weights and metadata to the server via SFTP."""
    print(f"\nConnecting to server {SERVER_IP} via SSH...")
    transport = paramiko.Transport((SERVER_IP, SSH_PORT))
    transport.connect(username=SSH_USER, password=SSH_PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)
    
    # Upload weights
    remote_weights = f"{SERVER_UPLOAD_DIR}/{client_id}_weights.pt"
    sftp.put(weights_path, remote_weights)
    
    # Upload metadata
    meta = {"client_id": client_id, "class_name": client_id.split('_')[1]}
    with open("meta.json", "w") as f: json.dump(meta, f)
    sftp.put("meta.json", f"{SERVER_UPLOAD_DIR}/{client_id}_meta.json")
    
    sftp.close()
    transport.close()
    print("Transfer complete!")

if __name__ == "__main__":
    target_class = input("Enter the class to train on (e.g., 'person', 'car'): ").strip()
    yaml_path, client_id = setup_local_dataset(target_class)
    
    print(f"\nStarting YOLO26 Local Training for {client_id}...")
    model = YOLO("yolo26n.pt")
    
    # Using 3 epochs for a fast local simulation
    results = model.train(data=yaml_path, epochs=3, imgsz=640, project=client_id, name="train") 
    
    best_weights = os.path.abspath(f"{client_id}/train/weights/best.pt")
    
    if os.path.exists(best_weights):
        # We wrap SSH in a try/except so you can test local training even if SSH isn't set up yet
        try:
            ssh_transfer(client_id, best_weights)
        except Exception as e:
            print(f"SSH Transfer failed (Is your server running?): {e}")
            print(f"Weights are saved locally at: {best_weights}")
    else:
        print("Training failed to produce weights.")