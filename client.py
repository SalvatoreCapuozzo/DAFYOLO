import os
import json
import shutil
import paramiko
import yaml
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# --- SSH Configuration ---
SERVER_IP = os.getenv("SERVER_IP")
SSH_PORT = 22
SSH_USER = os.getenv("USERNAME")      # CHANGE THIS
SSH_PASSWORD = os.getenv("PASSWORD")  # CHANGE THIS
SERVER_UPLOAD_DIR = "/datadrive/DAFYOLO/uploads" # CHANGE THIS (Path on the SSH server)
SERVER_DOWNLOAD_DIR = "/datadrive/DAFYOLO/global_model" # CHANGE THIS

def setup_local_dataset(target_class_name):
    """Downloads COCO128 and filters it for the chosen class."""
    print("Checking/Downloading base dataset...")
    YOLO("yolo26n.pt").train(data="coco128.yaml", epochs=0, imgsz=640)
    
    # Read COCO128 yaml to find the original class ID
    with open('datasets/coco128.yaml', 'r') as f:
        coco_data = yaml.safe_load(f)
    
    names = coco_data.get('names', {})
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
        
    original_id = None
    for k, v in names.items():
        if v.lower() == target_class_name.lower():
            original_id = k
            break
            
    if original_id is None:
        raise ValueError(f"Class '{target_class_name}' not found in COCO128!")

    # Setup directories
    client_id = f"client_{target_class_name}"
    base_dir = os.path.abspath(os.path.join(os.getcwd(), 'datasets', 'coco128'))
    out_dir = os.path.abspath(f"./{client_id}_data")
    
    out_img_dir = os.path.join(out_dir, "images", "train")
    out_lbl_dir = os.path.join(out_dir, "labels", "train")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    
    labels_dir = os.path.join(base_dir, 'labels', 'train2017')
    images_dir = os.path.join(base_dir, 'images', 'train2017')
    
    print(f"Filtering dataset strictly for: {target_class_name}...")
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'): continue
        
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            lines = f.readlines()
            
        filtered = []
        for line in lines:
            if int(line.split()[0]) == original_id:
                parts = line.split()
                parts[0] = "0" # Force ID to 0 locally
                filtered.append(" ".join(parts) + "\n")
                
        if filtered:
            with open(os.path.join(out_lbl_dir, label_file), 'w') as f:
                f.writelines(filtered)
            shutil.copy(os.path.join(images_dir, label_file.replace('.txt', '.jpg')), 
                        os.path.join(out_img_dir, label_file.replace('.txt', '.jpg')))
                        
    yaml_path = f"{client_id}.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump({"path": out_dir, "train": "images/train", "val": "images/train", 
                   "nc": 1, "names": [target_class_name]}, f, sort_keys=False)
                   
    return yaml_path, client_id

def ssh_transfer(client_id, weights_path):
    """Transfers weights and metadata to the server via SFTP."""
    print("\nConnecting to server via SSH...")
    transport = paramiko.Transport((SERVER_IP, SSH_PORT))
    transport.connect(username=SSH_USER, password=SSH_PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)
    
    # Upload weights
    remote_weights = f"{SERVER_UPLOAD_DIR}/{client_id}_weights.pt"
    sftp.put(weights_path, remote_weights)
    
    # Upload metadata (so server knows which class this is)
    meta = {"client_id": client_id, "class_name": client_id.split('_')[1]}
    with open("meta.json", "w") as f: json.dump(meta, f)
    sftp.put("meta.json", f"{SERVER_UPLOAD_DIR}/{client_id}_meta.json")
    
    sftp.close()
    transport.close()
    print("Transfer complete!")

if __name__ == "__main__":
    target_class = input("Enter the class to train on (e.g., 'person', 'car', 'dog'): ").strip()
    yaml_path, client_id = setup_local_dataset(target_class)
    
    print("\nStarting YOLO26 Local Training...")
    model = YOLO("yolo26n.pt")
    # Using 3 epochs for a fast local simulation
    results = model.train(data=yaml_path, epochs=3, imgsz=640, project=client_id) 
    
    best_weights = f"{client_id}/train/weights/best.pt"
    ssh_transfer(client_id, best_weights)