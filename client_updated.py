import os
import json
import shutil
import paramiko
import yaml
import glob
from datetime import datetime
from ultralytics import YOLO, settings
from dotenv import load_dotenv
from ultralytics.utils.downloads import download
import torch
from ultralytics.models.yolo.detect import DetectionTrainer

load_dotenv()

SSH_PORT = 22
# --- Change this section at the top of client_updated.py ---
SERVER_IP = os.getenv("SERVER_IP", "").strip().replace('"', '').replace("'", "")
SSH_USER = os.getenv("USERNAME", "").strip().replace('"', '').replace("'", "")
SSH_PASSWORD = os.getenv("PASSWORD", "").strip().replace('"', '').replace("'", "")

SERVER_UPLOAD_DIR = "/datadrive/DAFYOLO/uploads"
SERVER_DOWNLOAD_DIR = "/datadrive/DAFYOLO/global_model"

LOCAL_MODELS_DIR = "runs/detect"
DOWNLOADED_MODELS_DIR = "global_models"
os.makedirs(DOWNLOADED_MODELS_DIR, exist_ok=True)

def _ssh_connect():
    """Robust SSH Connection helper to fix Authentication issues"""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=SERVER_IP, 
        port=SSH_PORT, 
        username=SSH_USER, 
        password=SSH_PASSWORD,
        look_for_keys=False, # Ignore background SSH keys
        allow_agent=False,   # Ignore SSH agents
        timeout=15
    )
    return ssh

# ==============================================================================
# UNIFIED SMART TRAINER
# ==============================================================================

class SmartFLTrainer(DetectionTrainer):
    """
    Dynamically freezes layers based on the server's requested strategy.
    Configured via class attributes before instantiation.
    """
    strategy = "fedhead"
    is_round_1 = True

    def __init__(self, overrides=None, _callbacks=None):
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        self.add_callback("on_train_start", self._apply_strategy_freezing)

    def _apply_strategy_freezing(self, trainer):
        detect_idx = len(trainer.model.model) - 1
        detect_prefix = f'model.{detect_idx}.'
        frozen_count, unfrozen_count = 0, 0

        for name, param in trainer.model.named_parameters():
            if not name.startswith(detect_prefix):
                # Backbone + Neck ALWAYS frozen
                param.requires_grad = False
                frozen_count += 1
            else:
                # Head behavior depends on strategy
                if SmartFLTrainer.strategy in ['ties', 'fedavg'] or SmartFLTrainer.is_round_1:
                    # Train full head (to generate task vector or establish round 1 features)
                    param.requires_grad = True
                    unfrozen_count += 1
                else:
                    # FedHead/Stitch Round 2+: Freeze intermediates, train final 1x1 only
                    is_final_layer = any(x in name for x in ['cv3.', 'one2one_cv3.']) and ('.2.weight' in name or '.2.bias' in name)
                    if is_final_layer:
                        param.requires_grad = True
                        unfrozen_count += 1
                    else:
                        param.requires_grad = False
                        frozen_count += 1

        print(f"\n🧠 [SmartTrainer] Server requested strategy: '{SmartFLTrainer.strategy.upper()}'")
        mode = "FULL HEAD" if (SmartFLTrainer.strategy in ['ties', 'fedavg'] or SmartFLTrainer.is_round_1) else "FINAL LAYER ONLY"
        print(f"🧠 [SmartTrainer] Mode activated: {mode}")
        print(f"🧠 [SmartTrainer] Frozen Tensors: {frozen_count} | Trainable: {unfrozen_count}\n")


# ==============================================================================
# CLIENT-SERVER HANDSHAKE & FILE MANAGEMENT
# ==============================================================================

def fetch_server_info():
    """Reads the server's broadcast to ensure the client configures itself correctly."""
    print(f"\n🔄 Connecting to {SERVER_IP} to handshake with server...")
    try:
        ssh = _ssh_connect()
        sftp = ssh.open_sftp()
        
        try:
            sftp.get(f"{SERVER_DOWNLOAD_DIR}/server_info.json", "local_server_info.json")
            with open("local_server_info.json", "r") as f:
                info = json.load(f)
            sftp.close()
            ssh.close()
            return info
        except FileNotFoundError:
            sftp.close()
            ssh.close()
            raise ValueError("Server is not running or hasn't initialized.")
    except Exception as e:
        print(f"❌ Handshake failed: {e}")
        return None

def download_global_model(strategy):
    """Downloads the global model uniquely versioned by timestamp."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_path = os.path.join(DOWNLOADED_MODELS_DIR, f"global_model_{strategy}_{ts}.pt")
    
    try:
        ssh = _ssh_connect()
        sftp = ssh.open_sftp()
        sftp.get(f"{SERVER_DOWNLOAD_DIR}/global_model.pt", local_path)
        sftp.close()
        ssh.close()
        
        if os.path.getsize(local_path) < 1000:
            os.remove(local_path)
            return None
        return local_path
    except Exception as e:
        print(f"❌ Failed to download global model: {e}")
        return None

def select_file_interactive(prompt_text, search_pattern):
    """Generic CLI menu to select a file from a glob pattern."""
    files = sorted(glob.glob(search_pattern), key=os.path.getmtime, reverse=True)
    if not files:
        print("❌ No files found matching your request.")
        return None
    
    print(f"\n{prompt_text}")
    for i, f in enumerate(files):
        print(f"  [{i+1}] {f}")
    print(f"  [0] Cancel")
    
    try:
        choice = int(input("Select number: ").strip())
        if choice == 0: return None
        return files[choice - 1]
    except (ValueError, IndexError):
        print("❌ Invalid choice.")
        return None
    
def validate_and_compare():
    print("\n--- Validate and Compare Metrics ---")
    
    # 1. Use the interactive selector to pick the global model
    global_model_path = select_file_interactive("Select a Global Model to Validate:", f"{DOWNLOADED_MODELS_DIR}/*.pt")
    if not global_model_path:
        return

    global_model = YOLO(global_model_path)
    global_names = global_model.names
    print(f"\n🌍 Global Model Classes: {global_names}")

    voc_classes = {
        0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle',
        5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow',
        10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
        15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'
    }

    global_to_voc = {
        g_id: v_id for g_id, g_name in global_names.items()
        for v_id, v_name in voc_classes.items() if g_name.lower() == v_name.lower()
    }

    datasets_root = settings['datasets_dir']
    base_dir = os.path.join(datasets_root, 'VOC')

    # Find the validation dataset path
    for ldir, idir in [('labels/val2012', 'images/val2012'), ('labels/val', 'images/val'), ('labels/train2012', 'images/train2012')]:
        labels_dir = os.path.join(base_dir, ldir)
        images_dir = os.path.join(base_dir, idir)
        if os.path.exists(labels_dir): break

    if not os.path.exists(labels_dir):
        print(f"❌ Could not find VOC labels directory in {base_dir}")
        return

    val_dir = os.path.abspath("./global_val_data")
    out_img_dir = os.path.join(val_dir, "images", "val")
    out_lbl_dir = os.path.join(val_dir, "labels", "val")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # Build the merged validation dataset
    images_copied = 0
    for label_file in os.listdir(labels_dir)[:500]:
        if not label_file.endswith('.txt'): continue
        with open(os.path.join(labels_dir, label_file), 'r') as f: lines = f.readlines()
        
        filtered = []
        for line in lines:
            if not line.strip(): continue
            try:
                orig_id = int(line.split()[0])
                for g_id, v_id in global_to_voc.items():
                    if orig_id == v_id:
                        parts = line.split()
                        parts[0] = str(g_id)
                        filtered.append(" ".join(parts) + "\n")
            except ValueError: pass
            
        if filtered:
            img_filename = label_file.replace('.txt', '.jpg')
            src_img = os.path.join(images_dir, img_filename)
            if os.path.exists(src_img):
                with open(os.path.join(out_lbl_dir, label_file), 'w') as f: f.writelines(filtered)
                shutil.copy(src_img, os.path.join(out_img_dir, img_filename))
                images_copied += 1

    yaml_path = "global_val.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump({"path": val_dir, "train": "images/val", "val": "images/val", "nc": len(global_names), "names": global_names}, f, sort_keys=False)

    print(f"\n⚙️ Running Global Model Validation ({images_copied} images)...")
    global_metrics = global_model.val(data=yaml_path, split='val', verbose=False)

    # 2. Evaluate Local Models
    local_classes_input = input("\nLocal classes to compare (comma-separated, e.g. 'person,car'): ").strip()
    local_classes = [c.strip() for c in local_classes_input.split(',')] if local_classes_input else []

    local_results = {}
    for local_class in local_classes:
        # Dynamically find the LATEST local run for this class
        pattern = f"{LOCAL_MODELS_DIR}/client_{local_class}_*/weights/best.pt"
        matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        
        local_yaml = f"client_{local_class}.yaml"
        if matches and os.path.exists(local_yaml):
            latest_model_path = matches[0]
            print(f"⚙️ Running Local Validation for '{local_class}' using {os.path.basename(os.path.dirname(os.path.dirname(latest_model_path)))}...")
            local_model = YOLO(latest_model_path)
            local_metrics = local_model.val(data=local_yaml, split='val', verbose=False)
            local_results[local_class] = local_metrics.box.map50
        else:
            print(f"⚠️ Latest local model or YAML for '{local_class}' not found. Skipping.")

    # 3. Print Results
    print("\n" + "=" * 50)
    print("🏆 FEDERATED LEARNING METRICS COMPARISON 🏆")
    print("=" * 50)
    for local_class, map50 in local_results.items():
        print(f"\n[ LOCAL MODEL ('{local_class}' only) ]")
        print(f"  - Overall mAP@50:      {map50:.3f}")
        print(f"  - Classes Known:       1 ({local_class})")
        
    print(f"\n[ GLOBAL MODEL (Merged Knowledge) ]")
    print(f"  - Target Model:        {os.path.basename(global_model_path)}")
    print(f"  - Overall mAP@50:      {global_metrics.box.map50:.3f}")
    print(f"  - Classes Known:       {len(global_names)}")
    
    print("\n[ GLOBAL MODEL Class Breakdown ]")
    for i, class_name in global_names.items():
        try:
            idx = global_metrics.ap_class_index.tolist().index(i)
            print(f"  - {class_name:<15}: mAP@50 = {global_metrics.box.maps[idx]:.3f}")
        except ValueError:
            print(f"  - {class_name:<15}: No instances found in validation sample.")
    print("=" * 50)

def setup_local_dataset(target_class_name, max_images=None):
    """Filters VOC dataset for the target class and splits into Train/Val."""
    model_name = "yolo26n.pt"
    if not os.path.exists(model_name):
        download(f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_name}")

    datasets_root = settings['datasets_dir']
    base_dir = os.path.join(datasets_root, 'VOC')
    if not os.path.exists(base_dir):
        try: YOLO(model_name).train(data="VOC.yaml", epochs=0, imgsz=640)
        except Exception: pass

    for ldir, idir in [('labels/train', 'images/train'), ('labels/train2012', 'images/train2012'), ('labels', 'images')]:
        labels_dir = os.path.join(base_dir, ldir)
        images_dir = os.path.join(base_dir, idir)
        if os.path.exists(labels_dir): break

    voc_classes = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person', 15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'}
    original_id = next((k for k, v in voc_classes.items() if v.lower() == target_class_name.lower()), None)
    if original_id is None: raise ValueError(f"Class '{target_class_name}' not found.")

    client_id = f"client_{target_class_name}"
    out_dir = os.path.abspath(f"./{client_id}_data")
    if os.path.exists(out_dir): shutil.rmtree(out_dir)

    for split in ('train', 'val'):
        os.makedirs(os.path.join(out_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'labels', split), exist_ok=True)

    all_samples = []
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'): continue
        with open(os.path.join(labels_dir, label_file), 'r') as f: lines = f.readlines()
        filtered = []
        for line in lines:
            if not line.strip(): continue
            try:
                if int(line.split()[0]) == original_id:
                    parts = line.split()
                    parts[0] = "0"
                    filtered.append(" ".join(parts) + "\n")
            except ValueError: pass
        if filtered:
            img_filename = label_file.replace('.txt', '.jpg')
            if os.path.exists(os.path.join(images_dir, img_filename)):
                all_samples.append((label_file, img_filename, filtered))
        if max_images and len(all_samples) >= max_images: break

    split_idx = int(len(all_samples) * 0.8)
    splits = {'train': all_samples[:split_idx], 'val': all_samples[split_idx:]}

    for split, samples in splits.items():
        for label_file, img_filename, filtered in samples:
            with open(os.path.join(out_dir, 'labels', split, label_file), 'w') as f: f.writelines(filtered)
            shutil.copy(os.path.join(images_dir, img_filename), os.path.join(out_dir, 'images', split, img_filename))

    yaml_path = f"{client_id}.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump({"path": out_dir, "train": "images/train", "val": "images/val", "nc": 1, "names": {0: target_class_name}}, f, sort_keys=False)
    return yaml_path, client_id


# ==============================================================================
# PIPELINE ACTIONS
# ==============================================================================

def train_and_send():
    server_info = fetch_server_info()
    if not server_info: return
    strategy = server_info['strategy']

    target_class = input("\nEnter class to train (e.g., 'person', 'car'): ").strip()
    limit_input = input("Max images? (Enter for ALL): ").strip()
    max_images = int(limit_input) if limit_input.isdigit() else None
    
    # --- Dataset Setup ---
    model_name = "yolo26n.pt"
    if not os.path.exists(model_name): download(f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_name}")
    # (Assume VOC dataset is pre-downloaded for brevity here, or re-insert the filtering logic from v4)
    # FOR THIS SCRIPT, assuming you have setup_local_dataset() defined as before.
    yaml_path, _ = setup_local_dataset(target_class, max_images)

    # --- Handshake Routing ---
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    client_run_name = f"client_{target_class}_{strategy}_{ts}"

    if strategy in ['ties', 'fedavg']:
        starting_model = "yolo26n.pt"
        SmartFLTrainer.is_round_1 = True
    else:
        global_model = download_global_model(strategy)
        if global_model:
            starting_model = global_model
            SmartFLTrainer.is_round_1 = False
        else:
            print("[FL] No global model found. Acting as Round 1.")
            starting_model = "yolo26n.pt"
            SmartFLTrainer.is_round_1 = True

    SmartFLTrainer.strategy = strategy

    print(f"\n🚀 Training starting -> {client_run_name}")
    trainer = SmartFLTrainer(overrides=dict(
        model=starting_model, data=yaml_path, epochs=50, imgsz=640,
        name=client_run_name, lr0=0.01, patience=15
    ))
    trainer.train()

    best_weights = os.path.abspath(f"runs/detect/{client_run_name}/weights/best.pt")
    if os.path.exists(best_weights):
        ssh_transfer(client_run_name, best_weights, target_class)

def ssh_transfer(client_id, weights_path, class_name):
    print(f"\nUploading {weights_path} to server...")
    ssh = _ssh_connect()
    sftp = ssh.open_sftp()
    sftp.put(weights_path, f"{SERVER_UPLOAD_DIR}/{client_id}_weights.pt")
    
    meta = {"client_id": client_id, "class_name": class_name}
    with open("meta.json", "w") as f: json.dump(meta, f)
    sftp.put("meta.json", f"{SERVER_UPLOAD_DIR}/{client_id}_meta.json")
    sftp.close()
    ssh.close()
    print("✅ Transfer complete!")

def trigger_server_reset():
    """Sends a hot-reset command to the server to start a fresh FL session."""
    print("\n⚠️ WARNING: This will archive the server's current global model.")
    print("The next client to train will start a brand new Round 1.")
    confirm = input("Are you sure you want to reset the server session? (y/N): ").strip().lower()
    
    if confirm == 'y':
        print(f"\n🔄 Connecting to {SERVER_IP} to send reset command...")
        try:
            # Create the trigger file
            with open("CMD_RESET.json", "w") as f:
                json.dump({"command": "reset", "timestamp": str(datetime.now())}, f)

            ssh = _ssh_connect()
            sftp = ssh.open_sftp()
            sftp.put("CMD_RESET.json", f"{SERVER_UPLOAD_DIR}/CMD_RESET.json")
            sftp.close()
            ssh.close()
            
            os.remove("CMD_RESET.json")
            print("✅ Reset command sent! The server is now starting a fresh session.")
        except Exception as e:
            print(f"❌ Failed to send reset command: {e}")
    else:
        print("Cancel: Server session remains active.")

def run_inference():
    print("\n--- Run Inference ---")
    print("1. Select a Global Model")
    print("2. Select a Local Model")
    choice = input("Select: ").strip()

    model_path = None
    if choice == '1':
        model_path = select_file_interactive("Select a Global Model:", f"{DOWNLOADED_MODELS_DIR}/*.pt")
    elif choice == '2':
        model_path = select_file_interactive("Select a Local Model:", f"{LOCAL_MODELS_DIR}/**/best.pt")

    if not model_path: return
    
    img_path = input("\nImage path (Enter for bus.jpg): ").strip() or "bus.jpg"
    if img_path == "bus.jpg" and not os.path.exists(img_path): download("https://ultralytics.com/images/bus.jpg")

    model = YOLO(model_path)
    model.predict(source=img_path, save=True, show=False)
    print("✅ Inference complete. Check runs/detect/predict/")

if __name__ == "__main__":
    while True:
        print("\n=== DAFYOLO Smart Client (v5.1) ===")
        print("1. Sync & Train (Auto-Configured)")
        print("2. Download Global Model Backup")
        print("3. Run Inference (Interactive Selector)")
        print("4. Validate and Compare Metrics 📊")
        print("5. 🧨 Start New Server Session (Hot Reset)")
        print("6. Exit")

        choice = input("Select (1-6): ").strip()
        if choice == '1': train_and_send()
        elif choice == '2': 
            server_info = fetch_server_info()
            if server_info:
                path = download_global_model(server_info['strategy'])
                if path: print(f"✅ Saved global model to {path}")
        elif choice == '3': run_inference()
        elif choice == '4': validate_and_compare()
        elif choice == '5': trigger_server_reset()
        elif choice == '6': break