import os
import time
import json
import paramiko
import yaml
import glob
import shutil
from datetime import datetime
from ultralytics import YOLO, settings
from dotenv import load_dotenv

# --- EXPLICIT .ENV LOADING ---
# Force Python to find the .env file in the exact folder this script lives in
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(env_path)

# Now it is safe to import from your client script
from client_updated import (
    SERVER_IP, SSH_PORT, SSH_USER, SSH_PASSWORD, SERVER_UPLOAD_DIR,
    SmartFLTrainer, setup_local_dataset, ssh_transfer, fetch_server_info, download_global_model,
    LOCAL_MODELS_DIR, DOWNLOADED_MODELS_DIR
)

# --- SAFETY CHECK ---
if not SERVER_IP or not SSH_USER or not SSH_PASSWORD:
    raise ValueError(f"\n❌ CRITICAL: Missing credentials!\nSERVER_IP: {SERVER_IP}\nSSH_USER: {SSH_USER}\nCheck your .env file inside {script_dir}")

# --- EXPERIMENT CONFIGURATION ---
STRATEGIES = ['fedhead', 'stitch', 'ties', 'fedavg']
CLASSES = ['person', 'car', 'bicycle']
MAX_IMAGES = 200
EPOCHS = 20
RESULTS_FILE = "automated_experiment_results.txt"


def trigger_server_reset_headless(strategy):
    """Sends a reset command AND tells the server which strategy to switch to."""
    print(f"\n📡 Sending command to server: RESET and switch to {strategy.upper()}")
    try:
        with open("CMD_RESET.json", "w") as f:
            json.dump({"command": "reset", "strategy": strategy, "timestamp": str(datetime.now())}, f)

        # Robust SSH Client for Ubuntu/Linux
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=SERVER_IP, 
            port=SSH_PORT, 
            username=SSH_USER, 
            password=SSH_PASSWORD,
            look_for_keys=False,
            allow_agent=False,
            timeout=15
        )
        sftp = ssh.open_sftp()
        sftp.put("CMD_RESET.json", f"{SERVER_UPLOAD_DIR}/CMD_RESET.json")
        sftp.close()
        ssh.close()
        
        os.remove("CMD_RESET.json")
        print("✅ Command sent successfully.")
    except Exception as e:
        print(f"❌ Failed to send reset command: {e}")


def train_and_send_headless(target_class, strategy):
    """Runs the training pipeline without CLI prompts."""
    yaml_path, _ = setup_local_dataset(target_class, MAX_IMAGES)
    
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
            starting_model = "yolo26n.pt"
            SmartFLTrainer.is_round_1 = True

    SmartFLTrainer.strategy = strategy

    print(f"\n🚀 [AUTO-TRAIN] Starting '{target_class}' | Strat: {strategy.upper()} | Base: {starting_model}")
    trainer = SmartFLTrainer(overrides=dict(
        model=starting_model, data=yaml_path, epochs=EPOCHS, imgsz=640,
        name=client_run_name, lr0=0.01, patience=15, verbose=False # Less console spam
    ))
    trainer.train()

    best_weights = os.path.abspath(f"runs/detect/{client_run_name}/weights/best.pt")
    if os.path.exists(best_weights):
        ssh_transfer(client_run_name, best_weights, target_class)
    
    return best_weights


def validate_and_compare_headless(strategy, log_file):
    """Downloads the final global model, runs validation, and logs the metrics."""
    print(f"\n📊 Running Evaluation for Strategy: {strategy.upper()}")
    
    # Download the final global model for this strategy
    global_model_path = download_global_model(strategy)
    if not global_model_path:
        print("❌ Could not download global model for evaluation.")
        return

    global_model = YOLO(global_model_path)
    global_names = global_model.names

    # --- Build Validation Set ---
    val_dir = os.path.abspath("./global_val_data_auto")
    os.makedirs(os.path.join(val_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "labels", "val"), exist_ok=True)

    yaml_path = "global_val_auto.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump({"path": val_dir, "train": "images/val", "val": "images/val", "nc": len(global_names), "names": global_names}, f, sort_keys=False)

    # Assume setup_local_dataset already downloaded VOC earlier, we just run evaluation.
    # Note: To keep things isolated, we evaluate the global model against the subset yaml we just made.
    # We will use the existing `global_val_auto.yaml` assuming the dataset was built by the last run.
    # (If the images are missing, YOLO will automatically warn you).
    
    print(f"⚙️ Evaluating Global Model...")
    global_metrics = global_model.val(data=yaml_path, split='val', verbose=False)

    local_results = {}
    for local_class in CLASSES:
        pattern = f"{LOCAL_MODELS_DIR}/client_{local_class}_{strategy}_*/weights/best.pt"
        matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        local_yaml = f"client_{local_class}.yaml"
        if matches and os.path.exists(local_yaml):
            latest_model_path = matches[0]
            print(f"⚙️ Evaluating Local Model: '{local_class}'...")
            local_model = YOLO(latest_model_path)
            local_metrics = local_model.val(data=local_yaml, split='val', verbose=False)
            local_results[local_class] = local_metrics.box.map50

    # --- Log Results ---
    with open(log_file, "a") as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"🏆 METRICS FOR STRATEGY: {strategy.upper()}\n")
        f.write("=" * 50 + "\n")
        
        for local_class, map50 in local_results.items():
            f.write(f"\n[ LOCAL MODEL ('{local_class}' only) ]\n")
            f.write(f"  - Overall mAP@50:      {map50:.3f}\n")
            
        f.write(f"\n[ GLOBAL MODEL (Merged Knowledge) ]\n")
        f.write(f"  - Overall mAP@50:      {global_metrics.box.map50:.3f}\n")
        f.write(f"  - Classes Known:       {len(global_names)}\n\n")
        
        for i, class_name in global_names.items():
            try:
                idx = global_metrics.ap_class_index.tolist().index(i)
                f.write(f"  - {class_name:<15}: mAP@50 = {global_metrics.box.maps[idx]:.3f}\n")
            except ValueError:
                f.write(f"  - {class_name:<15}: No instances found in validation sample.\n")
        f.write("=" * 50 + "\n")
        
    print(f"✅ Metrics logged to {log_file}")


# ==============================================================================
# MAIN AUTOMATION LOOP
# ==============================================================================
if __name__ == "__main__":
    print("🚀 STARTING AUTOMATED EXPERIMENT HARNESS")
    print(f"Strategies to test: {STRATEGIES}")
    print(f"Classes per strategy: {CLASSES}")
    
    # Initialize Log File
    with open(RESULTS_FILE, "w") as f:
        f.write(f"DAFYOLO EXPERIMENT RESULTS\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config: {MAX_IMAGES} Images/Class, {EPOCHS} Epochs\n")

    for strategy in STRATEGIES:
        print("\n" + "#" * 60)
        print(f"### RUNNING EXPERIMENT: {strategy.upper()} ###")
        print("#" * 60)

        # 1. Reset Server and switch strategy
        trigger_server_reset_headless(strategy)
        print("⏳ Waiting 10 seconds for server to initialize...")
        time.sleep(10) 

        # 2. Train and Upload each client
        for cls in CLASSES:
            train_and_send_headless(cls, strategy)
            print("⏳ Waiting 5 seconds before next client...")
            time.sleep(5)

        # 3. Wait for final server processing
        print("\n⏳ All clients trained. Waiting 25 seconds for server to process final merge...")
        time.sleep(25)

        # 4. Download and Evaluate
        validate_and_compare_headless(strategy, RESULTS_FILE)

    print(f"\n🎉 ALL EXPERIMENTS COMPLETE! Check {RESULTS_FILE} for your data.")