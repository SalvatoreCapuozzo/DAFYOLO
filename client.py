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
    YOLO(model_name).train(data="VOC.yaml", epochs=1, imgsz=640)
    
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
    print(f"Uploading {weights_path} to {remote_weights}...")
    sftp.put(weights_path, remote_weights)
    
    # Upload metadata
    meta = {"client_id": client_id, "class_name": client_id.split('_')[1]}
    with open("meta.json", "w") as f: 
        json.dump(meta, f)
        
    remote_meta = f"{SERVER_UPLOAD_DIR}/{client_id}_meta.json"
    print(f"Uploading metadata to {remote_meta}...")
    sftp.put("meta.json", remote_meta)
    
    sftp.close()
    transport.close()
    print("Transfer complete! The server should process it shortly.")

def send_existing_weights():
    """Standalone function to transfer already trained weights."""
    target_class = input("\nEnter the class name you already trained (e.g., 'person', 'car'): ").strip()
    client_id = f"client_{target_class}"
    
    # Construct the path where YOLO saved the weights during the previous run
    best_weights = os.path.abspath(f"runs/detect/{client_id}/train/weights/best.pt")
    
    if os.path.exists(best_weights):
        print(f"Found existing weights at: {best_weights}")
        try:
            ssh_transfer(client_id, best_weights)
        except Exception as e:
            print(f"\n❌ SSH Transfer failed. Check your server IP, user, password, and paths.")
            print(f"Error details: {e}")
    else:
        print(f"\n❌ Could not find weights at {best_weights}.")
        print("Are you sure you trained this class already? (Check your folder structure)")

def download_and_inspect_global_model():
    """Connects to server, downloads global_model.pt, and prints its knowledge."""
    print(f"\nConnecting to server {SERVER_IP} via SSH...")
    try:
        transport = paramiko.Transport((SERVER_IP, SSH_PORT))
        transport.connect(username=SSH_USER, password=SSH_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        
        # Define the remote and local paths
        remote_global_model = f"{SERVER_DOWNLOAD_DIR}/global_model.pt"
        local_global_model = "./downloaded_global_model.pt"
        
        print(f"Downloading {remote_global_model}...")
        sftp.get(remote_global_model, local_global_model)
        
        sftp.close()
        transport.close()
        print("Download complete!\n")
        
        # Load the model locally to inspect its brain
        print("--- Inspecting Global Model ---")
        global_model = YOLO(local_global_model)
        
        # Extract the classes
        classes = global_model.model.names
        nc = global_model.model.nc
        
        print(f"Total Global Classes Discovered: {nc}")
        print("Class Mapping:")
        for global_id, name in classes.items():
            print(f"  [ID: {global_id}] -> {name}")
            
        print("\nModel is ready for inference or further training!")
        print(f"Saved locally as: {os.path.abspath(local_global_model)}")
        
    except FileNotFoundError:
        print("\n❌ Error: The global model doesn't exist on the server yet.")
        print("You need to train and upload at least one client first!")
    except Exception as e:
        print(f"\n❌ SSH Download failed. Error details: {e}")

def train_and_send():
    """The full pipeline: Extracts dataset, trains, and transfers."""
    target_class = input("\nEnter the class to train on (e.g., 'person', 'car'): ").strip()
    yaml_path, client_id = setup_local_dataset(target_class)
    
    print(f"\nStarting YOLO26 Local Training for {client_id}...")
    model = YOLO("yolo26n.pt")
    
    # Using 3 epochs for a fast local simulation
    #results = model.train(data=yaml_path, epochs=3, imgsz=640, project=client_id, name="train") 
    project_dir = f"runs/detect/{client_id}"
    results = model.train(data=yaml_path, epochs=1, imgsz=640, project=project_dir, name="train", freeze=12)
    
    best_weights = os.path.abspath(f"runs/detect/{client_id}/train/weights/best.pt")
    
    if os.path.exists(best_weights):
        print("\n✅ Training complete! Preparing to send weights...")
        try:
            ssh_transfer(client_id, best_weights)
        except Exception as e:
            print(f"\n❌ SSH Transfer failed (Is your server running?): {e}")
            print(f"Your weights are safely saved locally at: {best_weights}")
            print("You can try sending them again later using Option 2.")
    else:
        print("\n❌ Failed to send weights.")

def run_inference():
    """Lets the user choose a model (local or global) to run inference on an image."""
    print("\n--- Run Inference ---")
    print("Which model would you like to use?")
    print("1. The Downloaded Global Model (Combined Knowledge)")
    print("2. A Locally Trained Model (Single Class Knowledge)")
    print("3. Cancel")
    
    choice = input("Select an option (1-3): ").strip()
    
    if choice == '1':
        model_path = "./downloaded_global_model.pt"
        if not os.path.exists(model_path):
            print(f"\n❌ Error: '{model_path}' not found.")
            print("Please use Option 3 to download the global model first!")
            return
            
    elif choice == '2':
        target_class = input("Enter the class name you trained locally (e.g., 'person', 'car'): ").strip()
        client_id = f"runs/detect/client_{target_class}"
        model_path = os.path.abspath(f"{client_id}/train/weights/best.pt")
        
        if not os.path.exists(model_path):
            print(f"\n❌ Error: Could not find locally trained weights at {model_path}.")
            print("Are you sure you trained this class? (Check Option 1)")
            return
            
    elif choice == '3':
        return
        
    else:
        print("Invalid choice. Returning to main menu.")
        return

    print(f"\nLoading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Failed to load model. Error: {e}")
        return
    
    # Get image path from user, or use a default test image
    print("\nEnter the path to a test image (e.g., 'test.jpg').")
    print("Or press Enter to use the default test image (bus.jpg).")
    img_path = input("Image Path: ").strip()
    
    if img_path == "":
        img_path = "bus.jpg"
        if not os.path.exists(img_path):
            print("Downloading default test image...")
            download("https://ultralytics.com/images/bus.jpg")
            
    if not os.path.exists(img_path):
        print(f"❌ Error: Image '{img_path}' not found.")
        return

    print(f"\nAnalyzing '{img_path}'...")
    
    # Run prediction
    results = model.predict(source=img_path, save=True, show=False, conf=0.25)
    
    # Print a summary of what it found
    result = results[0]
    boxes = result.boxes
    print(f"\n✅ Detection Complete! Found {len(boxes)} objects.")
    
    if len(boxes) > 0:
        names = model.names
        detections = {}
        for box in boxes:
            cls_id = int(box.cls[0].item())
            class_name = names.get(cls_id, f"Unknown ID {cls_id}")
            detections[class_name] = detections.get(class_name, 0) + 1
            
        print("\nObjects Detected:")
        for name, count in detections.items():
            print(f"  - {name}: {count}")
    else:
        print("No objects detected above the confidence threshold (0.25).")

    if result.save_dir:
        save_path = os.path.join(result.save_dir, os.path.basename(img_path))
        print(f"\n🖼️ Saved annotated image to: {os.path.abspath(save_path)}")

def validate_and_compare():
    """Builds a joint validation dataset and compares Global vs Local model metrics."""
    global_model_path = "./downloaded_global_model.pt"
    if not os.path.exists(global_model_path):
        print(f"\n❌ Error: '{global_model_path}' not found. Use Option 3 first!")
        return

    print("\n--- Validating Models ---")
    global_model = YOLO(global_model_path)
    global_names = global_model.names
    
    if len(global_names) < 2:
        print("⚠️ The global model only knows 1 class. Merge at least 2 clients to see the FL benefit!")
    
    print(f"Global Model Knowledge: {global_names}")
    
    # 1. Reverse lookup VOC IDs for the classes the global model knows
    voc_classes = {
        0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle',
        5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow',
        10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
        15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'
    }
    
    # Map Global ID -> Original VOC ID
    global_to_voc = {}
    for g_id, g_name in global_names.items():
        for v_id, v_name in voc_classes.items():
            if g_name.lower() == v_name.lower():
                global_to_voc[g_id] = v_id
                break

    # 2. Build the Joint Validation Dataset
    print("\nBuilding Joint Validation Dataset for fair comparison...")
    datasets_root = settings['datasets_dir']
    base_dir = os.path.join(datasets_root, 'VOC')
    labels_dir = os.path.join(base_dir, 'labels', 'val') # Use VOC validation set
    images_dir = os.path.join(base_dir, 'images', 'val')
    
    # Fallback to train if val doesn't exist (depending on VOC download structure)
    if not os.path.exists(labels_dir):
        labels_dir = os.path.join(base_dir, 'labels', 'train')
        images_dir = os.path.join(base_dir, 'images', 'train')

    val_dir = os.path.abspath("./global_val_data")
    out_img_dir = os.path.join(val_dir, "images", "val")
    out_lbl_dir = os.path.join(val_dir, "labels", "val")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    images_copied = 0
    # Process up to 500 images for a fast but statistically significant validation
    for label_file in os.listdir(labels_dir)[:500]:
        if not label_file.endswith('.txt'): continue
        
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            lines = f.readlines()
            
        filtered = []
        for line in lines:
            if not line.strip(): continue
            try:
                orig_id = int(line.split()[0])
                # If this object is one of our global classes, rewrite it to the Global ID
                for g_id, v_id in global_to_voc.items():
                    if orig_id == v_id:
                        parts = line.split()
                        parts[0] = str(g_id)
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

    yaml_path = "global_val.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump({
            "path": val_dir, "train": "images/val", "val": "images/val", 
            "nc": len(global_names), "names": global_names
        }, f, sort_keys=False)

    print(f"Generated validation set with {images_copied} joint images.")

    # 3. Evaluate Global Model
    print("\n--- Evaluating GLOBAL Model ---")
    global_metrics = global_model.val(data=yaml_path, split='val', verbose=False)
    
    # 4. Evaluate Local Model
    local_class = input("\nEnter the class name you trained locally to compare (e.g., 'person'): ").strip()
    client_id = f"runs/detect/client_{local_class}"
    local_model_path = os.path.abspath(f"{client_id}/train/weights/best.pt")
    local_yaml = f"client_{local_class}.yaml"
    
    if os.path.exists(local_model_path) and os.path.exists(local_yaml):
        print(f"\n--- Evaluating LOCAL Model ({local_class}) ---")
        local_model = YOLO(local_model_path)
        # We evaluate the local model on ITS OWN isolated validation set to show what it knows
        local_metrics = local_model.val(data=local_yaml, split='val', verbose=False)
        
        # --- Print the Comparison ---
        print("\n" + "="*50)
        print("🏆 FEDERATED LEARNING METRICS COMPARISON 🏆")
        print("="*50)
        
        print(f"\n[ LOCAL MODEL ('{local_class}' only) ]")
        print(f"  - Overall mAP@50:      {local_metrics.box.map50:.3f}")
        print(f"  - Classes Known:       1 ({local_class})")
        
        print(f"\n[ GLOBAL MODEL (Merged Knowledge) ]")
        print(f"  - Overall mAP@50:      {global_metrics.box.map50:.3f}")
        print(f"  - Classes Known:       {len(global_names)}")
        
        print("\n[ GLOBAL MODEL Class Breakdown ]")
        for i, class_name in global_names.items():
            # YOLO metrics arrays are indexed by the classes present in the validation set
            try:
                # Find the index of this class in the metrics results
                metric_idx = global_metrics.ap_class_index.tolist().index(i)
                class_map50 = global_metrics.box.maps[metric_idx]
                print(f"  - {class_name:<15}: mAP@50 = {class_map50:.3f}")
            except ValueError:
                print(f"  - {class_name:<15}: No instances found in validation sample.")
                
        print("="*50)
    else:
        print(f"\n❌ Local model or YAML for '{local_class}' not found. Skipping local comparison.")

if __name__ == "__main__":
    while True:
        print("\n=== DAFYOLO Federated Learning Client ===")
        print("1. Train a new model and send weights to server")
        print("2. Send EXISTING weights to server (skip training)")
        print("3. Download Global Model & Inspect Classes")
        print("4. Run Inference (Local or Global Model)")
        print("5. Validate and Compare Metrics 📊")
        print("6. Exit")
        
        choice = input("Select an option (1-6): ").strip()
        
        if choice == '1':
            train_and_send()
        elif choice == '2':
            send_existing_weights()
        elif choice == '3':
            download_and_inspect_global_model()
        elif choice == '4':
            run_inference()
        elif choice == '5':
            validate_and_compare()
        elif choice == '6':
            print("Exiting. Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1-6.")