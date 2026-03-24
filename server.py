import os
import time
import json
import torch
import torch.nn as nn
import shutil
from datetime import datetime
from ultralytics import YOLO

# --- Server Config ---
UPLOAD_DIR = "/datadrive/DAFYOLO/uploads" 
GLOBAL_MODEL_DIR = "/datadrive/DAFYOLO/global_model"
PROCESSED_DIR = "/datadrive/DAFYOLO/processed_models"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GLOBAL_MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

class FLServer:
    def __init__(self):
        self.global_model = None
        self.registry = {} 
        self.nc = 0
        print("Server Initialized. Waiting for first client...")

    def _expand_classification_head(self):
        """Dynamically expands the final YOLO Conv2d layers by 1 channel."""
        print(f"Expanding PyTorch classification head from {self.nc} to {self.nc + 1} classes...")
        head = self.global_model.model.model[-1] 
        
        for i in range(len(head.cv3)):
            seq = head.cv3[i] 
            last_idx = len(seq) - 1
            last_layer = seq[last_idx]
            old_conv = last_layer if isinstance(last_layer, nn.Conv2d) else last_layer.conv
                
            new_conv = nn.Conv2d(
                in_channels=old_conv.in_channels,
                out_channels=self.nc + 1,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None)
            ).to(old_conv.weight.device)
            
            with torch.no_grad():
                new_conv.weight[:self.nc] = old_conv.weight
                nn.init.normal_(new_conv.weight[self.nc:], std=0.01) 
                if old_conv.bias is not None:
                    new_conv.bias[:self.nc] = old_conv.bias
                    nn.init.zeros_(new_conv.bias[self.nc:])
                    
            if isinstance(last_layer, nn.Conv2d):
                seq[last_idx] = new_conv
            else:
                seq[last_idx].conv = new_conv
                
        self.nc += 1
        self.global_model.model.nc = self.nc

    def merge_client(self, client_weights_path, class_name, alpha=0.5):
        print(f"\n--- Processing Incoming Client: '{class_name}' ---")
        
        # 1. Base Initialization
        if self.global_model is None:
            print(f"First client detected. Initializing Global Model...")
            # CRITICAL FIX: We must load the base pretrained model first to get its flawless cv2/dfl layers
            self.global_model = YOLO("yolo26n.pt") 
            
            # Now we surgically slice the pretrained 80-class cv3 down to 1 class
            # to match the architecture size we need, but we DO NOT touch cv2.
            self.nc = 1
            self.registry[class_name] = 0
            self.global_model.model.names = {0: class_name}
            
            head = self.global_model.model.model[-1]
            for i in range(len(head.cv3)):
                seq = head.cv3[i]
                last_layer = seq[-1]
                old_conv = last_layer if isinstance(last_layer, nn.Conv2d) else last_layer.conv
                
                new_conv = nn.Conv2d(old_conv.in_channels, 1, old_conv.kernel_size, old_conv.stride, old_conv.padding, bias=(old_conv.bias is not None)).to(old_conv.weight.device)
                with torch.no_grad():
                    # Keep the pretrained weights for the first channel to maintain feature stability
                    new_conv.weight[0] = old_conv.weight[0] 
                    if old_conv.bias is not None:
                        new_conv.bias[0] = old_conv.bias[0]
                
                if isinstance(last_layer, nn.Conv2d): seq[-1] = new_conv
                else: seq[-1].conv = new_conv
            
            self.global_model.model.nc = 1
            
            # Immediately run the merge logic below to apply the client's trained updates to this new base
            
        # 2. Dynamic Expansion
        elif class_name not in self.registry:
            self.registry[class_name] = self.nc
            self.global_model.model.names[self.nc] = class_name
            self._expand_classification_head()
            print(f"Discovered new class '{class_name}'. Assigned Global ID: {self.registry[class_name]}")

        target_id = self.registry[class_name]
        print(f"Merging Client's Local ID 0 into Server's Global ID {target_id}...")
        
        global_sd = self.global_model.model.state_dict()
        client_model = YOLO(client_weights_path)
        client_sd = client_model.model.state_dict()
        
        # 3. STRICT PARTIAL AGGREGATION
        for key in global_sd.keys():
            if key in client_sd:
                # HEAD (cv3): Splice in the classification weights for this specific class
                if 'cv3' in key and ('weight' in key or 'bias' in key):
                    if global_sd[key].shape[0] == self.nc and client_sd[key].shape[0] == 1:
                        global_sd[key][target_id] = (1 - alpha) * global_sd[key][target_id] + alpha * client_sd[key][0]
                
                # BOXES (cv2, dfl): STRICTLY IGNORE THE CLIENT. 
                # The client's cv2 was randomly reinitialized and will destroy the model if merged.
                elif 'cv2' in key or 'dfl' in key:
                    continue 
                    
                # BACKBONE/NECK: Standard FedAvg. Shapes must match.
                elif global_sd[key].shape == client_sd[key].shape:
                    global_sd[key] = (1 - alpha) * global_sd[key] + alpha * client_sd[key]
                        
        self.global_model.model.load_state_dict(global_sd)
        self._save_model()

    def _save_model(self):
        out_path = os.path.join(GLOBAL_MODEL_DIR, "global_model.pt")
        self.global_model.save(out_path)
        print(f"✅ Global model updated and saved: {out_path}")
        print(f"Current Global Classes: {self.registry}")

def run_server():
    server = FLServer()
    print(f"Server listening for SSH uploads in {UPLOAD_DIR}...")
    print(f"Successfully merged models will be archived in {PROCESSED_DIR}...")
    
    while True:
        files = os.listdir(UPLOAD_DIR)
        meta_files = [f for f in files if f.endswith('_meta.json')]
        
        for meta_file in meta_files:
            meta_path = os.path.join(UPLOAD_DIR, meta_file)
            
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
            except json.JSONDecodeError:
                # Sometimes the server reads the file right as SFTP is still writing it
                # Skip it this second, it will read cleanly on the next loop iteration
                continue
                
            client_id = meta.get('client_id')
            class_name = meta.get('class_name')
            
            if not client_id or not class_name:
                print(f"⚠️ Invalid meta file found: {meta_file}. Skipping.")
                continue

            weights_file = f"{client_id}_weights.pt"
            weights_path = os.path.join(UPLOAD_DIR, weights_file)
            
            # Check if the weight file has fully arrived
            if os.path.exists(weights_path):
                # 1. Merge the model
                server.merge_client(weights_path, class_name)
                
                # 2. Archive the files instead of deleting them!
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Create unique archive names so clients can upload multiple times
                archived_meta = os.path.join(PROCESSED_DIR, f"{client_id}_{timestamp}_meta.json")
                archived_weights = os.path.join(PROCESSED_DIR, f"{client_id}_{timestamp}_weights.pt")
                
                # Move them to the vault
                shutil.move(meta_path, archived_meta)
                shutil.move(weights_path, archived_weights)
                
                print(f"📦 Archived {client_id} to the processed vault!")
                
        time.sleep(5) # Polling interval

if __name__ == "__main__":
    run_server()