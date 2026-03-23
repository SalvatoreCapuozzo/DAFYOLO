import os
import time
import json
import torch
import torch.nn as nn
from ultralytics import YOLO

# --- Server Config ---
UPLOAD_DIR = "/datadrive/DAFYOLO/uploads" # Assuming this is your path based on logs
GLOBAL_MODEL_DIR = "/datadrive/DAFYOLO/global_model"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GLOBAL_MODEL_DIR, exist_ok=True)

class FLServer:
    def __init__(self):
        self.global_model = None
        self.registry = {} # e.g. {"person": 0, "car": 1}
        self.nc = 0
        print("Server Initialized. Waiting for first client to define base architecture...")

    def _expand_classification_head(self):
        """Dynamically expands the final YOLO Conv2d layers by 1 channel."""
        print(f"Expanding PyTorch model head from {self.nc} classes to {self.nc + 1} classes...")
        head = self.global_model.model.model[-1] # The Detect head
        
        for i in range(len(head.cv3)):
            seq = head.cv3[i] # This is a sequence of layers for one feature map scale
            last_idx = len(seq) - 1
            last_layer = seq[last_idx]
            
            # Locate the actual Conv2d object
            old_conv = last_layer if isinstance(last_layer, nn.Conv2d) else last_layer.conv
                
            new_conv = nn.Conv2d(
                in_channels=old_conv.in_channels,
                out_channels=self.nc + 1,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None)
            ).to(old_conv.weight.device)
            
            # Copy old weights and initialize the new channel
            with torch.no_grad():
                new_conv.weight[:self.nc] = old_conv.weight
                nn.init.normal_(new_conv.weight[self.nc:], std=0.01) # Break symmetry
                if old_conv.bias is not None:
                    new_conv.bias[:self.nc] = old_conv.bias
                    nn.init.zeros_(new_conv.bias[self.nc:])
                    
            # Inject the new layer back into the model
            if isinstance(last_layer, nn.Conv2d):
                seq[last_idx] = new_conv
            else:
                seq[last_idx].conv = new_conv
                
        # Update internal config variables
        self.nc += 1
        self.global_model.model.nc = self.nc

    def merge_client(self, client_weights_path, class_name, alpha=0.5):
        print(f"\n--- Processing Incoming Client: '{class_name}' ---")
        
        if self.global_model is None:
            print(f"First client detected! Initializing global model from '{class_name}'.")
            self.global_model = YOLO(client_weights_path)
            self.registry[class_name] = 0
            self.nc = 1
            self.global_model.model.names = {0: class_name}
            self._save_model()
            return

        if class_name not in self.registry:
            self.registry[class_name] = self.nc
            self.global_model.model.names[self.nc] = class_name
            self._expand_classification_head()
            print(f"Discovered new class '{class_name}'. Assigned Global ID: {self.registry[class_name]}")

        target_id = self.registry[class_name]
        print(f"Merging Client's Local ID 0 into Server's Global ID {target_id}...")
        
        global_sd = self.global_model.model.state_dict()
        client_model = YOLO(client_weights_path)
        client_sd = client_model.model.state_dict()
        
        # Protective Tensor Surgery
        for key in global_sd.keys():
            if key in client_sd:
                # 1. Classification Head (cv3) - Shapes WON'T match (Server has nc, Client has 1)
                if 'cv3' in key and ('weight' in key or 'bias' in key) and global_sd[key].shape[0] == self.nc and client_sd[key].shape[0] == 1:
                    global_sd[key][target_id] = (1 - alpha) * global_sd[key][target_id] + alpha * client_sd[key][0]
                
                # 2. Standard Layers (Shapes MUST match)
                elif global_sd[key].shape == client_sd[key].shape:
                    # Bounding Box Head (cv2) & DFL layers - Highly sensitive to background penalization!
                    if 'cv2' in key or 'dfl' in key:
                        global_sd[key] = 0.9 * global_sd[key] + 0.1 * client_sd[key]
                    # Shared Backbone / Neck
                    else:
                        global_sd[key] = (1 - alpha) * global_sd[key] + alpha * client_sd[key]
                else:
                    if 'cv3' not in key: # Suppress warnings for cv3 since we handle it above
                        print(f"⚠️ Warning: Unhandled shape mismatch on {key}")
                        
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
    
    while True:
        files = os.listdir(UPLOAD_DIR)
        meta_files = [f for f in files if f.endswith('_meta.json')]
        
        for meta_file in meta_files:
            meta_path = os.path.join(UPLOAD_DIR, meta_file)
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                
            client_id = meta['client_id']
            weights_file = f"{client_id}_weights.pt"
            weights_path = os.path.join(UPLOAD_DIR, weights_file)
            
            if os.path.exists(weights_path):
                server.merge_client(weights_path, meta['class_name'])
                os.remove(meta_path)
                os.remove(weights_path)
                
        time.sleep(5)

if __name__ == "__main__":
    run_server()