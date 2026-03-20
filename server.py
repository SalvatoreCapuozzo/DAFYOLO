import os
import time
import json
import torch
from ultralytics import YOLO

# --- Server Config ---
UPLOAD_DIR = "/datadrive/DAFYOLO/uploads" # MUST MATCH CLIENT SCRIPT
GLOBAL_MODEL_DIR = "/datadrive/DAFYOLO/global_model"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GLOBAL_MODEL_DIR, exist_ok=True)

class FLServer:
    def __init__(self):
        print("Initializing Global YOLO26 Model...")
        self.global_model = YOLO("yolo26n.pt")
        self.registry = {} # e.g. {"person": 0, "car": 1}
        self.nc = 0
        
    def expand_head(self):
        """Surgically expands the final classification layer for YOLO26 one-to-one head."""
        # Note: YOLO internal structures vary. This targets the standard Detect/C2f layers.
        # We find layers matching the old number of classes and expand them.
        model_dict = self.global_model.model.state_dict()
        
        for name, param in model_dict.items():
            if 'cv3' in name or 'cls' in name: # Target classification heads
                if param.shape[0] == self.nc - 1: # It has the old number of classes
                    new_shape = list(param.shape)
                    new_shape[0] = self.nc
                    new_param = torch.zeros(new_shape, device=param.device)
                    
                    # Copy old weights
                    new_param[:self.nc-1] = param
                    # Initialize new class weights randomly to break symmetry
                    torch.nn.init.normal_(new_param[self.nc-1:], std=0.01)
                    
                    model_dict[name] = new_param

        # Update the model's internal class count
        self.global_model.model.nc = self.nc
        self.global_model.model.names = {v: k for k, v in self.registry.items()}
        self.global_model.model.load_state_dict(model_dict, strict=False)

    def merge_client(self, client_weights_path, class_name, alpha=0.5):
        print(f"\nProcessing weights for class: {class_name}")
        
        if class_name not in self.registry:
            self.registry[class_name] = self.nc
            self.nc += 1
            if self.nc > 80: # YOLO26n starts with 80 classes, we only expand if we exceed base or if we stripped it
                pass # For simplicity in this script, we assume standard merging. 
                     # If starting from scratch (nc=1), expand_head() handles the tensor resizing.
            print(f"Discovered new class '{class_name}'. Global ID: {self.registry[class_name]}")
            
        global_sd = self.global_model.model.state_dict()
        client_model = YOLO(client_weights_path)
        client_sd = client_model.model.state_dict()
        
        target_id = self.registry[class_name]
        
        for key in global_sd.keys():
            if key in client_sd:
                if 'cv3' in key or 'cls' in key: # Classification layer
                    # YOLO classification tensors are usually shaped [nc, channels, ...]
                    # ONLY update the specific index for the class the client trained on
                    if len(global_sd[key].shape) > 0 and global_sd[key].shape[0] > target_id:
                        global_sd[key][target_id] = (
                            (1 - alpha) * global_sd[key][target_id] + 
                            alpha * client_sd[key][0] # Client only has 1 class, so it's at index 0
                        )
                else: # Backbone/Neck
                    global_sd[key] = (1 - alpha) * global_sd[key] + alpha * client_sd[key]
                    
        self.global_model.model.load_state_dict(global_sd)
        
        # Save the new global model
        out_path = os.path.join(GLOBAL_MODEL_DIR, "global_model.pt")
        self.global_model.save(out_path)
        print(f"Global model updated and saved to {out_path}!")

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
                # We have both files!
                server.merge_client(weights_path, meta['class_name'])
                
                # Cleanup so we don't process them again
                os.remove(meta_path)
                os.remove(weights_path)
                
        time.sleep(5) # Polling interval

if __name__ == "__main__":
    run_server()