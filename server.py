import os
import time
import json
import torch
import torch.nn as nn
import shutil
import copy
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
    def __init__(self, strategy='ties'):
        self.global_model = None
        self.registry = {} 
        self.nc = 0
        self.strategy = strategy
        self.bootstrap_server()

    def bootstrap_server(self):
        """Restores existing model, or rebuilds it from the archive vault."""
        global_model_path = os.path.join(GLOBAL_MODEL_DIR, "global_model.pt")
        
        if os.path.exists(global_model_path):
            print(f"Loading existing global model from {global_model_path}...")
            self.global_model = YOLO(global_model_path)
            self.registry = {v: k for k, v in self.global_model.model.names.items()}
            self.nc = len(self.registry)
            print(f"Restored Global Classes: {self.registry}")
        else:
            processed_metas = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('_meta.json')]
            if processed_metas:
                print(f"Rebuilding history from {len(processed_metas)} archived updates using strategy: {self.strategy.upper()}...")
                processed_metas.sort(key=lambda x: os.path.getmtime(os.path.join(PROCESSED_DIR, x)))
                for meta_file in processed_metas:
                    try:
                        with open(os.path.join(PROCESSED_DIR, meta_file), 'r') as f: meta = json.load(f)
                    except json.JSONDecodeError: continue
                    class_name = meta.get('class_name')
                    weights_file = meta_file.replace('_meta.json', '_weights.pt')
                    weights_path = os.path.join(PROCESSED_DIR, weights_file)
                    if os.path.exists(weights_path):
                        self.merge_client(weights_path, class_name)
            else:
                print(f"Server Initialized (Strategy: {self.strategy.upper()}). Waiting for first client...")

    def _expand_classification_head(self):
        """Dynamically expands BOTH classification heads for YOLO end-to-end models."""
        print(f"Expanding PyTorch classification head from {self.nc} to {self.nc + 1} classes...")
        head = self.global_model.model.model[-1] 
        
        cv3_lists = []
        if hasattr(head, 'cv3'): cv3_lists.append(head.cv3)
        if hasattr(head, 'one2one_cv3'): cv3_lists.append(head.one2one_cv3)
        
        for cv3_module in cv3_lists:
            for i in range(len(cv3_module)):
                seq = cv3_module[i] 
                last_idx = len(seq) - 1
                last_layer = seq[last_idx]
                old_conv = last_layer if isinstance(last_layer, nn.Conv2d) else last_layer.conv
                    
                new_conv = nn.Conv2d(
                    in_channels=old_conv.in_channels, out_channels=self.nc + 1,
                    kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                    padding=old_conv.padding, bias=(old_conv.bias is not None)
                ).to(old_conv.weight.device)
                
                with torch.no_grad():
                    new_conv.weight[:self.nc] = old_conv.weight
                    nn.init.normal_(new_conv.weight[self.nc:], std=0.01) 
                    if old_conv.bias is not None:
                        new_conv.bias[:self.nc] = old_conv.bias
                        nn.init.zeros_(new_conv.bias[self.nc:])
                        
                if isinstance(last_layer, nn.Conv2d): seq[last_idx] = new_conv
                else: seq[last_idx].conv = new_conv
                
        self.nc += 1
        self.global_model.model.nc = self.nc
        if hasattr(head, 'nc'): head.nc = self.nc
        if hasattr(head, 'no'): head.no += 1

    def merge_client(self, client_weights_path, class_name):
        print(f"\n--- Processing Incoming Client: '{class_name}' ---")
        
        # --- FIRST CLIENT INITIALIZATION ---
        if self.global_model is None:
            print(f"First client detected. Initializing Global Model from '{class_name}'...")
            client_model = YOLO(client_weights_path)
            client_sd = client_model.model.state_dict()
            base_model = YOLO("yolo26n.pt")
            base_sd = base_model.model.state_dict()
            
            for key in client_sd.keys():
                if any(x in key for x in ['cv2', 'dfl']):
                    if base_sd[key].shape == client_sd[key].shape:
                        client_sd[key] = base_sd[key]
                        
            client_model.model.load_state_dict(client_sd)
            self.global_model = client_model
            self.nc = 1
            self.registry[class_name] = 0
            self.global_model.model.names = {0: class_name}
            self._save_model()
            return

        # --- DYNAMIC EXPANSION ---
        if class_name not in self.registry:
            self.registry[class_name] = self.nc
            self.global_model.model.names[self.nc] = class_name
            self._expand_classification_head()
            print(f"Discovered new class '{class_name}'. Assigned Global ID: {self.registry[class_name]}")

        target_id = self.registry[class_name]
        print(f"Merging Client's Local ID 0 into Server's Global ID {target_id} using strategy: {self.strategy.upper()}...")
        
        global_sd = self.global_model.model.state_dict()
        client_sd = YOLO(client_weights_path).model.state_dict()
        
        current_num_classes = len(self.registry)
        dynamic_alpha = 1.0 / current_num_classes

        # ========================================================
        # STRATEGY 1: TIES-MERGING (Task Vector)
        # ========================================================
        if self.strategy == 'ties':
            for key in global_sd.keys():
                if key in client_sd:
                    if any(x in key for x in ['cv3', 'one2one_cv3']) and ('weight' in key or 'bias' in key) and global_sd[key].shape[0] == self.nc and client_sd[key].shape[0] == 1:
                        global_sd[key][target_id] = client_sd[key][0].clone()
                    elif global_sd[key].shape == client_sd[key].shape:
                        if any(x in key for x in ['cv2', 'dfl']):
                            continue 
                        else:
                            task_vector = client_sd[key] - global_sd[key]
                            num_elements = task_vector.numel()
                            if num_elements > 100:
                                keep_ratio = 0.30
                                k_index = int(num_elements * (1 - keep_ratio))
                                abs_vector = torch.abs(task_vector)
                                threshold = torch.kthvalue(abs_vector.flatten(), k_index).values
                                mask = abs_vector >= threshold
                                trimmed_vector = task_vector * mask
                            else:
                                trimmed_vector = task_vector
                            global_sd[key] = global_sd[key] + (dynamic_alpha * trimmed_vector)

        # ========================================================
        # STRATEGY 2: STANDARD FEDAVG (No Trimming)
        # ========================================================
        elif self.strategy == 'fedavg':
            for key in global_sd.keys():
                if key in client_sd:
                    if any(x in key for x in ['cv3', 'one2one_cv3']) and ('weight' in key or 'bias' in key) and global_sd[key].shape[0] == self.nc and client_sd[key].shape[0] == 1:
                        global_sd[key][target_id] = client_sd[key][0].clone()
                    elif global_sd[key].shape == client_sd[key].shape:
                        if any(x in key for x in ['cv2', 'dfl']):
                            continue 
                        else:
                            # Direct mathematical blending
                            global_sd[key] = (1 - dynamic_alpha) * global_sd[key] + dynamic_alpha * client_sd[key]

        # ========================================================
        # STRATEGY 3: DATA-FREE KNOWLEDGE DISTILLATION (DFKD)
        # ========================================================
        elif self.strategy == 'dfkd':
            device = next(self.global_model.model.parameters()).device
            client_model = YOLO(client_weights_path)
            
            # Step 1: Pure Injection for Output Nodes
            for key in global_sd.keys():
                if key in client_sd:
                    if any(x in key for x in ['cv3', 'one2one_cv3']) and ('weight' in key or 'bias' in key) and global_sd[key].shape[0] == self.nc and client_sd[key].shape[0] == 1:
                        global_sd[key][target_id] = client_sd[key][0].clone()
            
            self.global_model.model.load_state_dict(global_sd)
            
            print("\n[DFKD] Initiating Model Inversion and Hallucination phase...")
            
            client_model.model.eval()
            for param in client_model.model.parameters(): param.requires_grad = False
            
            global_teacher = copy.deepcopy(self.global_model.model).eval()
            for param in global_teacher.parameters(): param.requires_grad = False
            
            # ====================================================
            # THE WIRETAP (PyTorch Hooks)
            # We intercept the raw Feature Pyramids right before they enter the final Head.
            # ====================================================
            client_features = []
            global_features = []
            student_features = []

            def hook_client(module, input, output):
                client_features.clear()
                x = input[0] if isinstance(input[0], (list, tuple)) else [input[0]]
                client_features.extend(x)

            def hook_global(module, input, output):
                global_features.clear()
                x = input[0] if isinstance(input[0], (list, tuple)) else [input[0]]
                global_features.extend(x)

            def hook_student(module, input, output):
                student_features.clear()
                x = input[0] if isinstance(input[0], (list, tuple)) else [input[0]]
                student_features.extend(x)

            # Attach wiretaps to the final 'Detect' layer (model[-1]) of all 3 models
            h1 = client_model.model.model[-1].register_forward_hook(hook_client)
            h2 = global_teacher.model[-1].register_forward_hook(hook_global)
            h3 = self.global_model.model.model[-1].register_forward_hook(hook_student)
            # ====================================================

            # Generate Dummy Data (Noise Tensor)
            dummy_images = torch.randn(4, 3, 640, 640, device=device, requires_grad=True)
            optimizer_noise = torch.optim.Adam([dummy_images], lr=0.1)
            
            # DeepDream: Train the noise to maximize teacher activations
            for i in range(20):
                optimizer_noise.zero_grad()
                _ = client_model.model(dummy_images) # The hook silently captures the features!
                
                # Maximize the activations of the captured feature maps
                loss_noise = -sum(f.mean() for f in client_features)
                loss_noise.backward()
                optimizer_noise.step()
                
            print("[DFKD] Distilling Teacher Knowledge into Global Student...")
            self.global_model.model.train()
            
            # Protect BatchNorm layers from getting corrupted by the noise
            for m in self.global_model.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            
            optimizer_student = torch.optim.Adam(self.global_model.model.parameters(), lr=0.0005)
            criterion = nn.MSELoss()
            
            # Distillation Loop
            for i in range(50):
                optimizer_student.zero_grad()
                
                # Get Teacher predictions (No grad)
                with torch.no_grad():
                    _ = client_model.model(dummy_images)
                    _ = global_teacher(dummy_images)

                # Get Student prediction
                _ = self.global_model.model(dummy_images)
                
                # Align the student's wiretapped features to the teachers' wiretapped features
                loss = 0
                for s_feat, tc_feat, tg_feat in zip(student_features, client_features, global_features):
                    loss += criterion(s_feat, tc_feat) * dynamic_alpha + criterion(s_feat, tg_feat) * (1 - dynamic_alpha)
                
                loss.backward()
                optimizer_student.step()
                
            # Remove the wiretaps to clean up memory
            h1.remove()
            h2.remove()
            h3.remove()
            
            # Grab the distilled weights
            global_sd = self.global_model.model.state_dict()

        # ========================================================
        # STRATEGY 4: HEAD-STITCHING (Zero-Interference)
        # ========================================================
        elif self.strategy == 'stitch':
            for key in global_sd.keys():
                if key in client_sd:
                    # 1. 100% Pure Injection for the specific class output nodes
                    if any(x in key for x in ['cv3', 'one2one_cv3']) and ('weight' in key or 'bias' in key) and global_sd[key].shape[0] == self.nc and client_sd[key].shape[0] == 1:
                        global_sd[key][target_id] = client_sd[key][0].clone()
                        
                    # 2. DO ABSOLUTELY NOTHING ELSE! 
                    # We skip all hidden layers to maintain the pristine, shared pre-trained backbone.

        # Save Final State
        self.global_model.model.load_state_dict(global_sd)
        self._save_model()

    def _save_model(self):
        out_path = os.path.join(GLOBAL_MODEL_DIR, "global_model.pt")
        self.global_model.save(out_path)
        print(f"✅ Global model updated and saved: {out_path}")
        print(f"Current Global Classes: {self.registry}")

def run_server():
    print("==================================================")
    print(" 🏆 DAFYOLO FEDERATED LEARNING SERVER ENGINE 🏆 ")
    print("==================================================")
    print("Select Server Merging Strategy:")
    print("  [1] TIES-Merging (Task Vector Trimming) [Stable]")
    print("  [2] Standard FedAvg (Mathematical 50/50 blend)")
    print("  [3] Data-Free Distillation (Model Inversion) [Experimental!]")
    print("  [4] Head-Stitching (Zero-Interference / Frozen Backbone) [NEW]")
    print("==================================================")
    
    choice = input("Enter your choice (1/2/3/4): ").strip()
    strategy_map = {'1': 'ties', '2': 'fedavg', '3': 'dfkd', '4': 'stitch'}
    selected_strategy = strategy_map.get(choice, 'stitch') # Default to stitch if invalid
    
    print(f"\nBooting Server with [{selected_strategy.upper()}] strategy...")
    
    server = FLServer(strategy=selected_strategy)
    print(f"\nServer listening for SSH uploads in {UPLOAD_DIR}...")
    
    while True:
        # HOT-RELOAD WATCHDOG
        global_model_path = os.path.join(GLOBAL_MODEL_DIR, "global_model.pt")
        if server.global_model is not None and not os.path.exists(global_model_path):
            print("\n[WATCHDOG] 🚨 Global model file was deleted from disk!")
            print(f"[WATCHDOG] Wiping RAM and rebuilding from vault using {selected_strategy.upper()}...")
            server.global_model = None
            server.registry = {}
            server.nc = 0
            server.bootstrap_server()
            
        files = os.listdir(UPLOAD_DIR)
        meta_files = [f for f in files if f.endswith('_meta.json')]
        
        if meta_files:
            print(f"\n[DEBUG] 👀 Found incoming files: {meta_files}")
            
        for meta_file in meta_files:
            meta_path = os.path.join(UPLOAD_DIR, meta_file)
            
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
            except json.JSONDecodeError:
                print(f"[DEBUG] ⚠️ JSON not finished writing yet: {meta_file}. Skipping this tick.")
                continue
                
            client_id = meta.get('client_id')
            class_name = meta.get('class_name')
            
            if not client_id or not class_name:
                continue

            weights_file = f"{client_id}_weights.pt"
            weights_path = os.path.join(UPLOAD_DIR, weights_file)
            
            if os.path.exists(weights_path):
                print(f"[DEBUG] 🚀 Weights found for '{class_name}'! Triggering merge...")
                
                # 1. Merge the model
                server.merge_client(weights_path, class_name)
                
                # 2. Archive the files
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archived_meta = os.path.join(PROCESSED_DIR, f"{client_id}_{timestamp}_meta.json")
                archived_weights = os.path.join(PROCESSED_DIR, f"{client_id}_{timestamp}_weights.pt")
                
                shutil.move(meta_path, archived_meta)
                shutil.move(weights_path, archived_weights)
                
                print(f"📦 Archived {client_id} to the vault!\n")
                
        time.sleep(5)

if __name__ == "__main__":
    run_server()