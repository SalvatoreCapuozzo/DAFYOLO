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
BASE_DIR = "/datadrive/DAFYOLO"
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
GLOBAL_MODEL_DIR = os.path.join(BASE_DIR, "global_model")
ARCHIVE_DIR = os.path.join(GLOBAL_MODEL_DIR, "archive")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_models")

for d in [UPLOAD_DIR, GLOBAL_MODEL_DIR, ARCHIVE_DIR, PROCESSED_DIR]:
    os.makedirs(d, exist_ok=True)


class FLServer:
    def __init__(self, strategy='fedhead'):
        self.global_model = None
        self.registry = {}
        self.nc = 0
        self.strategy = strategy
        self.merge_counts = {}
        self._write_server_info()
        self.bootstrap_server()

    def _write_server_info(self):
        """Broadcasts the active strategy so clients can auto-configure themselves."""
        info = {
            "strategy": self.strategy,
            "boot_time": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        info_path = os.path.join(GLOBAL_MODEL_DIR, "server_info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f)
        print(f"📡 Broadcasted server info: {info}")

    def bootstrap_server(self):
        global_model_path = os.path.join(GLOBAL_MODEL_DIR, "global_model.pt")

        if os.path.exists(global_model_path):
            print(f"Loading existing global model from {global_model_path}...")
            self.global_model = YOLO(global_model_path)
            self.registry = {v: k for k, v in self.global_model.model.names.items()}
            self.nc = len(self.registry)
            print(f"Restored Global Classes: {self.registry}")
        else:
            processed_metas = sorted(
                [f for f in os.listdir(PROCESSED_DIR) if f.endswith('_meta.json')],
                key=lambda x: os.path.getmtime(os.path.join(PROCESSED_DIR, x))
            )
            if processed_metas:
                print(f"Rebuilding from {len(processed_metas)} archived updates...")
                for meta_file in processed_metas:
                    try:
                        with open(os.path.join(PROCESSED_DIR, meta_file), 'r') as f:
                            meta = json.load(f)
                    except json.JSONDecodeError:
                        continue
                    class_name = meta.get('class_name')
                    weights_file = meta_file.replace('_meta.json', '_weights.pt')
                    weights_path = os.path.join(PROCESSED_DIR, weights_file)
                    if class_name and os.path.exists(weights_path):
                        self.merge_client(weights_path, class_name)
            else:
                print(f"Server Initialized. Waiting for first client...")

    def reset_session(self):
        """Archives the current session and resets the server to a blank state."""
        global_model_path = os.path.join(GLOBAL_MODEL_DIR, "global_model.pt")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Archive the existing model
        if os.path.exists(global_model_path):
            archive_path = os.path.join(ARCHIVE_DIR, f"global_model_archived_{ts}.pt")
            shutil.move(global_model_path, archive_path)
            print(f"📦 Previous global model safely archived to {archive_path}")

        # 2. FIX: Archive the processed client files so they don't corrupt a future reboot
        processed_files = os.listdir(PROCESSED_DIR)
        if processed_files:
            session_archive_dir = os.path.join(ARCHIVE_DIR, f"session_{ts}")
            os.makedirs(session_archive_dir, exist_ok=True)
            for file in processed_files:
                shutil.move(os.path.join(PROCESSED_DIR, file), os.path.join(session_archive_dir, file))
            print(f"🧹 Swept {len(processed_files)} old client updates into {session_archive_dir}")

        # 3. Wipe internal memory
        self.global_model = None
        self.registry = {}
        self.nc = 0
        self.merge_counts = {}
        
        # 4. Broadcast the new session
        self._write_server_info()
        print("✨ Server memory wiped. Ready for a brand new Federated Learning session!")

    def _expand_classification_head(self):
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

    def _init_from_first_client(self, client_weights_path, class_name):
        client_model = YOLO(client_weights_path)
        base_model   = YOLO("yolo26n.pt")
        client_sd = client_model.model.state_dict()
        base_sd   = base_model.model.state_dict()

        detect_idx    = len(client_model.model.model) - 1
        detect_prefix = f'model.{detect_idx}.'

        merged_sd = {}
        for key, client_tensor in client_sd.items():
            if key.startswith(detect_prefix):
                merged_sd[key] = client_tensor.clone()
            elif key in base_sd and base_sd[key].shape == client_tensor.shape:
                merged_sd[key] = base_sd[key].clone()
            else:
                merged_sd[key] = client_tensor.clone()

        client_model.model.load_state_dict(merged_sd, strict=True)
        self.global_model = client_model
        self.nc = 1
        self.registry[class_name] = 0
        self.global_model.model.names = {0: class_name}
        self._save_model()

    def merge_client(self, client_weights_path, class_name):
        print(f"\n--- Processing client: '{class_name}' [{self.strategy.upper()}] ---")
        if self.global_model is None:
            self._init_from_first_client(client_weights_path, class_name)
            return

        if class_name not in self.registry:
            self.registry[class_name] = self.nc
            self.global_model.model.names[self.nc] = class_name
            self._expand_classification_head()

        target_id = self.registry[class_name]
        global_sd = self.global_model.model.state_dict()
        client_sd = YOLO(client_weights_path).model.state_dict()
        detect_idx = len(self.global_model.model.model) - 1
        detect_prefix = f'model.{detect_idx}.'

        if self.strategy == 'fedhead':
            self._merge_fedhead(global_sd, client_sd, target_id, detect_prefix)
        elif self.strategy == 'stitch':
            self._merge_stitch(global_sd, client_sd, target_id)
        elif self.strategy == 'ties':
            self._merge_ties(global_sd, client_sd, target_id, detect_prefix)
        elif self.strategy == 'fedavg':
            self._merge_fedavg(global_sd, client_sd, target_id, detect_prefix)

        self.global_model.model.load_state_dict(global_sd)
        self._save_model()

    def _merge_fedhead(self, global_sd, client_sd, target_id, detect_prefix):
        skip_keys = {'dfl', 'stride', 'anchors'}
        for key in global_sd.keys():
            if key not in client_sd or not key.startswith(detect_prefix) or any(s in key for s in skip_keys):
                continue
            is_cv3_branch = any(x in key for x in ['cv3.', 'one2one_cv3.'])
            if (is_cv3_branch and ('2.weight' in key or '2.bias' in key) and
                    global_sd[key].shape[0] == self.nc and client_sd[key].shape[0] == 1):
                global_sd[key][target_id] = client_sd[key][0].clone()

    def _merge_stitch(self, global_sd, client_sd, target_id):
        for key in global_sd.keys():
            if key in client_sd and (any(x in key for x in ['cv3.', 'one2one_cv3.']) and
                    ('2.weight' in key or '2.bias' in key) and
                    global_sd[key].shape[0] == self.nc and client_sd[key].shape[0] == 1):
                global_sd[key][target_id] = client_sd[key][0].clone()

    def _merge_ties(self, global_sd, client_sd, target_id, detect_prefix):
        base_model = YOLO("yolo26n.pt")
        base_sd = base_model.model.state_dict()
        alpha = 1.0 / len(self.registry)

        for key in global_sd.keys():
            if key not in client_sd:
                continue
            if (any(x in key for x in ['cv3.', 'one2one_cv3.']) and
                    ('2.weight' in key or '2.bias' in key) and
                    global_sd[key].shape[0] == self.nc and client_sd[key].shape[0] == 1):
                global_sd[key][target_id] = client_sd[key][0].clone()
            elif global_sd[key].shape == client_sd[key].shape:
                if any(x in key for x in ['cv2', 'dfl', 'anchors', 'stride']):
                    continue
                if key in base_sd:
                    task_vector = client_sd[key].float() - base_sd[key].float()
                    if task_vector.numel() > 100:
                        threshold = torch.kthvalue(task_vector.abs().flatten(), max(1, int(task_vector.numel() * 0.70))).values
                        task_vector = task_vector * (task_vector.abs() >= threshold)
                    global_sd[key] = (global_sd[key].float() + alpha * task_vector).to(global_sd[key].dtype)

    def _merge_fedavg(self, global_sd, client_sd, target_id, detect_prefix):
        alpha = 1.0 / len(self.registry)
        for key in global_sd.keys():
            if key not in client_sd: continue
            if (any(x in key for x in ['cv3.', 'one2one_cv3.']) and
                    ('2.weight' in key or '2.bias' in key) and
                    global_sd[key].shape[0] == self.nc and client_sd[key].shape[0] == 1):
                global_sd[key][target_id] = client_sd[key][0].clone()
            elif global_sd[key].shape == client_sd[key].shape:
                if any(x in key for x in ['cv2', 'dfl']): continue
                global_sd[key] = ((1 - alpha) * global_sd[key].float() + alpha * client_sd[key].float()).to(global_sd[key].dtype)

    def _save_model(self):
        out_path = os.path.join(GLOBAL_MODEL_DIR, "global_model.pt")
        self.global_model.save(out_path)
        
        # Keep an archive history so we don't lose previous states
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = os.path.join(ARCHIVE_DIR, f"global_model_{ts}.pt")
        shutil.copy(out_path, archive_path)
        
        print(f"✅ Global model updated. Classes: {self.registry}")


def run_server():
    print("=" * 52)
    print("  🏆 DAFYOLO SERVER (v5 - Handshake & Hot Reset) 🏆")
    print("=" * 52)
    choice = input("Strategy: [1]FedHead [2]Stitch [3]TIES [4]FedAvg: ").strip()
    strategy_map = {'1': 'fedhead', '2': 'stitch', '3': 'ties', '4': 'fedavg'}
    strategy = strategy_map.get(choice, 'fedhead')

    print(f"\n🚀 Booting server with [{strategy.upper()}] strategy...\n")
    server = FLServer(strategy=strategy)
    print(f"👀 Watching {UPLOAD_DIR} for incoming uploads or commands...\n")

    while True:
        # --- 1. Hot Reset Listener (UPDATED FOR AUTOMATION) ---
        reset_cmd_path = os.path.join(UPLOAD_DIR, "CMD_RESET.json")
        if os.path.exists(reset_cmd_path):
            print("\n🔄 Received RESET command from client.")
            try:
                with open(reset_cmd_path, 'r') as f:
                    cmd_data = json.load(f)
                
                # Remote Strategy Switching
                if 'strategy' in cmd_data:
                    server.strategy = cmd_data['strategy']
                    print(f"🔀 Switched server strategy to: {server.strategy.upper()}")
            except Exception as e:
                print(f"⚠️ Error reading reset command: {e}")

            server.reset_session()
            try:
                os.remove(reset_cmd_path)
            except OSError:
                pass
            time.sleep(2)
            continue

        # --- 2. Manual Deletion Watchdog ---
        global_model_path = os.path.join(GLOBAL_MODEL_DIR, "global_model.pt")
        if server.global_model is not None and not os.path.exists(global_model_path):
            print("\n⚠️ [WATCHDOG] Global model deleted from disk — rebuilding from vault...")
            server.global_model = None
            server.registry = {}
            server.nc = 0
            server.merge_counts = {}
            server.bootstrap_server()

        # --- 3. Process Incoming Client Updates ---
        meta_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith('_meta.json')]
        for meta_file in meta_files:
            meta_path = os.path.join(UPLOAD_DIR, meta_file)
            try:
                with open(meta_path, 'r') as f: 
                    meta = json.load(f)
            except json.JSONDecodeError:
                continue # JSON is still writing, skip this tick

            client_id = meta.get('client_id')
            class_name = meta.get('class_name')
            if not client_id or not class_name:
                continue

            weights_path = os.path.join(UPLOAD_DIR, f"{client_id}_weights.pt")
            if not os.path.exists(weights_path):
                continue # Weights haven't finished uploading

            # Execute the merge
            server.merge_client(weights_path, class_name)

            # Safely archive the processed files
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.move(meta_path, os.path.join(PROCESSED_DIR, f"{client_id}_{ts}_meta.json"))
            shutil.move(weights_path, os.path.join(PROCESSED_DIR, f"{client_id}_{ts}_weights.pt"))
            print(f"📦 Archived {client_id}\n")

        time.sleep(3) # Short sleep to prevent CPU spiking

if __name__ == "__main__":
    run_server()