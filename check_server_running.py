#!/usr/bin/env python3
"""Check if server_updated.py is running on the server"""

import os
import paramiko
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(env_path)

from client_updated import SERVER_IP, SSH_PORT, SSH_USER, SSH_PASSWORD

print("🔍 Checking if server_updated.py is running...")

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(
    hostname=SERVER_IP, 
    port=SSH_PORT, 
    username=SSH_USER, 
    password=SSH_PASSWORD,
    look_for_keys=False, 
    allow_agent=False
)

# Check for server_updated.py process
stdin, stdout, stderr = ssh.exec_command("ps aux | grep server_updated.py | grep -v grep")
output = stdout.read().decode().strip()

if output:
    print("✅ server_updated.py IS RUNNING on the server!")
    print("\nProcess info:")
    print(output)
else:
    print("❌ server_updated.py is NOT RUNNING on the server!")
    print("\n⚠️  You need to start it with:")
    print("   cd /datadrive/DAFYOLO")
    print("   python server_updated.py")
    print("\n   Or in background:")
    print("   nohup python server_updated.py > server.log 2>&1 &")

# Check the server log if available
print("\n📋 Recent server activity:")
stdin, stdout, stderr = ssh.exec_command("tail -20 /datadrive/DAFYOLO/server.log 2>/dev/null || echo 'No log file yet'")
log_output = stdout.read().decode()
print(log_output)

ssh.close()
