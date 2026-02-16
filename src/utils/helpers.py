import subprocess
import psutil
import os

def get_system_health():
    """Fetches VRAM, RAM, and CPU health for the ROG Strix G16."""
    health = {"gpu": "N/A", "vram_used": 0, "vram_total": 12227, "ram_percent": 0, "status": "Unknown"}
    
    try:
        # 1. GPU / VRAM Check (Parsing nvidia-smi)
        cmd = "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,nounits,noheader"
        output = subprocess.check_output(cmd.split()).decode('utf-8').strip().split(',')
        health["vram_used"] = int(output[0])
        health["vram_total"] = int(output[1])
        health["gpu_util"] = f"{output[2]}%"
        
        # 2. System RAM Check
        ram = psutil.virtual_memory()
        health["ram_percent"] = ram.percent
        health["ram_used_gb"] = round(ram.used / (1024**3), 2)
        
        # 3. Status Logic
        if health["vram_used"] > 10000:
            health["status"] = "⚠️ Heavy Load (VRAM)"
        else:
            health["status"] = "✅ Healthy"
            
    except Exception:
        health["status"] = "❌ Hardware Monitor Error"
        
    return health
