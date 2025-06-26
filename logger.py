import os
from datetime import datetime

def log_event(tag, message):
    os.makedirs("logs", exist_ok=True)
    with open("logs/prediction_log.txt", "a") as f:
        f.write(f"[{datetime.now()}] [{tag}] {message}\n")