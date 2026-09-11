import subprocess
import threading
from fusion.fusion import run_fusion

def run_eyetrax():
    subprocess.run(["python", "eyetrax/src/eyetrax/app/demo.py"])

def run_mousetracker():
    subprocess.run(["python", "mousetracker/app.py"])  # sesuaikan path

# Jalanin paralel
t1 = threading.Thread(target=run_eyetrax)
t2 = threading.Thread(target=run_mousetracker)
t1.start(); t2.start()
t1.join(); t2.join()

# Setelah sesi selesai, fuse hasilnya
events = run_fusion("data/mouse_session.json", "data/gaze_log.csv")
for e in events:
    print(e)