import time
import csv
import threading
import cv2
from eyetrax import GazeEstimator
from pynput import mouse as pynput_mouse

# ── storage ──────────────────────────────────────────────────────────────────
gaze_data = []   # [{"timestamp", "gaze_x", "gaze_y"}]
mouse_data = []  # [{"timestamp", "x", "y", "subtype"}]
running = True

# ── gaze thread ──────────────────────────────────────────────────────────────
def gaze_thread():
    estimator = GazeEstimator()
    try:
        estimator.load_model("../eyetrax/gaze_model.pkl")
        print("[gaze] Model loaded.")
    except:
        from eyetrax import run_9_point_calibration
        run_9_point_calibration(estimator)
        estimator.save_model("../eyetrax/gaze_model.pkl")

    cap = cv2.VideoCapture(0)
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        features, blink = estimator.extract_features(frame)
        if features is not None and not blink:
            x, y = estimator.predict([features])[0]
            gaze_data.append({"timestamp": time.time(), "gaze_x": x, "gaze_y": y})
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

# ── mouse listener ────────────────────────────────────────────────────────────
def on_move(x, y):
    mouse_data.append({"timestamp": time.time(), "x": x, "y": y, "subtype": "move"})

def on_click(x, y, button, pressed):
    if pressed:
        mouse_data.append({"timestamp": time.time(), "x": x, "y": y, "subtype": "click"})

# ── fusion ────────────────────────────────────────────────────────────────────
def detect_rage_click(data, window_sec=1.0, threshold=3):
    clicks = [e for e in data if e["subtype"] == "click"]
    events = []
    used = set()
    for i, c in enumerate(clicks):
        if i in used:
            continue
        window = [d for d in clicks if c["timestamp"] <= d["timestamp"] <= c["timestamp"] + window_sec]
        if len(window) >= threshold:
            for j, d in enumerate(clicks):
                if d in window:
                    used.add(j)
            events.append({"type": "rage_click", "timestamp": c["timestamp"], "x": c["x"], "y": c["y"], "count": len(window)})
    return events

def detect_gaze_mouse_mismatch(gaze, mouse, tolerance_px=150, tolerance_sec=0.5):
    events = []
    clicks = [e for e in mouse if e["subtype"] == "click"]
    for c in clicks:
        nearby = [g for g in gaze if abs(g["timestamp"] - c["timestamp"]) <= tolerance_sec]
        if not nearby:
            continue
        g = nearby[0]
        dist = ((g["gaze_x"] - c["x"])**2 + (g["gaze_y"] - c["y"])**2)**0.5
        if dist > tolerance_px:
            events.append({"type": "gaze_mouse_mismatch", "timestamp": c["timestamp"],
                           "mouse_pos": (c["x"], c["y"]), "gaze_pos": (round(g["gaze_x"]), round(g["gaze_y"])),
                           "distance_px": round(dist)})
    return events

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Sesi dimulai. Tekan Ctrl+C untuk stop dan lihat hasil.")

    t = threading.Thread(target=gaze_thread, daemon=True)
    t.start()

    listener = pynput_mouse.Listener(on_move=on_move, on_click=on_click)
    listener.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        running = False
        listener.stop()
        print("\n── Hasil Fusion ──────────────────────────────")
        
        rage = detect_rage_click(mouse_data)
        mismatch = detect_gaze_mouse_mismatch(gaze_data, mouse_data)
        all_events = sorted(rage + mismatch, key=lambda e: e["timestamp"])
        
        if all_events:
            for e in all_events:
                print(e)
        else:
            print("Tidak ada friction events terdeteksi.")
        
        print(f"\nTotal gaze samples: {len(gaze_data)}")
        print(f"Total klik: {len([e for e in mouse_data if e['subtype'] == 'click'])}")