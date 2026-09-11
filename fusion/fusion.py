import json
import pandas as pd

def load_mouse(path):
    with open(path) as f:
        data = json.load(f)
    rows = []
    for e in data["events"]:
        rows.append({
            "timestamp": e["ts"],
            "x": e["x"],
            "y": e["y"],
            "subtype": e["meta"]["subtype"],
            "speed": e["meta"].get("speed_px_s", 0),
            "erratic": e["meta"].get("erratic", False)
        })
    return pd.DataFrame(rows)

def load_gaze(path):
    df = pd.read_csv(path, names=["timestamp", "gaze_x", "gaze_y"])
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df

def detect_rage_click(mouse_df, window_sec=1.0, threshold=3):
    clicks = mouse_df[mouse_df["subtype"] == "click"].sort_values("timestamp")
    events = []
    used = set()
    for i, row in clicks.iterrows():
        if i in used:
            continue
        window = clicks[
            (clicks["timestamp"] >= row["timestamp"]) &
            (clicks["timestamp"] <= row["timestamp"] + window_sec)
        ]
        if len(window) >= threshold:
            for idx in window.index:
                used.add(idx)
            events.append({
                "timestamp": row["timestamp"],
                "type": "rage_click",
                "x": row["x"], "y": row["y"],
                "count": len(window)
            })
    return events

def detect_gaze_mouse_mismatch(gaze_df, mouse_df, tolerance_px=150, tolerance_sec=0.5):
    """Gaze jauh dari mouse saat klik = user lihat tempat lain, confusion signal"""
    events = []
    clicks = mouse_df[mouse_df["subtype"] == "click"]
    for _, m_row in clicks.iterrows():
        nearby = gaze_df[abs(gaze_df["timestamp"] - m_row["timestamp"]) <= tolerance_sec]
        if nearby.empty:
            continue
        g = nearby.iloc[0]
        dist = ((g["gaze_x"] - m_row["x"])**2 + (g["gaze_y"] - m_row["y"])**2)**0.5
        if dist > tolerance_px:
            events.append({
                "timestamp": m_row["timestamp"],
                "type": "gaze_mouse_mismatch",
                "mouse_x": m_row["x"], "mouse_y": m_row["y"],
                "gaze_x": g["gaze_x"], "gaze_y": g["gaze_y"],
                "distance_px": round(dist, 1)
            })
    return events

def run_fusion(mouse_path, gaze_path):
    mouse_df = load_mouse(mouse_path)
    gaze_df = load_gaze(gaze_path)

    rage = detect_rage_click(mouse_df)
    mismatch = detect_gaze_mouse_mismatch(gaze_df, mouse_df)

    all_events = sorted(rage + mismatch, key=lambda e: e["timestamp"])
    return all_events

# Test
if __name__ == "__main__":
    events = run_fusion(
        mouse_path="../mouse-tracker/output/sessions/22a9f508.json",
        gaze_path="../eyetrax/gaze_log.csv"
    )
    for e in events:
        print(e)