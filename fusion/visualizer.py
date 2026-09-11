import json
import time
import threading
import cv2
import math
from eyetrax import GazeEstimator
from pynput import mouse as pynput_mouse
from collections import defaultdict

# ── storage ───────────────────────────────────────────────────────────────
gaze_data = []
mouse_data = []
running = True

# ── gaze thread ───────────────────────────────────────────────────────────
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

# ── mouse listener ─────────────────────────────────────────────────────────
def on_move(x, y):
    mouse_data.append({"timestamp": time.time(), "x": x, "y": y, "subtype": "move"})

def on_click(x, y, button, pressed):
    if pressed:
        mouse_data.append({"timestamp": time.time(), "x": x, "y": y, "subtype": "click"})

# ── fusion ─────────────────────────────────────────────────────────────────
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
            events.append({"type": "rage_click", "timestamp": c["timestamp"],
                           "x": c["x"], "y": c["y"], "count": len(window)})
    return events

def detect_gaze_mouse_mismatch(gaze, mouse, tolerance_px=150, tolerance_sec=0.5):
    events = []
    clicks = [e for e in mouse if e["subtype"] == "click"]
    for c in clicks:
        nearby = [g for g in gaze if abs(g["timestamp"] - c["timestamp"]) <= tolerance_sec]
        if not nearby:
            continue
        g = nearby[0]
        dist = math.sqrt((g["gaze_x"] - c["x"])**2 + (g["gaze_y"] - c["y"])**2)
        if dist > tolerance_px:
            events.append({"type": "gaze_mouse_mismatch", "timestamp": c["timestamp"],
                           "mouse_pos": (c["x"], c["y"]),
                           "gaze_pos": (round(g["gaze_x"]), round(g["gaze_y"])),
                           "distance_px": round(dist)})
    return events

# ── HTML output ────────────────────────────────────────────────────────────
def generate_html(gaze_data, mouse_data, events, screen_w=1920, screen_h=1080):
    clicks = [e for e in mouse_data if e["subtype"] == "click"]
    moves  = [e for e in mouse_data if e["subtype"] == "move"]
    rage_events    = [e for e in events if e["type"] == "rage_click"]
    mismatch_events = [e for e in events if e["type"] == "gaze_mouse_mismatch"]

    # mouse path polyline
    path_points = " ".join(f"{e['x']},{e['y']}" for e in moves[:500])

    # click markers
    click_markers = ""
    for c in clicks:
        is_rage = any(
            abs(r["x"] - c["x"]) < 10 and abs(r["y"] - c["y"]) < 10
            for r in rage_events
        )
        color = "#ef4444" if is_rage else "#f59e0b"
        click_markers += f'<circle cx="{c["x"]}" cy="{c["y"]}" r="10" fill="{color}" fill-opacity="0.7" stroke="white" stroke-width="1.5"/>\n'

    # gaze heatmap cells (grid 40x40)
    cell = 40
    gaze_grid = defaultdict(int)
    for g in gaze_data:
        gx = int(g["gaze_x"] // cell) * cell
        gy = int(g["gaze_y"] // cell) * cell
        gaze_grid[(gx, gy)] += 1

    max_count = max(gaze_grid.values(), default=1)
    gaze_rects = ""
    for (gx, gy), count in gaze_grid.items():
        intensity = count / max_count
        r = int(255 * min(intensity * 2, 1))
        g_val = int(255 * max(0, 1 - intensity * 2))
        gaze_rects += f'<rect x="{gx}" y="{gy}" width="{cell}" height="{cell}" fill="rgb({r},{g_val},50)" fill-opacity="{0.15 + intensity * 0.55}"/>\n'

    # mismatch lines
    mismatch_lines = ""
    for m in mismatch_events:
        mx, my = m["mouse_pos"]
        gx, gy = m["gaze_pos"]
        mismatch_lines += f'<line x1="{mx}" y1="{my}" x2="{gx}" y2="{gy}" stroke="#a855f7" stroke-width="1.5" stroke-opacity="0.6" stroke-dasharray="5,3"/>\n'
        mismatch_lines += f'<circle cx="{gx}" cy="{gy}" r="6" fill="#a855f7" fill-opacity="0.5"/>\n'

    # friction event rows
    event_rows = ""
    all_events_sorted = sorted(events, key=lambda e: e["timestamp"])
    start_ts = all_events_sorted[0]["timestamp"] if all_events_sorted else 0
    for e in all_events_sorted:
        rel = round(e["timestamp"] - start_ts, 2)
        if e["type"] == "rage_click":
            badge = f'<span style="background:#ef4444;color:white;padding:2px 8px;border-radius:99px;font-size:12px">rage click</span>'
            detail = f'({e["x"]}, {e["y"]}) — {e["count"]}x klik'
        else:
            badge = f'<span style="background:#a855f7;color:white;padding:2px 8px;border-radius:99px;font-size:12px">gaze mismatch</span>'
            detail = f'mouse {e["mouse_pos"]} ↔ gaze {e["gaze_pos"]} — {e["distance_px"]}px'
        event_rows += f'''
        <tr>
          <td style="padding:8px 12px;color:#9ca3af">+{rel}s</td>
          <td style="padding:8px 12px">{badge}</td>
          <td style="padding:8px 12px;color:#e5e7eb;font-size:13px">{detail}</td>
        </tr>'''

    html = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<title>UX Analyzer — Fusion Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0f1117; color: #e5e7eb; font-family: 'Segoe UI', sans-serif; }}
  header {{ padding: 24px 32px; border-bottom: 1px solid #1f2937; display: flex; align-items: center; gap: 16px; }}
  header h1 {{ font-size: 20px; font-weight: 600; color: white; }}
  header span {{ font-size: 13px; color: #6b7280; }}
  .stats {{ display: flex; gap: 16px; padding: 20px 32px; }}
  .stat {{ background: #1f2937; border-radius: 10px; padding: 16px 20px; flex: 1; }}
  .stat .val {{ font-size: 28px; font-weight: 700; color: white; }}
  .stat .label {{ font-size: 12px; color: #6b7280; margin-top: 2px; }}
  .stat.red .val {{ color: #ef4444; }}
  .stat.purple .val {{ color: #a855f7; }}
  .stat.amber .val {{ color: #f59e0b; }}
  .stat.teal .val {{ color: #14b8a6; }}
  section {{ padding: 0 32px 28px; }}
  h2 {{ font-size: 14px; font-weight: 600; color: #9ca3af; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 12px; }}
  .map-wrap {{ position: relative; background: #111827; border-radius: 12px; overflow: hidden; border: 1px solid #1f2937; }}
  .map-wrap svg {{ display: block; width: 100%; }}
  .legend {{ display: flex; gap: 20px; margin-top: 10px; font-size: 12px; color: #6b7280; align-items: center; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 4px; }}
  table {{ width: 100%; border-collapse: collapse; background: #1f2937; border-radius: 10px; overflow: hidden; }}
  thead tr {{ background: #111827; }}
  thead td {{ padding: 10px 12px; font-size: 12px; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }}
  tbody tr:hover {{ background: #374151; }}
  .tabs {{ display: flex; gap: 8px; margin-bottom: 14px; }}
  .tab {{ padding: 6px 16px; border-radius: 99px; font-size: 13px; cursor: pointer; border: 1px solid #374151; color: #9ca3af; background: transparent; }}
  .tab.active {{ background: #3b82f6; border-color: #3b82f6; color: white; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>UX Analyzer — Fusion Report</h1>
    <span>{len(gaze_data)} gaze samples &nbsp;·&nbsp; {len(clicks)} klik &nbsp;·&nbsp; {len(events)} friction events</span>
  </div>
</header>

<div class="stats">
  <div class="stat red">
    <div class="val">{len(rage_events)}</div>
    <div class="label">Rage click</div>
  </div>
  <div class="stat purple">
    <div class="val">{len(mismatch_events)}</div>
    <div class="label">Gaze mismatch</div>
  </div>
  <div class="stat amber">
    <div class="val">{len(clicks)}</div>
    <div class="label">Total klik</div>
  </div>
  <div class="stat teal">
    <div class="val">{len(gaze_data)}</div>
    <div class="label">Gaze samples</div>
  </div>
</div>

<section>
  <h2>Heatmap &amp; Mouse Path</h2>
  <div class="map-wrap">
    <svg viewBox="0 0 {screen_w} {screen_h}" xmlns="http://www.w3.org/2000/svg">
      <rect width="{screen_w}" height="{screen_h}" fill="#0f1117"/>
      <!-- gaze heatmap -->
      {gaze_rects}
      <!-- mouse path -->
      <polyline points="{path_points}" fill="none" stroke="#3b82f6" stroke-width="1.5" stroke-opacity="0.4"/>
      <!-- mismatch lines -->
      {mismatch_lines}
      <!-- clicks -->
      {click_markers}
    </svg>
  </div>
  <div class="legend">
    <span><span class="dot" style="background:#3b82f6"></span>Mouse path</span>
    <span><span class="dot" style="background:#f59e0b"></span>Klik biasa</span>
    <span><span class="dot" style="background:#ef4444"></span>Rage click</span>
    <span><span class="dot" style="background:#a855f7"></span>Gaze mismatch</span>
    <span style="display:inline-flex;align-items:center;gap:4px">
      <span style="width:24px;height:8px;background:linear-gradient(to right,#32cd6480,#ff000080);border-radius:2px;display:inline-block"></span>Gaze intensity
    </span>
  </div>
</section>

<section>
  <h2>Friction Events</h2>
  <table>
    <thead><tr><td>Waktu</td><td>Tipe</td><td>Detail</td></tr></thead>
    <tbody>{event_rows if event_rows else '<tr><td colspan="3" style="padding:16px;text-align:center;color:#6b7280">Tidak ada friction events</td></tr>'}</tbody>
  </table>
</section>
</body>
</html>"""

    with open("fusion_report.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Report saved → fusion_report.html")

# ── main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Sesi dimulai. Tekan Ctrl+C untuk stop.")

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

        rage     = detect_rage_click(mouse_data)
        mismatch = detect_gaze_mouse_mismatch(gaze_data, mouse_data)
        all_events = sorted(rage + mismatch, key=lambda e: e["timestamp"])

        print(f"\nFriction events: {len(all_events)}")
        for e in all_events:
            print(e)

        generate_html(gaze_data, mouse_data, all_events)
        print("Buka fusion_report.html di browser.")