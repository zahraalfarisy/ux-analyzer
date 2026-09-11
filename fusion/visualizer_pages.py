import time
import threading
import math
import base64
import io
import cv2
import mss
import mss.tools
from PIL import Image
from eyetrax import GazeEstimator
from pynput import mouse as pynput_mouse
from collections import defaultdict
import win32gui
import win32process
import psutil
import numpy as np
from scipy.ndimage import gaussian_filter



# ── storage ───────────────────────────────────────────────────────────────
gaze_data = []
mouse_data = []
page_snapshots = []  # [{url, title, screenshot_b64, start_ts, end_ts}]
running = True
current_page = {"title": "", "screenshot": None, "start_ts": None}

# ── screen capture ─────────────────────────────────────────────────────────
def capture_screen_b64():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = sct.grab(monitor)
        pil = Image.frombytes("RGB", img.size, img.rgb)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode(), img.size

def get_active_window_title():
    try:
        hwnd = win32gui.GetForegroundWindow()
        return win32gui.GetWindowText(hwnd)
    except:
        return ""

# ── page tracker thread ────────────────────────────────────────────────────
def page_tracker_thread():
    last_title = ""
    while running:
        title = get_active_window_title()
        if title != last_title and "Chrome" in title:
            ts = time.time()
            screenshot_b64, size = capture_screen_b64()

            # simpan snapshot halaman sebelumnya
            if current_page["title"]:
                page_snapshots.append({
                    "title": current_page["title"],
                    "screenshot_b64": current_page["screenshot"],
                    "screen_size": current_page["screen_size"],
                    "start_ts": current_page["start_ts"],
                    "end_ts": ts,
                    "gaze": [],
                    "mouse": [],
                    "events": []
                })

            current_page["title"] = title
            current_page["screenshot"] = screenshot_b64
            current_page["screen_size"] = size
            current_page["start_ts"] = ts
            last_title = title
            print(f"[page] {title[:60]}")

        time.sleep(0.5)

# ── gaze thread ────────────────────────────────────────────────────────────
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
    for c in [e for e in mouse if e["subtype"] == "click"]:
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

def assign_to_pages(snapshots, gaze_all, mouse_all):
    for page in snapshots:
        s, e = page["start_ts"], page["end_ts"]
        page["gaze"]   = [g for g in gaze_all  if s <= g["timestamp"] <= e]
        page["mouse"]  = [m for m in mouse_all if s <= m["timestamp"] <= e]
        page["events"] = sorted(
            detect_rage_click(page["mouse"]) +
            detect_gaze_mouse_mismatch(page["gaze"], page["mouse"]),
            key=lambda x: x["timestamp"]
        )
    return snapshots

def generate_gaze_heatmap(gaze_data, screen_w, screen_h):
    """Generate smooth Gaussian heatmap, return base64 PNG"""
    canvas = np.zeros((screen_h, screen_w), dtype=np.float32)
    
    for g in gaze_data:
        x = int(max(0, min(g["gaze_x"], screen_w - 1)))
        y = int(max(0, min(g["gaze_y"], screen_h - 1)))
        canvas[y, x] += 1

    # Gaussian blur — sigma makin besar makin smooth
    canvas = gaussian_filter(canvas, sigma=40)
    
    if canvas.max() > 0:
        canvas = canvas / canvas.max()

    # colormap: transparan → hijau → kuning → merah
    rgba = np.zeros((screen_h, screen_w, 4), dtype=np.uint8)
    
    # merah
    rgba[:,:,0] = (np.clip(canvas * 2, 0, 1) * 255).astype(np.uint8)
    # hijau
    rgba[:,:,1] = (np.clip(2 - canvas * 2, 0, 1) * 255).astype(np.uint8)
    # biru
    rgba[:,:,2] = 0
    # alpha — area kosong transparan
    rgba[:,:,3] = (np.clip(canvas * 1.5, 0, 0.85) * 255).astype(np.uint8)

    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def generate_click_heatmap(mouse_data, screen_w, screen_h):
    canvas = np.zeros((screen_h, screen_w), dtype=np.float32)
    
    clicks = [m for m in mouse_data if m["subtype"] == "click"]
    for c in clicks:
        x = int(max(0, min(c["x"], screen_w - 1)))
        y = int(max(0, min(c["y"], screen_h - 1)))
        canvas[y, x] += 1

    canvas = gaussian_filter(canvas, sigma=25)
    
    if canvas.max() > 0:
        canvas = canvas / canvas.max()

    rgba = np.zeros((screen_h, screen_w, 4), dtype=np.uint8)
    rgba[:,:,0] = 0
    rgba[:,:,1] = (np.clip(canvas * 2, 0, 1) * 255).astype(np.uint8)
    rgba[:,:,2] = 255
    rgba[:,:,3] = (np.clip(canvas * 1.5, 0, 0.85) * 255).astype(np.uint8)

    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ── HTML generator ─────────────────────────────────────────────────────────
def generate_html(snapshots):
    tabs_html = ""
    panels_html = ""

    for i, page in enumerate(snapshots):
        sw, sh = page["screen_size"]
        moves  = [m for m in page["mouse"] if m["subtype"] == "move"]
        clicks = [m for m in page["mouse"] if m["subtype"] == "click"]
        rage_events     = [e for e in page["events"] if e["type"] == "rage_click"]
        mismatch_events = [e for e in page["events"] if e["type"] == "gaze_mouse_mismatch"]

        # gaze heatmap
        cell = 40
        gaze_grid = defaultdict(int)
        for g in page["gaze"]:
            gaze_grid[(int(g["gaze_x"]//cell)*cell, int(g["gaze_y"]//cell)*cell)] += 1
        max_c = max(gaze_grid.values(), default=1)

        try:
            heatmap_b64 = generate_gaze_heatmap(page["gaze"], sw, sh)
        except Exception as e:
            print(f"[warn] gaze heatmap gagal: {e}")
            heatmap_b64 = None

        try:
            click_heatmap_b64 = generate_click_heatmap(page["mouse"], sw, sh)
        except Exception as e:
            print(f"[warn] click heatmap gagal: {e}")
            click_heatmap_b64 = None

        # mouse path
        path_pts = " ".join(f"{m['x']},{m['y']}" for m in moves[:800])

        gaze_img = f'<image href="data:image/png;base64,{heatmap_b64}" width="{sw}" height="{sh}"/>' if heatmap_b64 else ""
        click_img = f'<image href="data:image/png;base64,{click_heatmap_b64}" width="{sw}" height="{sh}"/>' if click_heatmap_b64 else ""

        # clicks
        click_svg = ""
        for c in clicks:
            is_rage = any(abs(r["x"]-c["x"])<20 and abs(r["y"]-c["y"])<20 for r in rage_events)
            col = "#ef4444" if is_rage else "#f59e0b"
            click_svg += f'<circle cx="{c["x"]}" cy="{c["y"]}" r="12" fill="{col}" fill-opacity="0.75" stroke="white" stroke-width="2"/>\n'

        # mismatch lines
        mismatch_svg = ""
        for m in mismatch_events:
            mx, my = m["mouse_pos"]
            gx, gy = m["gaze_pos"]
            mismatch_svg += f'<line x1="{mx}" y1="{my}" x2="{gx}" y2="{gy}" stroke="#a855f7" stroke-width="2" stroke-opacity="0.7" stroke-dasharray="6,3"/>\n'
            mismatch_svg += f'<circle cx="{gx}" cy="{gy}" r="7" fill="#a855f7" fill-opacity="0.6"/>\n'

        # event rows
        event_rows = ""
        start_ts = snapshots[0]["start_ts"]
        for ev in page["events"]:
            rel = round(ev["timestamp"] - start_ts, 2)
            if ev["type"] == "rage_click":
                badge = '<span class="badge red">rage click</span>'
                detail = f'({ev["x"]}, {ev["y"]}) — {ev["count"]}x klik'
            else:
                badge = '<span class="badge purple">gaze mismatch</span>'
                detail = f'mouse {ev["mouse_pos"]} ↔ gaze {ev["gaze_pos"]} — {ev["distance_px"]}px'
            event_rows += f'<tr><td>+{rel}s</td><td>{badge}</td><td>{detail}</td></tr>'

        short_title = page["title"][:40]
        active = "active" if i == 0 else ""
        tabs_html += f'<button class="tab {active}" onclick="showPage({i})">{short_title}</button>'

        panels_html += f'''
<div class="panel" id="panel-{i}" style="display:{'block' if i==0 else 'none'}">
  <div class="stats">
    <div class="stat red"><div class="val">{len(rage_events)}</div><div class="lbl">Rage click</div></div>
    <div class="stat purple"><div class="val">{len(mismatch_events)}</div><div class="lbl">Gaze mismatch</div></div>
    <div class="stat amber"><div class="val">{len(clicks)}</div><div class="lbl">Klik</div></div>
    <div class="stat teal"><div class="val">{len(page["gaze"])}</div><div class="lbl">Gaze samples</div></div>
  </div>
  <div class="map-wrap">
    <svg viewBox="0 0 {sw} {sh}" xmlns="http://www.w3.org/2000/svg">
  <image href="data:image/jpeg;base64,{page['screenshot_b64']}" width="{sw}" height="{sh}"/>
  {gaze_img}
  {click_img}
  <polyline points="{path_pts}" fill="none" stroke="#ffffff" stroke-width="1.5" stroke-opacity="0.3"/>
  {mismatch_svg}
  {click_svg}
</svg>
  </div>
  <div class="legend">
    <span><span class="dot blue"></span>Mouse path</span>
    <span><span class="dot amber"></span>Klik</span>
    <span><span class="dot red"></span>Gaze heatmap</span>
    <span><span class="dot" style="background:#00ffff"></span>Click heatmap</span>
    <span><span class="dot red"></span>Rage click</span>
    <span><span class="dot purple"></span>Gaze mismatch</span>
    <span style="display:flex;align-items:center;gap:4px"><span style="width:28px;height:8px;background:linear-gradient(to right,#32cd6480,#ff000080);border-radius:2px"></span>Gaze intensity</span>
  </div>
  <h2 style="margin-top:20px">Friction Events</h2>
  <table>
    <thead><tr><th>Waktu</th><th>Tipe</th><th>Detail</th></tr></thead>
    <tbody>{event_rows or '<tr><td colspan="3" class="empty">Tidak ada friction events</td></tr>'}</tbody>
  </table>
</div>'''

    html = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<title>UX Analyzer — Per Page Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0f1117;color:#e5e7eb;font-family:'Segoe UI',sans-serif;padding-bottom:40px}}
  header{{padding:20px 32px;border-bottom:1px solid #1f2937}}
  header h1{{font-size:18px;font-weight:600;color:white}}
  header p{{font-size:13px;color:#6b7280;margin-top:2px}}
  .tabs{{display:flex;gap:8px;padding:16px 32px;flex-wrap:wrap;border-bottom:1px solid #1f2937}}
  .tab{{padding:6px 14px;border-radius:99px;font-size:12px;cursor:pointer;border:1px solid #374151;color:#9ca3af;background:transparent;max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
  .tab.active{{background:#3b82f6;border-color:#3b82f6;color:white}}
  .panel{{padding:20px 32px}}
  .stats{{display:flex;gap:12px;margin-bottom:16px}}
  .stat{{background:#1f2937;border-radius:10px;padding:14px 18px;flex:1}}
  .val{{font-size:26px;font-weight:700;color:white}}
  .lbl{{font-size:12px;color:#6b7280;margin-top:2px}}
  .stat.red .val{{color:#ef4444}}.stat.purple .val{{color:#a855f7}}
  .stat.amber .val{{color:#f59e0b}}.stat.teal .val{{color:#14b8a6}}
  .map-wrap{{background:#111827;border-radius:12px;overflow:hidden;border:1px solid #1f2937;margin-bottom:10px}}
  .map-wrap svg{{display:block;width:100%}}
  .legend{{display:flex;gap:16px;font-size:12px;color:#6b7280;align-items:center;margin-bottom:20px}}
  .dot{{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:4px}}
  .dot.blue{{background:#3b82f6}}.dot.amber{{background:#f59e0b}}
  .dot.red{{background:#ef4444}}.dot.purple{{background:#a855f7}}
  h2{{font-size:13px;font-weight:600;color:#9ca3af;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px}}
  table{{width:100%;border-collapse:collapse;background:#1f2937;border-radius:10px;overflow:hidden}}
  th{{padding:10px 12px;font-size:12px;color:#6b7280;text-align:left;background:#111827;font-weight:600;text-transform:uppercase}}
  td{{padding:8px 12px;font-size:13px;color:#e5e7eb}}
  tr:hover td{{background:#374151}}
  .empty{{text-align:center;color:#6b7280;padding:16px!important}}
  .badge{{padding:2px 10px;border-radius:99px;font-size:11px;color:white}}
  .badge.red{{background:#ef4444}}.badge.purple{{background:#a855f7}}
</style>
</head>
<body>
<header>
  <h1>UX Analyzer — Per Page Report</h1>
  <p>{len(snapshots)} halaman direkam</p>
</header>
<div class="tabs">{tabs_html}</div>
{panels_html}
<script>
function showPage(i) {{
  document.querySelectorAll('.panel').forEach((p,j) => p.style.display = j===i?'block':'none');
  document.querySelectorAll('.tab').forEach((t,j) => t.classList.toggle('active', j===i));
}}
</script>
</body>
</html>"""

    with open("fusion_report_pages.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Report saved → fusion_report_pages.html")

# ── main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Sesi dimulai. Buka Chrome, browsing, tekan Ctrl+C untuk stop.")

    t_gaze = threading.Thread(target=gaze_thread, daemon=True)
    t_page = threading.Thread(target=page_tracker_thread, daemon=True)
    t_gaze.start()
    t_page.start()

    listener = pynput_mouse.Listener(on_move=on_move, on_click=on_click)
    listener.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        running = False
        listener.stop()

        # flush halaman terakhir
        if current_page["title"]:
            page_snapshots.append({
                "title": current_page["title"],
                "screenshot_b64": current_page["screenshot"],
                "screen_size": current_page.get("screen_size", (1920, 1080)),
                "start_ts": current_page["start_ts"],
                "end_ts": time.time(),
                "gaze": [], "mouse": [], "events": []
            })

        if not page_snapshots:
            print("Tidak ada halaman yang terekam. Pastikan Chrome aktif saat sesi.")
        else:
            snapshots = assign_to_pages(page_snapshots, gaze_data, mouse_data)
            generate_html(snapshots)
            print("Buka fusion_report_pages.html di browser.")