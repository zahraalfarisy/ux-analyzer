"""
Friction Detector
Mengambil semua events dari semua sinyal (sudah di-sort by ts),
lalu mendeteksi momen friction point berdasarkan collision antar sinyal.

Friction score 0.0–1.0, semakin tinggi semakin parah.
"""
import time
from config import FRICTION_WEIGHTS


def compute_friction(events: list[dict], window: float = 1.5) -> list[dict]:
    """
    events  : list event dari semua modul, sudah di-sort by 'ts'
    window  : rentang waktu (detik) untuk menganggap sinyal bersamaan

    Return  : list friction events, format Event dengan type="friction"
    """
    frictions = []
    click_events = [e for e in events
                    if e["type"] == "mouse" and e["meta"].get("subtype") == "click"]

    for click in click_events:
        ts = click["ts"]

        # Kumpulkan sinyal dalam window di sekitar klik ini
        nearby = [e for e in events
                  if abs(e["ts"] - ts) <= window and e is not click]

        signals_fired = []
        score = 0.0

        # Sinyal dari mouse itu sendiri
        if click["meta"].get("rage_click"):
            signals_fired.append("rage_click")
            score += FRICTION_WEIGHTS["rage_click"]

        if click["meta"].get("hesitated"):
            signals_fired.append("hesitation")
            score += FRICTION_WEIGHTS["hesitation"]

        # Erratic movement sebelum klik
        erratic_moves = [e for e in nearby
                         if e["type"] == "mouse"
                         and e["meta"].get("subtype") == "move"
                         and e["meta"].get("erratic")]
        if erratic_moves:
            signals_fired.append("erratic")
            score += FRICTION_WEIGHTS["erratic"]

        # FER: frustrasi / bingung di sekitar klik
        fer_nearby = [e for e in nearby if e["type"] == "fer"
                      and e["meta"].get("emotion") in ("angry", "fear", "disgust")]
        if fer_nearby:
            signals_fired.append("frustrated")
            score += FRICTION_WEIGHTS["frustrated"]

        # EyeTrax: gaze jauh dari area klik
        gaze_nearby = [e for e in nearby if e["type"] == "gaze"]
        if gaze_nearby:
            avg_gaze_x = sum(g["x"] for g in gaze_nearby) / len(gaze_nearby)
            avg_gaze_y = sum(g["y"] for g in gaze_nearby) / len(gaze_nearby)
            import math
            gaze_dist = math.hypot(avg_gaze_x - click["x"],
                                   avg_gaze_y - click["y"])
            if gaze_dist > 200:  # px
                signals_fired.append("gaze_off")
                score += FRICTION_WEIGHTS["gaze_off"]

        # Simpan hanya jika ada minimal satu sinyal
        if signals_fired:
            frictions.append({
                "type":       "friction",
                "ts":         ts,
                "session_id": click["session_id"],
                "x":          click["x"],
                "y":          click["y"],
                "score":      round(min(score, 1.0), 3),
                "meta": {
                    "signals": signals_fired,
                    "click_hesitation_s": click["meta"].get("hesitation_s", 0),
                }
            })

    return sorted(frictions, key=lambda e: e["score"], reverse=True)
