import sys
import time
import json

from modules.base import new_session_id, Event
from modules.mouse_tracker import MouseTracker
from analysis.friction_detector import compute_friction
from analysis.timestamp_fusion import fuse, save_session


def run_demo():
    sid = new_session_id()
    now = time.time()
    print(f"[DEMO] Session: {sid}")

    mouse_events = [
        Event.make("mouse", sid, x=400, y=300, subtype="move", speed_px_s=120.0, erratic=False).to_dict(),
        Event.make("mouse", sid, x=401, y=301, subtype="move", speed_px_s=3500.0, erratic=True).to_dict(),
        Event.make("mouse", sid, x=402, y=300, subtype="click", button="Button.left",
                   rage_click=True, hesitated=True, hesitation_s=2.5).to_dict(),
        Event.make("mouse", sid, x=402, y=300, subtype="click", button="Button.left",
                   rage_click=True, hesitated=False, hesitation_s=0.1).to_dict(),
    ]

    fer_events = [
        Event.make("fer", sid, emotion="angry", score=0.87).to_dict(),
    ]

    timeline  = fuse(mouse_events, fer_events)
    path      = save_session(sid, timeline)
    frictions = compute_friction(timeline)

    print(f"[DEMO] Session saved → {path}")
    print(f"[DEMO] Friction points: {len(frictions)}")
    for f in frictions:
        print(f"  score={f['score']:.2f}  signals={f['meta']['signals']}")


def run_live(duration: int = 30):
    sid     = new_session_id()
    tracker = MouseTracker(session_id=sid)
    print(f"[LIVE] Session: {sid} | Durasi: {duration}s")
    tracker.start()
    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        print("\n[LIVE] Dihentikan.")
    finally:
        tracker.stop()

    timeline  = fuse(tracker.events)
    path      = save_session(sid, timeline)
    frictions = compute_friction(timeline)
    print(f"[LIVE] Events: {len(timeline)} | Friction: {len(frictions)}")
    print(f"[LIVE] Saved → {path}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if mode == "live":
        run_live(int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    else:
        run_demo()