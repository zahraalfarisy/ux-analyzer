import json
import time
from pathlib import Path

from config import OUTPUT_DIR

def fuse(*event_lists):
    merged = []
    for lst in event_lists:
        merged.extend(lst)
    return sorted(merged, key=lambda e: e['ts'])

def save_session(session_id, events):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f'{session_id}.json'
    with open(path, 'w') as f:
        json.dump({
            'session_id': session_id,
            'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'total_events': len(events),
            'events': events,
        }, f, indent=2)
    return path

def load_session(session_id):
    path = OUTPUT_DIR / f'{session_id}.json'
    with open(path) as f:
        return json.load(f)
