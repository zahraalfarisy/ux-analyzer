import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output" / "sessions"
REPORT_DIR = BASE_DIR / "output" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds untuk friction detection
RAGE_CLICK_RADIUS   = 50    # px, radius area rage click
RAGE_CLICK_WINDOW   = 2.0   # detik
RAGE_CLICK_MIN      = 3     # jumlah klik minimum
HESITATION_PAUSE    = 2.0   # detik diam sebelum klik
ERRATIC_SPEED_PX_S  = 3000  # px/s, threshold gerakan erratik

# Bobot sinyal untuk friction score (total = 1.0)
FRICTION_WEIGHTS = {
    "rage_click":  0.35,
    "hesitation":  0.25,
    "erratic":     0.15,
    "frustrated":  0.15,   # dari FER
    "gaze_off":    0.10,   # dari EyeTrax
}
