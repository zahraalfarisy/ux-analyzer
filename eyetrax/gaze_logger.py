from eyetrax import GazeEstimator, run_9_point_calibration
import cv2
import csv
import time

estimator = GazeEstimator()

# Kalau udah punya model, load aja. Kalau belum, kalibrasi dulu
try:
    estimator.load_model("gaze_model.pkl")
    print("Model loaded.")
except:
    run_9_point_calibration(estimator)
    estimator.save_model("gaze_model.pkl")

cap = cv2.VideoCapture(0)

with open("gaze_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "gaze_x", "gaze_y"])

    print("Recording gaze... tekan ESC untuk berhenti.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        features, blink = estimator.extract_features(frame)
        if features is not None and not blink:
            x, y = estimator.predict([features])[0]
            writer.writerow([time.time(), round(x), round(y)])

        if cv2.waitKey(1) == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
print("Saved to gaze_log.csv")