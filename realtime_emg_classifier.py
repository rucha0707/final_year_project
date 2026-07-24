import serial
import numpy as np
import pickle
import joblib
from collections import deque
import time
import sys
import warnings

# =========================
# BASIC SETTINGS
# =========================
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')

# =========================
# LOAD MODEL + SCALER
# =========================
print("Loading model...")

# FIXED: Use joblib to load the model to avoid the _pickle.UnpicklingError
model = joblib.load("emg_rf_model123.pkl")

# FIXED: Filename must match "emg_scaler (1).pkl" exactly
scaler = joblib.load("emg_scaler123.pkl")

print("Model loaded")
print("Model expects features:", model.n_features_in_)

# =========================
# SERIAL CONNECTION
# =========================
print("Connecting to Arduino...")

ser = serial.Serial('COM3', 115200, timeout=1)

time.sleep(2)

print("Connected")

# =========================
# WINDOW PARAMETERS
# =========================
WINDOW_SIZE = 500
OVERLAP = 250

# Scaling factors
V1_SCALE = 0.00025
V2_SCALE = 0.00025

# Sampling frequency
FS = 2000

# =========================
# SIGNAL BUFFERS
# =========================
buffer_ch1 = deque(maxlen=WINDOW_SIZE)
buffer_ch2 = deque(maxlen=WINDOW_SIZE)

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(window):

    arr = np.array(window, dtype=float)

    # Time-domain features
    mav  = np.mean(np.abs(arr))
    rms  = np.sqrt(np.mean(arr**2))
    var  = np.var(arr)
    std  = np.std(arr)

    wl   = np.sum(np.abs(np.diff(arr)))

    zc   = np.sum(((arr[:-1] * arr[1:]) < 0).astype(int))

    ssc  = np.sum(
        ((np.diff(arr[:-1]) * np.diff(arr[1:])) < 0).astype(int)
    )

    iemg = np.sum(np.abs(arr))

    maxv = np.max(arr)
    minv = np.min(arr)

    # Skewness + Kurtosis
    if std < 1e-8:
        skew = 0
        kurt = 0
    else:
        skew = (
            np.mean((arr - np.mean(arr))**3)
        ) / (std**3)

        kurt = (
            np.mean((arr - np.mean(arr))**4)
        ) / (std**4)

    # =========================
    # Frequency-domain features
    # =========================
    from scipy.fft import rfft, rfftfreq

    N = len(arr)

    freqs = rfftfreq(N, 1 / FS)

    fft_vals = np.abs(rfft(arr))

    power = np.sum(fft_vals**2) / N

    pkf = freqs[np.argmax(fft_vals)]

    mnf = np.sum(freqs * fft_vals) / (
        np.sum(fft_vals) + 1e-8
    )

    cumsum = np.cumsum(fft_vals)

    mdf = freqs[
        np.where(cumsum >= cumsum[-1] / 2)[0][0]
    ]

    return [
        mav, rms, var, std,
        wl, zc, ssc, iemg,
        maxv, minv, skew, kurt,
        mnf, mdf, pkf, power
    ]

# =========================
# START REALTIME CLASSIFIER
# =========================
print("Starting real-time EMG classification...")

while True:

    # =========================
    # READ SERIAL DATA
    # =========================
    try:
        line = ser.readline().decode().strip()

        if not line:
            continue

        if ',' not in line:
            continue

        v1_str, v2_str = line.split(',')

        v1 = float(v1_str) * V1_SCALE
        v2 = float(v2_str) * V2_SCALE

    except:
        continue

    # =========================
    # STORE IN BUFFER
    # =========================
    buffer_ch1.append(v1)
    buffer_ch2.append(v2)

    # Wait until full window collected
    if len(buffer_ch1) < WINDOW_SIZE:
        continue

    # =========================
    # FEATURE EXTRACTION
    # =========================
    feats1 = extract_features(buffer_ch1)
    feats2 = extract_features(buffer_ch2)

    # Combine features
    feature_vector = np.array(
        feats1 + feats2
    ).reshape(1, -1)

    # =========================
    # SCALE FEATURES
    # =========================
    feature_vector = scaler.transform(feature_vector)

    # =========================
    # MODEL PREDICTION
    # =========================
    pred = model.predict(feature_vector)[0]

    # =========================
    # DEBUG OUTPUT
    # =========================
    print(
        f"MAV1={feats1[0]:.6f}, "
        f"MAV2={feats2[0]:.6f}, "
        f"Pred={pred}"
    )

    # =========================
    # HUMAN-READABLE OUTPUT + SERIAL WRITE
    # =========================
    if pred == 0:
        print("Prediction -> Hand Close")
        ser.write(b'0') # NEW: Send Close command
    else:
        print("Prediction -> Hand Open")
        ser.write(b'1') # NEW: Send Open command

    # =========================
    # SLIDING WINDOW
    # =========================
    for _ in range(OVERLAP):
        if len(buffer_ch1) > 0:
            buffer_ch1.popleft()
        if len(buffer_ch2) > 0:
            buffer_ch2.popleft()