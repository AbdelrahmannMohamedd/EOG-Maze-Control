import os
import sys
import pickle
import numpy as np
import pygame
import tkinter as tk
from tkinter import filedialog
import pywt
from scipy.integrate import simpson
from statsmodels.tsa.ar_model import AutoReg
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# 1.  LOAD THE PAIRED PICKLE
# ──────────────────────────────────────────────────────────
PAIRED_MODEL_PATH = "best_paired_model.pkl"

if not os.path.exists(PAIRED_MODEL_PATH):
    sys.exit(f"[ERROR] '{PAIRED_MODEL_PATH}' not found. "
             "Run the training script first.")

with open(PAIRED_MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model        = bundle["model"]
scaler       = bundle["scaler"]
CLASS_NAMES  = bundle.get("class_names",  ["Up", "Down", "Right", "Left", "Blink"])
FEATURE_NAMES= bundle.get("feature_names",
                          [
                                'H Mean', 'H Std Dev', 'H Variance' ,
                                'H AUC', 'H Max' , 'H Min',
                                'H AR Coeff 1', 'H AR Coeff 2', 'H AR Coeff 3',
                                'V Mean', 'V Std Dev', 'V Variance' ,
                                'V AUC', 'V Max' , 'V Min',
                                'V AR Coeff 1', 'V AR Coeff 2', 'V AR Coeff 3',
                            ])

FS           = bundle.get("fs", 176)
label        = bundle.get("label", "Paired model")
accuracy     = bundle.get("accuracy", None)

acc_str = f"{accuracy:.1f}%" if accuracy is not None else "N/A"
print(f"[INFO] Loaded paired model : {label}")
print(f"[INFO] Test accuracy       : {acc_str}")
print(f"[INFO] Classes             : {CLASS_NAMES}")
print(f"[INFO] Feature count       : {len(FEATURE_NAMES) if FEATURE_NAMES else 'unknown'}")

# ──────────────────────────────────────────────────────────
# 2.  PREPROCESSING & FEATURE EXTRACTION  (mirrors training)
# ──────────────────────────────────────────────────────────
def preprocess_signal(signal: np.ndarray) -> np.ndarray:
    coeffs   = pywt.wavedec(signal, "db4", level=2)
    filtered = pywt.waverec([coeffs[0]], "db4")
    return filtered


def extract_hybrid_features(signal: np.ndarray) -> np.ndarray:
    features = []
    # Statistical & morphological
    features.append(np.mean(signal))
    features.append(np.std(signal))
    features.append(np.var(signal))
    features.append(simpson(y=signal))      # AUC
    features.append(np.max(signal))
    features.append(np.min(signal))
    # Auto-Regression lags 1-3
    try:
        ar_model = AutoReg(signal, lags=3).fit()
        c = ar_model.params
        features.extend([c[1], c[2], c[3]])
    except Exception:
        features.extend([0.0, 0.0, 0.0])
    return np.array(features)


def load_and_predict_paired(h_path: str, v_path: str):
    with open(h_path, "r") as f:
        h_raw = [float(l.strip()) for l in f
                 if l.strip().replace(".", "", 1).replace("-", "", 1).isdigit()]
    with open(v_path, "r") as f:
        v_raw = [float(l.strip()) for l in f
                 if l.strip().replace(".", "", 1).replace("-", "", 1).isdigit()]

    h_clean = preprocess_signal(np.array(h_raw))
    v_clean = preprocess_signal(np.array(v_raw))

    feats = np.concatenate([
        extract_hybrid_features(h_clean),
        extract_hybrid_features(v_clean)
    ]).reshape(1, -1)

    feats_scaled = scaler.transform(feats)
    pred_idx     = model.predict(feats_scaled)[0]
    return CLASS_NAMES[pred_idx], feats_scaled

# ──────────────────────────────────────────────────────────
# 3.  MAZE DEFINITION
# ──────────────────────────────────────────────────────────
MAZE_MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 2, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# ──────────────────────────────────────────────────────────
# 4.  PYGAME MAZE (PAIRED MODE)
# ──────────────────────────────────────────────────────────
def run_maze(classifier, scaler):
    pygame.init()
    pygame.font.init()

    CELL_SIZE = 50
    ROWS = len(MAZE_MAP)
    COLS = len(MAZE_MAP[0])
    screen = pygame.display.set_mode((COLS * CELL_SIZE, ROWS * CELL_SIZE + 50))
    pygame.display.set_caption("EOG Maze - Press ' 1 ' to Load H Signal, ' 2 ' to Load V Signal")
    font = pygame.font.SysFont('Arial', 20)

    player_pos = [1, 1]
    status_msg = "Load H signal (' 1 ') then V signal (' 2 ') to move"
    pending_h = None

    running = True
    while running:
        screen.fill((20, 20, 20))
        for r in range(ROWS):
            for c in range(COLS):
                rect = (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if MAZE_MAP[r][c] == 1:
                    pygame.draw.rect(screen, (100, 100, 100), rect)
                    pygame.draw.rect(screen, (50, 50, 50), rect, 2)
                elif MAZE_MAP[r][c] == 2:
                    pygame.draw.rect(screen, (0, 255, 0), rect)

        px = player_pos[1] * CELL_SIZE + (CELL_SIZE // 4)
        py = player_pos[0] * CELL_SIZE + (CELL_SIZE // 4)
        pygame.draw.rect(screen, (0, 150, 255), (px, py, CELL_SIZE//2, CELL_SIZE//2))

        text_surf = font.render(status_msg, True, (255, 255, 255))
        screen.blit(text_surf, (10, ROWS * CELL_SIZE + 10))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    root = tk.Tk()
                    root.withdraw()
                    file_path = filedialog.askopenfilename(title="Select H (Horizontal) EOG Signal", filetypes=[("Text", "*.txt")])
                    root.destroy()
                    if file_path:
                        try:
                            with open(file_path, 'r') as f:
                                raw_data = [float(line.strip()) for line in f if line.strip().replace('.','',1).replace('-','',1).isdigit()]
                            pending_h = preprocess_signal(np.array(raw_data))
                            status_msg = "H loaded. Now press ' 2 ' to load V signal."
                        except Exception as e:
                            status_msg = f"Error loading H: {e}"

                elif event.key == pygame.K_2:
                    if pending_h is None:
                        status_msg = "Load H signal first with ' 1 '."
                    else:
                        root = tk.Tk()
                        root.withdraw()
                        file_path = filedialog.askopenfilename(title="Select V (Vertical) EOG Signal", filetypes=[("Text", "*.txt")])
                        root.destroy()
                        if file_path:
                            try:
                                with open(file_path, 'r') as f:
                                    raw_data = [float(line.strip()) for line in f if line.strip().replace('.','',1).replace('-','',1).isdigit()]

                                clean_v = preprocess_signal(np.array(raw_data))
                                feats = np.concatenate([
                                    extract_hybrid_features(pending_h),
                                    extract_hybrid_features(clean_v)
                                ]).reshape(1, -1)
                                feats_scaled = scaler.transform(feats)

                                pred = classifier.predict(feats_scaled)[0]
                                cmd = CLASS_NAMES[pred]
                                pending_h = None
                                status_msg = f"Predicted: {cmd}. Moving..."

                                new_r, new_c = player_pos[0], player_pos[1]
                                if cmd == "Up": new_r -= 1
                                elif cmd == "Down": new_r += 1
                                elif cmd == "Right": new_c += 1
                                elif cmd == "Left": new_c -= 1

                                if MAZE_MAP[new_r][new_c] != 1:
                                    player_pos = [new_r, new_c]
                                    if MAZE_MAP[new_r][new_c] == 2:
                                        status_msg = "GOAL REACHED! Press 'R' to reset."
                                else:
                                    status_msg = f"Predicted {cmd}, but blocked by wall!"
                            except Exception as e:
                                status_msg = f"Error loading V: {e}"

                elif event.key == pygame.K_r:
                    player_pos = [1, 1]
                    pending_h = None
                    status_msg = "Load H signal (' 1 ') then V signal (' 2 ') to move"

        pygame.display.flip()
    pygame.quit()


# ──────────────────────────────────────────────────────────
# 5.  ENTRY POINT
# ──────────────────────────────────────────────────────────
print("\n[INFO] Starting Paired EOG Maze...")
print("       Controls:  '1' = Load H file | '2' = Load V file | 'R' = Reset\n")
run_maze(model  , scaler)