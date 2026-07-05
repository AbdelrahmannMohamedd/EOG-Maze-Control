import os
import numpy as np
import pygame
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import pywt
from scipy.integrate import simpson
from statsmodels.tsa.ar_model import AutoReg

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================
FS = 176
DATASET_PATH = "."
CLASS_NAMES = ["Up", "Down", "Right", "Left", "Blink"]

PAIRED_MODEL_PATH = "best_paired_model.pkl"   # best model trained on paired H+V files
SINGLE_MODEL_PATH = "best_single_model.pkl"   # best model trained on single files

def load_data_strict(folder_path):
    print("Scanning dataset using strict file filtering...")
    signals, labels = [], []
    label_map = {'up': 0, 'down': 1, 'right': 2, 'left': 3, 'blink': 4}

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return signals, np.array(labels)

    for cls, lbl in label_map.items():
        cls_dir = os.path.join(folder_path, cls.capitalize())
        if not os.path.exists(cls_dir):
            print(f"Warning: '{cls_dir}' not found, skipping.")
            continue

        h_files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith('h.txt'))
        count = 0
        for hf in h_files:
            vf = hf[:-5] + 'v.txt'
            if not os.path.exists(os.path.join(cls_dir, vf)):
                continue
            try:
                with open(os.path.join(cls_dir, hf), 'r') as f:
                    h_data = [float(l.strip()) for l in f if l.strip().replace('.','',1).replace('-','',1).isdigit()]
                with open(os.path.join(cls_dir, vf), 'r') as f:
                    v_data = [float(l.strip()) for l in f if l.strip().replace('.','',1).replace('-','',1).isdigit()]
                if len(h_data) > 10 and len(v_data) > 10:
                    signals.append((np.array(h_data), np.array(v_data)))
                    labels.append(lbl)
                    count += 1
            except Exception:
                pass
        print(f"[{cls.capitalize()}] loaded {count} pairs")
    return signals, np.array(labels)


def load_data_single(folder_path):
    print("\nScanning dataset for individual (unpaired) files...")
    signals, labels = [], []
    label_map = {'up': 0, 'down': 1, 'right': 2, 'left': 3, 'blink': 4}

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return signals, np.array(labels)

    for cls, lbl in label_map.items():
        cls_dir = os.path.join(folder_path, cls.capitalize())
        if not os.path.exists(cls_dir):
            print(f"Warning: '{cls_dir}' not found, skipping.")
            continue

        txt_files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith('.txt'))
        count = 0
        for tf in txt_files:
            try:
                with open(os.path.join(cls_dir, tf), 'r') as f:
                    data = [float(l.strip()) for l in f if l.strip().replace('.','',1).replace('-','',1).isdigit()]
                if len(data) > 10:
                    signals.append(np.array(data))
                    labels.append(lbl)
                    count += 1
            except Exception:
                pass
        print(f"[{cls.capitalize()}] loaded {count} single files")
    return signals, np.array(labels)

# ==========================================
# 2. LEAN EOG PREPROCESSING & FEATURES
# ==========================================
def preprocess_signal(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=2)
    filtered = pywt.waverec([coeffs[0]], 'db4')
    return filtered

def extract_hybrid_features(signal):
    features = []

    # 1. Statistical & Morphological
    features.append(np.mean(signal))
    features.append(np.std(signal))
    features.append(np.var(signal))

    features.append(simpson(y=signal)) # AUC
    features.append(np.max(signal))
    features.append(np.min(signal))

    # 3. Auto-Regression
    try:
        ar_model = AutoReg(signal, lags=3).fit()
        coeffs = ar_model.params
        features.extend([coeffs[1], coeffs[2], coeffs[3]])
    except Exception:
        features.extend([0, 0, 0])

    return np.array(features)

FEATURE_NAMES = [
    'H Mean', 'H Std Dev', 'H Variance' ,
    'H AUC', 'H Max' , 'H Min',
    'H AR Coeff 1', 'H AR Coeff 2', 'H AR Coeff 3',
    'V Mean', 'V Std Dev', 'V Variance' ,
    'V AUC', 'V Max' , 'V Min',
    'V AR Coeff 1', 'V AR Coeff 2', 'V AR Coeff 3',
]

FEATURE_NAMES_SINGLE = [
    'Mean', 'Std Dev', 'Variance',
    'AUC', 'Max', 'Min',
    'AR Coeff 1', 'AR Coeff 2', 'AR Coeff 3',
]

# ==========================================
# 3. PLOTTING FUNCTION
# ==========================================
def plot_analytics(model_names, accuracies, best_model, X_test, y_test, rf_model,
                   feature_names, title_suffix=""):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.canvas.manager.set_window_title(f'EOG Lean Models - ML Analytics {title_suffix}')

    # Plot 1: Accuracy Comparison
    sns.barplot(x=model_names, y=accuracies, ax=axes[0], palette="coolwarm")
    axes[0].set_title(f"Model Accuracy Showdown {title_suffix}", fontsize=14)
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

    # Plot 2: Confusion Matrix
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    axes[1].set_title(f"Confusion Matrix ({best_model.__class__.__name__})", fontsize=14)
    axes[1].set_ylabel("Actual Movement")
    axes[1].set_xlabel("Predicted Movement")

    # Plot 3: Feature Importance
    importances = rf_model.feature_importances_
    top_n = min(10, len(feature_names))
    indices = np.argsort(importances)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    axes[2].barh(range(top_n), importances[indices], color='gold')
    axes[2].set_yticks(range(top_n))
    axes[2].set_yticklabels(top_features)
    axes[2].set_title(f"Top {top_n} Most Useful Features", fontsize=14)

    plt.tight_layout()
    print("\n[INFO] Close the plotting window to continue...")
    plt.show()

# ==========================================
# 4. ML TRAINING (CORE BUILT-IN MODELS)
# ==========================================
def train_models(X, y, feature_names=None, title_suffix=""):
    if feature_names is None:
        feature_names = FEATURE_NAMES

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm_clf = SVC(probability=True, random_state=42)
    rf_clf = RandomForestClassifier(random_state=42)
    knn_clf = KNeighborsClassifier()
    nn_clf = MLPClassifier(random_state=42, max_iter=2000)

    param_grids = {
        'SVM': {'C': [0.1, 1, 10, 50, 100], 'gamma': ['scale', 'auto', 0.01, 0.1], 'kernel': ['linear', 'rbf']},
        'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
        'KNN': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']},
        'Neural Network': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.01]}
    }

    best_estimators = {}
    model_names = []
    accuracies = []

    print(f"\n--- Training Core Models {title_suffix} ---")
    print(f"{'Model':<20} | {'Train Acc':>10} | {'Val Acc (CV)':>13} | {'Test Acc':>10}")
    print("-" * 62)

    cv_folds = max(2, min(4, np.min(np.bincount(y_train))) if len(np.bincount(y_train)) > 1 else 2)

    classifiers = [
        ('SVM', svm_clf),
        ('Random Forest', rf_clf),
        ('KNN', knn_clf),
        ('Neural Network', nn_clf)
    ]

    for name, clf in classifiers:
        grid = GridSearchCV(clf, param_grids[name], cv=cv_folds, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        best_estimators[name] = grid.best_estimator_

        train_acc = accuracy_score(y_train, grid.predict(X_train_scaled)) * 100
        val_acc   = grid.best_score_ * 100
        test_acc  = accuracy_score(y_test, grid.predict(X_test_scaled)) * 100

        model_names.append(name)
        accuracies.append(test_acc)
        print(f"{name:<20} | {train_acc:>9.1f}% | {val_acc:>12.1f}% | {test_acc:>9.1f}%")

    best_acc_idx = np.argmax(accuracies)
    overall_best_model = best_estimators[model_names[best_acc_idx]]
    print(f"\n>>> CHAMPION: {model_names[best_acc_idx]} with {max(accuracies):.1f}% test accuracy <<<")

    plot_analytics(model_names, accuracies, overall_best_model, X_test_scaled, y_test,
                   best_estimators['Random Forest'], feature_names, title_suffix)

    return overall_best_model, scaler, model_names, accuracies, best_estimators

# ==========================================
# 5. FULL MAZE UI
# ==========================================
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
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

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

# ==========================================
# 6. MAIN: DUAL TRAINING + PICKLE SAVING
# ==========================================

signals, y_pairs = load_data_strict(DATASET_PATH)

signals_single, y_single = load_data_single(DATASET_PATH)

all_results = {}  # { model_label: (clf, scaler, test_acc) }

if len(signals) >= 4:
    print(f"\nSuccessfully loaded {len(signals)} H/V pairs.")
    X_pairs = np.array([
        np.concatenate([
            extract_hybrid_features(preprocess_signal(h)),
            extract_hybrid_features(preprocess_signal(v))
        ]) for h, v in signals
    ])
    best_clf_pairs, fitted_scaler_pairs, names_p, accs_p, estimators_p = train_models(
        X_pairs, y_pairs, feature_names=FEATURE_NAMES, title_suffix="[Paired H+V]"
    )
    for name, acc in zip(names_p, accs_p):
        all_results[f"{name} (Paired)"] = (estimators_p[name], fitted_scaler_pairs, acc)
else:
    print("Not enough paired H/V files found. Skipping paired training.")
    best_clf_pairs, fitted_scaler_pairs = None, None

if len(signals_single) >= 4:
    print(f"\nSuccessfully loaded {len(signals_single)} single files.")
    X_single = np.array([
        extract_hybrid_features(preprocess_signal(s)) for s in signals_single
    ])
    best_clf_single, fitted_scaler_single, names_s, accs_s, estimators_s = train_models(
        X_single, y_single, feature_names=FEATURE_NAMES_SINGLE, title_suffix="[Single File]"
    )
    for name, acc in zip(names_s, accs_s):
        all_results[f"{name} (Single)"] = (estimators_s[name], fitted_scaler_single, acc)
else:
    print("Not enough single files found. Skipping single-file training.")
    best_clf_single, fitted_scaler_single = None, None

# --- Save best paired model and best single-file model separately ---
paired_results = {k: v for k, v in all_results.items() if k.endswith("(Paired)")}
single_results = {k: v for k, v in all_results.items() if k.endswith("(Single)")}

if paired_results:
    best_paired_label, (bp_model, bp_scaler, bp_acc) = max(paired_results.items(), key=lambda x: x[1][2])
    with open(PAIRED_MODEL_PATH, 'wb') as f:
        pickle.dump({'label': best_paired_label, 'model': bp_model, 'scaler': bp_scaler, 'accuracy': bp_acc}, f)
    print(f"\n[SAVED] Best paired model  -> '{PAIRED_MODEL_PATH}' | {best_paired_label} ({bp_acc:.1f}%)")

if single_results:
    best_single_label, (bs_model, bs_scaler, bs_acc) = max(single_results.items(), key=lambda x: x[1][2])
    with open(SINGLE_MODEL_PATH, 'wb') as f:
        pickle.dump({'label': best_single_label, 'model': bs_model, 'scaler': bs_scaler, 'accuracy': bs_acc}, f)
    print(f"[SAVED] Best single model  -> '{SINGLE_MODEL_PATH}' | {best_single_label} ({bs_acc:.1f}%)")

# Launch maze with best paired model (preferred) or best single
if paired_results or single_results:
    maze_clf    = best_clf_pairs      if best_clf_pairs    is not None else bs_model
    maze_scaler = fitted_scaler_pairs if fitted_scaler_pairs is not None else bs_scaler
    run_maze(maze_clf, maze_scaler)
else:
    print("No models were trained. Check your dataset path and file structure.")