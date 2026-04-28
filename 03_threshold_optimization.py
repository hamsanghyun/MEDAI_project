from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# ============================================================
# 1. 경로 설정
# ============================================================

BASE_DIR = Path(r"C:\Users\hamsa\OneDrive\바탕 화면\MEDAI_project")
RESULT_DIR = BASE_DIR / "results_basic_1dcnn"
MODEL_PATH = RESULT_DIR / "best_basic_1dcnn.keras"

X_val_path = BASE_DIR / "X_validation_raw4096.npy"
y_val_path = BASE_DIR / "y_validation.npy"

X_test_path = BASE_DIR / "X_test_raw4096.npy"
y_test_path = BASE_DIR / "y_test.npy"

BATCH_SIZE = 32


# ============================================================
# 2. 데이터 로더
# ============================================================

class ECGSequence(tf.keras.utils.Sequence):
    def __init__(self, X_path, y_path, batch_size=32, shuffle=False):
        super().__init__()
        self.X = np.load(X_path, mmap_mode="r")
        self.y = np.load(y_path).astype(np.float32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.y))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = np.array(self.X[batch_idx], dtype=np.float32)
        y_batch = self.y[batch_idx]
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


val_seq = ECGSequence(X_val_path, y_val_path, batch_size=BATCH_SIZE, shuffle=False)
test_seq = ECGSequence(X_test_path, y_test_path, batch_size=BATCH_SIZE, shuffle=False)

y_val = np.load(y_val_path).astype(int)
y_test = np.load(y_test_path).astype(int)


# ============================================================
# 3. 모델 로드 및 확률 예측
# ============================================================

model = tf.keras.models.load_model(MODEL_PATH)

val_score = model.predict(val_seq).ravel()
test_score = model.predict(test_seq).ravel()


# ============================================================
# 4. threshold 탐색 함수
# ============================================================

def evaluate_at_threshold(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "tn": confusion_matrix(y_true, y_pred).ravel()[0],
        "fp": confusion_matrix(y_true, y_pred).ravel()[1],
        "fn": confusion_matrix(y_true, y_pred).ravel()[2],
        "tp": confusion_matrix(y_true, y_pred).ravel()[3],
    }


thresholds = np.linspace(0.01, 0.99, 991)

val_rows = []

for th in thresholds:
    row = evaluate_at_threshold(y_val, val_score, th)
    val_rows.append(row)

val_threshold_df = pd.DataFrame(val_rows)

# F1 최대 threshold
best_f1_row = val_threshold_df.loc[val_threshold_df["f1"].idxmax()]
best_f1_threshold = float(best_f1_row["threshold"])

# F2 최대 threshold
best_f2_row = val_threshold_df.loc[val_threshold_df["f2"].idxmax()]
best_f2_threshold = float(best_f2_row["threshold"])

# Recall 0.80 이상 중 Precision 최대 threshold
recall_target = 0.80
candidate = val_threshold_df[val_threshold_df["recall"] >= recall_target].copy()

if len(candidate) > 0:
    best_recall80_row = candidate.loc[candidate["precision"].idxmax()]
    best_recall80_threshold = float(best_recall80_row["threshold"])
else:
    best_recall80_threshold = None


# ============================================================
# 5. test set에 고정 threshold 적용
# ============================================================

test_summary_rows = []

threshold_settings = [
    ("Default 0.5", 0.5),
    ("Validation F1-optimal", best_f1_threshold),
    ("Validation F2-optimal", best_f2_threshold),
]

if best_recall80_threshold is not None:
    threshold_settings.append((f"Validation Recall>={recall_target:.2f}", best_recall80_threshold))

for name, th in threshold_settings:
    result = evaluate_at_threshold(y_test, test_score, th)
    result["setting"] = name
    test_summary_rows.append(result)

test_threshold_df = pd.DataFrame(test_summary_rows)

# 보기 좋게 컬럼 정리
test_threshold_df = test_threshold_df[
    [
        "setting",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f2",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
]

# 전체 threshold sweep 저장
val_threshold_df.to_csv(
    RESULT_DIR / "validation_threshold_sweep.csv",
    index=False,
    encoding="utf-8-sig"
)

test_threshold_df.to_csv(
    RESULT_DIR / "test_threshold_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 6. 출력
# ============================================================

print("\n===== Validation threshold optimization =====")
print("Best F1 threshold:", best_f1_threshold)
print(best_f1_row)

print("\nBest F2 threshold:", best_f2_threshold)
print(best_f2_row)

if best_recall80_threshold is not None:
    print(f"\nBest threshold with validation recall >= {recall_target:.2f}:")
    print(best_recall80_threshold)
    print(best_recall80_row)
else:
    print(f"\nNo threshold satisfied validation recall >= {recall_target:.2f}")

print("\n===== Test performance by selected thresholds =====")
print(test_threshold_df)

print("\n===== Threshold optimization complete =====")
print("Saved:", RESULT_DIR / "validation_threshold_sweep.csv")
print("Saved:", RESULT_DIR / "test_threshold_comparison.csv")