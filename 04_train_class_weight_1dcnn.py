from pathlib import Path
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


# ============================================================
# 1. 기본 설정
# ============================================================

BASE_DIR = Path(r"C:\Users\hamsa\OneDrive\바탕 화면\MEDAI_project")
RESULT_DIR = BASE_DIR / "results_class_weight_1dcnn"
RESULT_DIR.mkdir(exist_ok=True)

SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
INPUT_SHAPE = (4096, 12)

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow version:", tf.__version__)


# ============================================================
# 2. 데이터 경로
# ============================================================

X_train_path = BASE_DIR / "X_train_raw4096.npy"
y_train_path = BASE_DIR / "y_train.npy"

X_val_path = BASE_DIR / "X_validation_raw4096.npy"
y_val_path = BASE_DIR / "y_validation.npy"

X_test_path = BASE_DIR / "X_test_raw4096.npy"
y_test_path = BASE_DIR / "y_test.npy"


# ============================================================
# 3. 데이터 로더
# ============================================================

class ECGSequence(tf.keras.utils.Sequence):
    def __init__(self, X_path, y_path, batch_size=32, shuffle=False, **kwargs):
        super().__init__(**kwargs)
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


train_seq = ECGSequence(X_train_path, y_train_path, batch_size=BATCH_SIZE, shuffle=True)
val_seq = ECGSequence(X_val_path, y_val_path, batch_size=BATCH_SIZE, shuffle=False)
test_seq = ECGSequence(X_test_path, y_test_path, batch_size=BATCH_SIZE, shuffle=False)

y_train = np.load(y_train_path).astype(int)
y_val = np.load(y_val_path).astype(int)
y_test = np.load(y_test_path).astype(int)

print("Train label distribution:", np.bincount(y_train))
print("Validation label distribution:", np.bincount(y_val))
print("Test label distribution:", np.bincount(y_test))


# ============================================================
# 4. Class weight 계산
# ============================================================

classes = np.array([0, 1])
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weight = {
    0: float(weights[0]),
    1: float(weights[1])
}

print("\n===== Class Weight =====")
print(class_weight)

pd.DataFrame([
    {
        "class": 0,
        "class_name": "negative",
        "weight": class_weight[0]
    },
    {
        "class": 1,
        "class_name": "positive",
        "weight": class_weight[1]
    }
]).to_csv(RESULT_DIR / "class_weight.csv", index=False, encoding="utf-8-sig")


# ============================================================
# 5. 모델 정의
# ============================================================

def build_class_weight_1dcnn(input_shape=(4096, 12)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(
        filters=32,
        kernel_size=7,
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=4)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(
        filters=64,
        kernel_size=5,
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=4)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(
        filters=128,
        kernel_size=5,
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=4)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(
        filters=256,
        kernel_size=3,
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


model = build_class_weight_1dcnn(INPUT_SHAPE)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(curve="ROC", name="auroc"),
        tf.keras.metrics.AUC(curve="PR", name="auprc")
    ]
)

model.summary()


# ============================================================
# 6. 콜백 설정
# ============================================================

checkpoint_path = RESULT_DIR / "best_class_weight_1dcnn.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_auprc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auprc",
        mode="max",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auprc",
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )
]


# ============================================================
# 7. 모델 학습
# ============================================================

history = model.fit(
    train_seq,
    validation_data=val_seq,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

history_df = pd.DataFrame(history.history)
history_df.to_csv(
    RESULT_DIR / "training_history_class_weight_1dcnn.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 8. 학습 곡선 저장
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss - Class-weight 1D-CNN")
plt.legend()
plt.tight_layout()
plt.savefig(RESULT_DIR / "loss_curve_class_weight_1dcnn.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history["auprc"], label="train_auprc")
plt.plot(history.history["val_auprc"], label="val_auprc")
plt.xlabel("Epoch")
plt.ylabel("AUPRC")
plt.title("Training and Validation AUPRC - Class-weight 1D-CNN")
plt.legend()
plt.tight_layout()
plt.savefig(RESULT_DIR / "auprc_curve_class_weight_1dcnn.png", dpi=200)
plt.close()


# ============================================================
# 9. Best model 로드 후 test 평가
# ============================================================

best_model = tf.keras.models.load_model(checkpoint_path)

y_score = best_model.predict(test_seq).ravel()
y_pred = (y_score >= 0.5).astype(int)

metrics = {
    "AUROC": roc_auc_score(y_test, y_score),
    "AUPRC": average_precision_score(y_test, y_score),
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1-score": f1_score(y_test, y_pred, zero_division=0),
}

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(
    RESULT_DIR / "test_metrics_class_weight_1dcnn.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== Test Metrics: Class-weight 1D-CNN =====")
print(metrics_df)


# ============================================================
# 10. Confusion Matrix 저장
# ============================================================

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=["Actual Negative", "Actual Positive"],
    columns=["Predicted Negative", "Predicted Positive"]
)
cm_df.to_csv(
    RESULT_DIR / "confusion_matrix_class_weight_1dcnn.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Confusion Matrix - Class-weight 1D-CNN")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks([0, 1], ["Negative", "Positive"])
plt.yticks([0, 1], ["Negative", "Positive"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig(RESULT_DIR / "confusion_matrix_class_weight_1dcnn.png", dpi=200)
plt.close()


# ============================================================
# 11. ROC curve 저장
# ============================================================

fpr, tpr, _ = roc_curve(y_test, y_score)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUROC = {metrics['AUROC']:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Class-weight 1D-CNN")
plt.legend()
plt.tight_layout()
plt.savefig(RESULT_DIR / "roc_curve_class_weight_1dcnn.png", dpi=200)
plt.close()


# ============================================================
# 12. PR curve 저장
# ============================================================

precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_score)

plt.figure(figsize=(6, 5))
plt.plot(recall_curve, precision_curve, label=f"AUPRC = {metrics['AUPRC']:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Class-weight 1D-CNN")
plt.legend()
plt.tight_layout()
plt.savefig(RESULT_DIR / "pr_curve_class_weight_1dcnn.png", dpi=200)
plt.close()


# ============================================================
# 13. 예측 결과 저장
# ============================================================

record_ids_test_path = BASE_DIR / "record_ids_test.csv"

if record_ids_test_path.exists():
    record_ids_test = pd.read_csv(record_ids_test_path)
    pred_df = record_ids_test.copy()
else:
    pred_df = pd.DataFrame({"index": np.arange(len(y_test))})

pred_df["true_label"] = y_test
pred_df["pred_score"] = y_score
pred_df["pred_label_0.5"] = y_pred

pred_df.to_csv(
    RESULT_DIR / "test_predictions_class_weight_1dcnn.csv",
    index=False,
    encoding="utf-8-sig"
)

print(f"\n결과 저장 폴더: {RESULT_DIR}")
print("Class-weight 1D-CNN 학습 및 평가 완료")