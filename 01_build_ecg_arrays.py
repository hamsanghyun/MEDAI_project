from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# 1. 경로 설정
# ============================================================

BASE_DIR = Path(r"C:\Users\hamsa\OneDrive\바탕 화면\MEDAI_project")
MANIFEST_PATH = BASE_DIR / "manifest.csv"

TARGET_LEN = 4096
N_LEADS = 12

manifest = pd.read_csv(MANIFEST_PATH)

print("===== Manifest Loaded =====")
print(manifest.head())
print()

print("===== Split / Label Count =====")
print(manifest.groupby(["split", "label"]).size())
print()

print("===== Signal Length Summary =====")
print(
    manifest.groupby("split")["sig_len"]
    .agg(["count", "min", "median", "mean", "max"])
)
print()

print("===== Sampling Rate Check =====")
print(manifest["fs"].value_counts())
print()

print("===== Lead Count Check =====")
print(manifest["n_leads"].value_counts())
print()


# ============================================================
# 2. ECG 신호 로딩 함수
# ============================================================

def load_ecg_signal(hea_path):
    """
    WFDB 형식의 .hea/.dat record를 읽어 ECG 신호를 반환한다.
    반환 shape: (signal_length, 12)
    """
    hea_path = Path(hea_path)
    record_path = str(hea_path.with_suffix(""))  # .hea 제거

    record = wfdb.rdrecord(record_path)
    signal = record.p_signal

    if signal is None:
        raise ValueError(f"p_signal을 읽을 수 없습니다: {hea_path}")

    signal = signal.astype(np.float32)

    if signal.ndim != 2:
        raise ValueError(f"신호 차원 오류: {hea_path}, shape={signal.shape}")

    if signal.shape[1] != N_LEADS:
        raise ValueError(f"lead 수 오류: {hea_path}, shape={signal.shape}")

    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    return signal


# ============================================================
# 3. 길이 통일 함수
# ============================================================

def crop_or_pad(signal, target_len=4096):
    """
    ECG 길이를 target_len으로 통일한다.
    긴 신호는 중앙 crop, 짧은 신호는 zero-padding 적용.
    input shape:  (length, 12)
    output shape: (4096, 12)
    """
    length = signal.shape[0]

    if length == target_len:
        return signal

    if length > target_len:
        start = (length - target_len) // 2
        end = start + target_len
        return signal[start:end, :]

    pad_total = target_len - length
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    padded = np.pad(
        signal,
        pad_width=((pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0
    )

    return padded


# ============================================================
# 4. record 단위 lead-wise z-score 정규화
# ============================================================

def normalize_per_record(signal):
    """
    각 ECG record 내부에서 lead별 z-score 정규화 수행.
    각 lead의 평균을 0, 표준편차를 1에 가깝게 조정한다.
    """
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)

    std[std < 1e-6] = 1.0

    normalized = (signal - mean) / std

    return normalized.astype(np.float32)


# ============================================================
# 5. split별 numpy 배열 생성
# ============================================================

def build_split_array(split_name):
    df = manifest[manifest["split"] == split_name].copy()
    df = df.reset_index(drop=True)

    n = len(df)

    X_path = BASE_DIR / f"X_{split_name}_raw4096.npy"
    y_path = BASE_DIR / f"y_{split_name}.npy"
    id_path = BASE_DIR / f"record_ids_{split_name}.csv"

    print(f"\n===== Building {split_name} set =====")
    print(f"records: {n}")
    print(f"X save path: {X_path}")

    X = np.lib.format.open_memmap(
        X_path,
        mode="w+",
        dtype=np.float32,
        shape=(n, TARGET_LEN, N_LEADS)
    )

    y = df["label"].values.astype(np.int64)

    failed = []

    for i, row in tqdm(df.iterrows(), total=n):
        try:
            signal = load_ecg_signal(row["hea_path"])
            signal = crop_or_pad(signal, TARGET_LEN)
            signal = normalize_per_record(signal)

            X[i] = signal

        except Exception as e:
            failed.append({
                "record_id": row["record_id"],
                "split": split_name,
                "error": str(e)
            })

            X[i] = np.zeros((TARGET_LEN, N_LEADS), dtype=np.float32)

    np.save(y_path, y)

    df[["record_id", "split", "label", "hea_path", "dat_path", "sig_len", "age", "sex"]].to_csv(
        id_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"{split_name} X shape: {X.shape}")
    print(f"{split_name} y shape: {y.shape}")
    print(f"{split_name} failed count: {len(failed)}")

    if len(failed) > 0:
        failed_path = BASE_DIR / f"failed_{split_name}.csv"
        pd.DataFrame(failed).to_csv(failed_path, index=False, encoding="utf-8-sig")
        print(f"failed file saved: {failed_path}")
        print(pd.DataFrame(failed).head())

    return failed


# ============================================================
# 6. 실행
# ============================================================

failed_train = build_split_array("train")
failed_validation = build_split_array("validation")
failed_test = build_split_array("test")

print("\n===== Final Failed Count =====")
print("train:", len(failed_train))
print("validation:", len(failed_validation))
print("test:", len(failed_test))


# ============================================================
# 7. 저장된 배열 확인
# ============================================================

X_train = np.load(BASE_DIR / "X_train_raw4096.npy", mmap_mode="r")
y_train = np.load(BASE_DIR / "y_train.npy")

X_validation = np.load(BASE_DIR / "X_validation_raw4096.npy", mmap_mode="r")
y_validation = np.load(BASE_DIR / "y_validation.npy")

X_test = np.load(BASE_DIR / "X_test_raw4096.npy", mmap_mode="r")
y_test = np.load(BASE_DIR / "y_test.npy")

print("\n===== Saved Array Shape Check =====")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_validation:", X_validation.shape)
print("y_validation:", y_validation.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\n===== Label Distribution Check =====")
print("train:", np.bincount(y_train))
print("validation:", np.bincount(y_validation))
print("test:", np.bincount(y_test))


# ============================================================
# 8. 샘플 ECG plot 저장
# ============================================================

sample_idx = 0
sample_signal = X_train[sample_idx]

plt.figure(figsize=(12, 6))
for lead_idx in range(12):
    plt.plot(sample_signal[:, lead_idx] + lead_idx * 5)

plt.title("Sample 12-lead ECG after preprocessing")
plt.xlabel("Sample index")
plt.ylabel("Normalized amplitude with offset")
plt.tight_layout()

plot_path = BASE_DIR / "sample_preprocessed_ecg.png"
plt.savefig(plot_path, dpi=200)
plt.close()

print(f"\nSample ECG plot saved: {plot_path}")

print("\n전처리 완료")