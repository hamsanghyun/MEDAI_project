from pathlib import Path
import pandas as pd

# =========================
# 1. 경로 설정
# =========================
BASE_DIR = Path(r"C:\Users\hamsa\OneDrive\바탕 화면\MEDAI_project")  # train/validation/test가 있는 상위 폴더

splits = ["train", "validation", "test"]

rows = []
missing_dat = []
missing_label = []
format_errors = []

# =========================
# 2. .hea 파싱 함수
# =========================
def parse_hea_file(hea_path):
    with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f.readlines()]
    
    # 첫 줄: record_id, lead 수, sampling rate, signal length
    first = lines[0].split()
    
    if len(first) < 4:
        raise ValueError(f"첫 줄 형식 오류: {hea_path}")
    
    record_id = first[0]
    n_leads = int(first[1])
    fs = int(float(first[2]))
    sig_len = int(first[3])
    
    age = None
    sex = None
    chagas_label = None
    source = None
    
    for line in lines:
        lower = line.lower()
        
        if lower.startswith("# age:"):
            age = line.split(":", 1)[1].strip()
            if age.lower() in ["nan", "none", ""]:
                age = None
            else:
                try:
                    age = float(age)
                except:
                    age = None
        
        elif lower.startswith("# sex:"):
            sex = line.split(":", 1)[1].strip()
        
        elif lower.startswith("# chagas label:"):
            value = line.split(":", 1)[1].strip().lower()
            
            if value == "true":
                chagas_label = 1
            elif value == "false":
                chagas_label = 0
            else:
                chagas_label = None
        
        elif lower.startswith("# source:"):
            source = line.split(":", 1)[1].strip()
    
    return {
        "record_id": record_id,
        "n_leads": n_leads,
        "fs": fs,
        "sig_len": sig_len,
        "age": age,
        "sex": sex,
        "label": chagas_label,
        "source": source,
    }

# =========================
# 3. train/validation/test 폴더 스캔
# =========================
for split in splits:
    split_path = BASE_DIR / split
    
    if not split_path.exists():
        raise FileNotFoundError(f"폴더가 존재하지 않습니다: {split_path}")
    
    hea_files = sorted(split_path.glob("*.hea"))
    print(f"{split}: .hea files = {len(hea_files)}")
    
    for hea_path in hea_files:
        dat_path = hea_path.with_suffix(".dat")
        
        if not dat_path.exists():
            missing_dat.append(str(hea_path))
            continue
        
        try:
            info = parse_hea_file(hea_path)
        except Exception as e:
            format_errors.append((str(hea_path), str(e)))
            continue
        
        if info["label"] is None:
            missing_label.append(str(hea_path))
            continue
        
        rows.append({
            "record_id": info["record_id"],
            "split": split,
            "label": info["label"],
            "hea_path": str(hea_path),
            "dat_path": str(dat_path),
            "n_leads": info["n_leads"],
            "fs": info["fs"],
            "sig_len": info["sig_len"],
            "age": info["age"],
            "sex": info["sex"],
            "source": info["source"],
        })

manifest = pd.DataFrame(rows)

# =========================
# 4. 기본 검증
# =========================
print("\n===== Manifest Preview =====")
print(manifest.head())

print("\n===== Total Records =====")
print(len(manifest))

print("\n===== Split / Label Count =====")
summary = (
    manifest
    .groupby(["split", "label"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={0: "negative", 1: "positive"})
)

summary["total"] = summary.sum(axis=1)
print(summary)

print("\n===== Signal Format Check =====")
print(manifest[["n_leads", "fs", "sig_len"]].drop_duplicates())

print("\n===== Missing .dat count =====")
print(len(missing_dat))

if missing_dat:
    print(missing_dat[:10])

print("\n===== Missing label count =====")
print(len(missing_label))

if missing_label:
    print(missing_label[:10])

print("\n===== Header format error count =====")
print(len(format_errors))

if format_errors:
    print(format_errors[:5])

# =========================
# 5. 기대 개수 검증
# =========================
expected = {
    ("train", 1): 3269,
    ("train", 0): 6471,
    ("validation", 1): 1635,
    ("validation", 0): 3235,
    ("test", 1): 1634,
    ("test", 0): 3236,
}

print("\n===== Expected Count Check =====")

all_ok = True

for (split, label), expected_count in expected.items():
    actual_count = len(
        manifest[
            (manifest["split"] == split) &
            (manifest["label"] == label)
        ]
    )
    
    status = "OK" if actual_count == expected_count else "CHECK"
    
    if actual_count != expected_count:
        all_ok = False
    
    label_name = "positive" if label == 1 else "negative"
    print(
        f"{split:10s} / {label_name:8s}: "
        f"actual={actual_count:5d}, expected={expected_count:5d} [{status}]"
    )

# =========================
# 6. split 간 중복 확인
# =========================
duplicated_ids = manifest[manifest.duplicated("record_id", keep=False)]

print("\n===== Duplicate record_id check =====")

if len(duplicated_ids) == 0:
    print("중복 record_id 없음")
else:
    print(f"중복 record_id 개수: {duplicated_ids['record_id'].nunique()}")
    print(duplicated_ids.sort_values("record_id").head(20))

# =========================
# 7. 저장
# =========================
output_path = BASE_DIR / "manifest.csv"
manifest.to_csv(output_path, index=False, encoding="utf-8-sig")

print("\n===== Final Status =====")
if all_ok and len(missing_dat) == 0 and len(missing_label) == 0 and len(format_errors) == 0 and len(duplicated_ids) == 0:
    print("데이터셋 검증 완료")
else:
    print("검증 필요: 위 항목 확인")

print(f"\nmanifest 저장 완료: {output_path}")