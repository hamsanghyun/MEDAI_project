\# Chagas ECG 1D-CNN Screening



This repository contains source code for a medical AI project on Chagas disease screening using raw 12-lead ECG signals and 1D-CNN models.



\## Contents



\- `manifest.py`: Manifest generation and dataset validation

\- `01\_build\_ecg\_arrays.py`: ECG signal loading and preprocessing

\- `02\_train\_basic\_1dcnn.py`: Basic 1D-CNN training and evaluation

\- `03\_threshold\_optimization.py`: Validation-based threshold optimization

\- `04\_train\_class\_weight\_1dcnn.py`: Class-weight 1D-CNN training and evaluation



\## Data



Original ECG data files (`.hea`, `.dat`), preprocessed arrays (`.npy`), trained models (`.keras`), result files, and report files are not included in this repository due to data size and usage constraints.



Users should modify the data path in each script according to their local environment.

