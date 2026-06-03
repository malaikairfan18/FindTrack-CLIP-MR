# Guide: Running FindTrack-CLIP-MR on Kaggle

This guide walks you through running the upgraded **FindTrack-CLIP-MR** (Multi-Reference) pipeline directly on raw Kaggle datasets (without creating any combined/symlinked directories).

---

## 1. Preparation & Setup

### Step 1: Uploading the Deployment Cell
1. Copy the contents of the local file `C:\Users\Giga TECH\.gemini\antigravity\scratch\kaggle_cell_mr.py`.
2. Paste it into the first code cell of your Kaggle notebook and run it. 
3. This will create all the python code files in your workspace, install the necessary libraries (`loralib`, `ftfy`, `omegaconf`,TIMM, etc.), and download the Alpha-CLIP model weights.

---

## 2. Running Experiments directly on Raw Datasets

The codebase has been updated to automatically detect nested directory structures (such as Ref-YouTube-VOS's double-nested paths) and calibrate the scores using Min-Max Normalization.

### Experiment 1: Ref-YouTube-VOS (Multi-Reference)
Run the evaluation directly on the raw Ref-YouTube-VOS input dataset:
```bash
!python run_ytvos.py --mode mask_crop --w_finder 0.5 --w_clip 0.5 --num_refs 3 --min_distance 15 --dataset_path /kaggle/input/datasets/malaikairfan/ref-yt-vos/Ref-Youtube-Vos/rvos-20251030T175423Z-1-001/rvos
```

### Experiment 2: MeViS (Multi-Reference)
Run the evaluation directly on the raw MeViS-001 release input dataset:
```bash
!python run_mevis.py --mode mask_crop --w_finder 0.5 --w_clip 0.5 --num_refs 3 --min_distance 15 --dataset_path /kaggle/input/datasets/malaikairfan/mevis-01/Mevis/MeViS_release-20251030T175415Z-1-001/MeViS_release
```

---

## 3. Key Parameters
* `--num_refs`: Number of anchor reference frames to select (default: 3). Set to `1` for single-reference baseline.
* `--min_distance`: Minimum number of frames between selected anchors to ensure temporal diversity (default: 15).
* `--ref_num`: Total candidate frames sampled from the video (default: 10).
* `--w_finder` / `--w_clip`: Weights for score fusion (default: 0.5 / 0.5). Normalization handles scale calibration automatically.
