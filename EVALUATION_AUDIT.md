# Evaluation Audit — Semiconductor Wafer Defect Classification (DefectAI)

Internal record of the data-integrity audit performed on this project's originally reported 92.05% top-1 / 99.97% top-5 accuracy, the leakage found, the fix applied, and the verified result. Written honestly — this document exists to let any claim in the README be traced back to a specific, reproducible step.

---

## 1. Objective

Verify whether the originally reported test accuracy (92.05% top-1, 99.97% top-5) on the WM-811K wafer defect dataset reflects genuine generalization, or whether it was inflated by a methodology flaw.

## 2. Original pipeline (as found)

Traced directly from the project's own source files:

- `scripts/preprocess_dataset.py` — for each defect class, loads **all** raw wafer maps for that class, converts each to an image, and — if the class is below a target count (5,000 for most classes, 10,000 for `none`) — immediately generates augmented siblings (flips, rotations, noise) of that same image. Originals and augmented copies are saved into the same `data/processed/<class>/` folder with filenames like `<class>_<idx>_orig.png` and `<class>_<idx>_aug0.png`, `_aug1.png`, etc.
- `scripts/split_dataset.py` — globs **every PNG** in each class folder (originals and augmented copies together, with no distinction between them), shuffles the combined list, and splits it 70/15/15 into train/val/test.

**The flaw:** augmentation happened *before* the train/val/test split, and the split treated augmented siblings of the same source wafer as independent samples. A rotated/flipped/noise-added copy of a wafer could land in the training set while another copy of the *same underlying wafer* landed in the test set.

## 3. Measuring the leakage

A dedicated audit script (`check_split_leakage.py`) was written to extract each image's source wafer ID from its filename (stripping the `_orig`/`_augN` suffix) and check whether any wafer ID appeared in more than one split.

**Result on the original `data/splits/` (run on Google Drive via Colab):**
- Total unique source wafers: 35,519
- Wafers with siblings split across multiple sets: 6,104 (17.19%)
- **Test-set wafers with a sibling in the training set: 3,493 out of 7,744 (45.11%)**

This confirms the leakage mechanism was not theoretical — nearly half the test set had train-time duplicates.

## 4. Model configuration (as originally trained)

Recovered from `train_local.py` and `models/results.csv` (Ultralytics training log):
- Architecture: YOLOv8n-cls
- Image size: 64×64
- Batch size: 32
- Epochs: 10, patience: 5
- Device: CPU (local machine)
- **No random seed was set** in the original training script — a separate, minor reproducibility gap noted but not corrected retroactively.
- `model_metrics.json` confirms the reported 92.05%/99.97% was evaluated on `data/splits/test` (9,370 images) — the same contaminated split measured in §3.

**Anomaly noted, not resolved:** per-epoch training time in `results.csv` shows epochs 1–6 taking 75–99 minutes each, while epochs 7–10 took only 12–14 minutes each (~6× faster). The most likely explanation is OS-level disk-cache warming (file reads getting faster as the OS caches them in RAM), which is common and benign on CPU training with large datasets. A hardware or environment change mid-run is a less likely but not fully excluded alternative. This does not appear to have affected the final metrics (see §6), but the exact training conditions cannot be fully reconstructed from available records.

## 5. The fix

A corrected pipeline (`scripts/preprocess_and_split_clean.py`) was written with one structural change: **the train/val/test split is performed on the original (pre-augmentation) wafer indices, by class, before any image processing or augmentation occurs.** Augmentation is then applied only to the training portion, to balance class counts. Validation and test sets contain only real, unaugmented wafers, preserving their natural (imbalanced) class distribution — the correct approach, since a model should be evaluated on realistic, non-rebalanced data.

Output was written to a new folder, `data/splits_clean/`, leaving the original (contaminated) `data/splits/` untouched for comparison.

**Verification (same leakage-check script, run against `data/splits_clean/`):**
- Total unique source wafers: 35,518 (matches original count, confirming this is a fair, like-for-like rebuild)
- Wafers with siblings split across multiple sets: **0**
- Test-set wafers with a sibling in train: **0 out of 5,333 (0.00%)**

## 6. Retraining and verified result

The model was retrained from scratch on `data/splits_clean` using the same architecture and hyperparameters as the original run (YOLOv8n-cls, imgsz=64, batch=32, epochs=10, patience=5), with one addition: `seed=42` was set this time for reproducibility (the original run had none, so this is not a perfectly identical replication, only a matched-hyperparameter one).

| | Original (leaked test set) | Retrained (clean, 0% leakage) |
|---|---|---|
| Top-1 accuracy | 92.05% | **91.56%** |
| Top-5 accuracy | 99.97% | **100.00%** |
| Test set | `data/splits/test` (45.11% leaked) | `data/splits_clean/test` (0% leaked, verified) |

The result was independently re-verified via direct per-image prediction (bypassing Ultralytics' internal confusion-matrix object, whose row/column indexing produced inconsistent per-class totals on first attempt and was discarded in favor of this more transparent method): overall accuracy 0.9158, matching the reported 0.9156 top-1 to within rounding.

## 7. Per-class accuracy (clean test set, verified by direct prediction)

| Class | Accuracy | Correct / Total |
|---|---|---|
| Near-full | 100.00% | 23 / 23 |
| Random | 98.47% | 129 / 131 |
| Donut | 97.62% | 82 / 84 |
| Center | 96.90% | 625 / 645 |
| Edge-Ring | 96.90% | 1407 / 1452 |
| Scratch | 94.44% | 170 / 180 |
| Edge-Loc | 91.01% | 709 / 779 |
| none | 87.53% | 1313 / 1500 |
| Loc | 79.07% | 427 / 540 |

This breakdown was specifically checked to rule out a hidden failure mode: that leakage might have disproportionately inflated performance on rare, heavily-augmented classes (e.g., `Near-full`, `Donut`) while an aggregate number looked fine. It did not — the rarest classes perform among the best. `Loc` is the weakest class, but it is a mid-frequency class, not the rarest, and its lower accuracy most plausibly reflects genuine visual confusion with similar defect types (`Edge-Loc`, `Center`) rather than a leakage artifact.

## 8. Final finding

**The originally reported 92.05%/99.97% accuracy was measured on a test set with confirmed, substantial data leakage (45.11% of test wafers had a training-time sibling). After fixing the leakage (splitting raw wafers before augmentation, balancing only the training set) and retraining with matched hyperparameters, the verified accuracy on a leakage-free test set is 91.56% top-1 / 100% top-5 — nearly identical to the original.** Per-class analysis confirms this holds across both common and rare defect categories. This is a confirmatory finding: the leakage, while real and worth catching and fixing, does not appear to have substantially inflated the model's true performance in aggregate, though the original evaluation protocol was not rigorous and should not be trusted on its own.

## 9. Limitations

- The original training run's exact conditions (possible mid-run environment change) cannot be fully reconstructed from available logs.
- The original run was not seeded; the retrained comparison uses a fixed seed, so the two runs are matched on hyperparameters but not on an identical random initialization.
- Only one retraining run was performed on the clean split; unlike the flood-detection project, this result has not yet been checked for run-to-run variance across multiple independent trainings.
- `Loc`'s lower per-class accuracy (79.07%) has not been further investigated (e.g., via confusion-matrix inspection of which classes it's confused with) — flagged as future work.

## 10. Future work

- Run 2–3 additional independent retrains on `data/splits_clean` (different seeds) to establish variance, matching the reproducibility-check standard used in the flood-detection project.
- Investigate `Loc`'s confusion pattern specifically — which classes is it most often misclassified as?
- Replace `models/best.pt` in the repository with the clean-split-trained checkpoint, so the deployed/demoed model matches the verified numbers.
