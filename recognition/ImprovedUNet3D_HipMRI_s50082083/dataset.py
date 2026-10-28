"""
Data loading, patient-level splitting, preprocessing and augmentation for the
HipMRI 3D dataset (CSIRO "Labelled weekly MR images of the male pelvis").

Pipeline:
    1. find_cases()      pair every MR volume with its label volume (patient + week)
    2. split_patients()  leakage-free split by PATIENT (repeat weekly scans of one
                         patient never appear in two splits)
    3. preprocess()      one-off: z-score normalise, fix depth to 128 slices, store
                         each case as uncompressed .npy so slices can be read fast
    4. HipMRISlices2D    PyTorch Dataset of full-resolution axial slices (2D models)

Array layout after preprocessing is (D, H, W) = (axial slice, x, y):
    D = superior-inferior (128 axial slices), H = left-right, W = anterior-posterior.
"""

import glob
import os
import random
import re


LABEL_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]
NUM_CLASSES = len(LABEL_NAMES)
PROSTATE = 5

# File-name rules, checked against the data audit (211 pairs, 0 unparsed).
PATIENT_RE = re.compile(r"case[_\-]?(\d+)", re.IGNORECASE)
WEEK_RE = re.compile(r"week[_\-]?(\d+)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# 1. Finding and pairing files
# --------------------------------------------------------------------------- #
def _is_label(path):
    """A file is a label volume if its name or folder says so (e.g. *_SEMANTIC_*)."""
    name = os.path.basename(path).lower()
    folder = os.path.basename(os.path.dirname(path)).lower()
    return any(k in name for k in ("label", "seg", "semantic")) or any(
        k in folder for k in ("label", "seg")
    )


def find_cases(root):
    """Return a sorted list of cases, one dict per (patient, week) image/label pair."""
    files = sorted(glob.glob(os.path.join(root, "**", "*.nii*"), recursive=True))
    images, labels = {}, {}
    for f in files:
        name = os.path.basename(f)
        patient = PATIENT_RE.search(name)
        if patient is None:
            continue
        week = WEEK_RE.search(name)
        key = (patient.group(1).zfill(3), week.group(1) if week else "0")
        (labels if _is_label(f) else images)[key] = f

    keys = sorted(set(images) & set(labels))
    if not keys:
        raise RuntimeError(f"No image/label pairs found under {root}")
    return [
        {"case_id": f"{p}_w{w}", "patient": p, "week": w,
         "image": images[(p, w)], "label": labels[(p, w)]}
        for p, w in keys
    ]


# --------------------------------------------------------------------------- #
# 2. Leakage-free split by patient
# --------------------------------------------------------------------------- #
def split_patients(patients, n_val=6, n_test=6, seed=42):
    """Shuffle unique patient IDs with a fixed seed and split them train/val/test.

    Splitting by patient (not by scan) matters because each patient has up to
    8 near-identical weekly scans; a scan-level split would leak test anatomy
    into training and inflate the scores.
    """
    patients = sorted(set(patients))
    rng = random.Random(seed)
    rng.shuffle(patients)
    test = sorted(patients[:n_test])
    val = sorted(patients[n_test:n_test + n_val])
    train = sorted(patients[n_test + n_val:])

    assert train and val and test, "Not enough patients for the requested split"
    assert set(train).isdisjoint(val), "Patient leakage: train/val overlap"
    assert set(train).isdisjoint(test), "Patient leakage: train/test overlap"
    assert set(val).isdisjoint(test), "Patient leakage: val/test overlap"
    return train, val, test


def case_ids_for(cases, patients):
    """All case IDs (every weekly scan) belonging to the given patients."""
    wanted = set(patients)
    return [c["case_id"] for c in cases if c["patient"] in wanted]


def patient_of(case_id):
    """'004_w3' -> '004'."""
    return case_id.split("_w")[0]
