#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brain FreeSurfer SPHARM coefficient diffusion.

This script is tailored for FreeSurfer subject folders such as:

  /data/human/ADNI/FreeSurfer/002_S_0559/I118679/surf/
    lh.sphere.reg
    lh.thickness
    lh.smoothwm.K1.crv
    lh.smoothwm.K2.crv
    rh.sphere.reg
    rh.thickness
    rh.smoothwm.K1.crv
    rh.smoothwm.K2.crv

Goal:
  1) Convert CTh, K1, K2 cortical surface maps to real SPHARM coefficients.
  2) Train a diffusion model directly on coefficient tensors [3, (degree+1)^2].
  3) Sample conditionally from demographics such as subject/group/sex/age/days/hemi.
  4) Reconstruct generated coefficients back to vertex-space CTh/K1/K2 maps.

Main modes:
  prepare_fs    FreeSurfer surf dirs -> prepared .npz coefficient files
  train         Train diffusion on prepared .npz files
  sample        Generate coefficient .npz samples from a trained checkpoint
  reconstruct   Coefficients -> vertex-space maps on a chosen FreeSurfer sphere

Recommended first experiment:
  degree=40 or 60, condition_keys=subject,group,sex,age,days,hemi, aux_channel_mask_prob=0.20~0.30.

Dependencies:
  numpy scipy torch nibabel
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import logging
import math
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import gammaln, lpmv
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

try:
    from nibabel.freesurfer import io as fsio
except Exception as exc:  # pragma: no cover
    fsio = None
    _NIBABEL_IMPORT_ERROR = exc
else:
    _NIBABEL_IMPORT_ERROR = None


# ---------------------------------------------------------------------
# Logging / seed
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("brain_spharm_diffusion")

FEATURE_NAMES = ("CTh", "K1", "K2")
DEFAULT_FEATURE_FILES = {
    "CTh": "{hemi}.thickness",
    "K1": "{hemi}.smoothwm.K1.crv",
    "K2": "{hemi}.smoothwm.K2.crv",
}


def require_nibabel() -> None:
    if fsio is None:
        raise RuntimeError(
            "nibabel is required for FreeSurfer IO. Install with `pip install nibabel`. "
            f"Original import error: {_NIBABEL_IMPORT_ERROR}"
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------
# SPHARM basis and coefficient projection
# ---------------------------------------------------------------------
def spharm_degree_index(degree: int) -> torch.Tensor:
    """Return length (degree+1)^2 tensor: [0, 1,1,1, 2,2,2,2,2, ...]."""
    degs: List[int] = []
    for ell in range(degree + 1):
        degs.extend([ell] * (2 * ell + 1))
    return torch.tensor(degs, dtype=torch.long)


def spharm_loss_weights(
    degree: int,
    low_freq_boost: float = 4.0,
    low_freq_power: float = 2.0,
    high_freq_base: float = 1.0,
) -> torch.Tensor:
    """
    Degree-wise loss weights.

    High frequencies keep base weight 1; low frequencies get a moderate boost.
    The mean is normalized to 1 so the learning-rate scale stays stable.
    """
    deg = spharm_degree_index(degree).float()
    w = high_freq_base + low_freq_boost / ((deg + 1.0) ** low_freq_power)
    w = w / w.mean()
    return w.view(1, 1, -1)


def spharm_lowpass_mask(degree: int, cut_l: Optional[int], tau: float = 10.0) -> torch.Tensor:
    """Smoothly damp coefficients above cut_l during sampling."""
    deg = spharm_degree_index(degree).float()
    if cut_l is None or cut_l < 0 or cut_l >= degree:
        return torch.ones(1, 1, (degree + 1) ** 2)
    over = torch.clamp(deg - float(cut_l), min=0.0)
    mask = torch.exp(-((over / float(tau)) ** 2))
    return mask.view(1, 1, -1)


def normalize_sphere_vertices(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vertices / norms


def real_spharm_basis_chunk(vertices_unit: np.ndarray, degree: int) -> np.ndarray:
    """
    Build real spherical harmonic basis for a chunk of unit-sphere vertices.

    Input:
      vertices_unit: [V, 3]
    Return:
      Y: [L, V], L=(degree+1)^2

    Ordering per degree ell is m=-ell,...,-1,0,1,...,ell.
    For m<0 we use sqrt(2)*N_lm*sin(|m| phi)*P_l^|m|(cos theta).
    For m>0 we use sqrt(2)*N_lm*cos(m phi)*P_l^m(cos theta).

    The exact convention is less important than using the same function for
    projection and reconstruction. This implementation is self-contained and
    does not rely on the custom `legendre` helper in older code.
    """
    x = normalize_sphere_vertices(vertices_unit)
    vx, vy, vz = x[:, 0], x[:, 1], x[:, 2]
    phi = np.arctan2(vy, vx)  # [-pi, pi]
    cos_theta = np.clip(vz, -1.0, 1.0)

    V = x.shape[0]
    L = (degree + 1) ** 2
    Y = np.empty((L, V), dtype=np.float64)

    idx = 0
    for ell in range(degree + 1):
        # negative m: -ell ... -1
        for m_abs in range(ell, 0, -1):
            log_norm = 0.5 * (
                math.log(2 * ell + 1) - math.log(4 * math.pi)
                + gammaln(ell - m_abs + 1) - gammaln(ell + m_abs + 1)
            )
            norm = math.sqrt(2.0) * math.exp(log_norm)
            P = lpmv(m_abs, ell, cos_theta)
            Y[idx] = norm * np.sin(m_abs * phi) * P
            idx += 1

        # m = 0
        norm0 = math.sqrt((2 * ell + 1) / (4 * math.pi))
        Y[idx] = norm0 * lpmv(0, ell, cos_theta)
        idx += 1

        # positive m: 1 ... ell
        for m in range(1, ell + 1):
            log_norm = 0.5 * (
                math.log(2 * ell + 1) - math.log(4 * math.pi)
                + gammaln(ell - m + 1) - gammaln(ell + m + 1)
            )
            norm = math.sqrt(2.0) * math.exp(log_norm)
            P = lpmv(m, ell, cos_theta)
            Y[idx] = norm * np.cos(m * phi) * P
            idx += 1

    assert idx == L
    return Y


def compute_vertex_area_weights(vertices_unit: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Barycentric vertex area weights on the sphere mesh, normalized to mean 1."""
    v = normalize_sphere_vertices(vertices_unit)
    f = np.asarray(faces, dtype=np.int64)
    tri = v[f]  # [F,3,3]
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
    weights = np.zeros(v.shape[0], dtype=np.float64)
    np.add.at(weights, f[:, 0], area / 3.0)
    np.add.at(weights, f[:, 1], area / 3.0)
    np.add.at(weights, f[:, 2], area / 3.0)
    weights = np.maximum(weights, np.percentile(weights[weights > 0], 1) if np.any(weights > 0) else 1.0)
    weights = weights / np.mean(weights)
    return weights


def project_features_to_spharm_fast_chunked(
    vertices_unit: np.ndarray,
    features: np.ndarray,
    degree: int,
    faces: Optional[np.ndarray] = None,
    chunk_size: int = 8192,
    use_area_weights: bool = True,
) -> np.ndarray:
    """
    Fast diagonal/quadrature projection of vertex features to SPHARM coefficients.

    This is much faster than full least-squares for FreeSurfer meshes. It uses

        c_l ~= sum_v w_v Y_l(v) f(v) / sum_v w_v Y_l(v)^2

    For a sufficiently uniform sphere mesh this is close to the continuous
    spherical harmonic projection, while avoiding the huge [L,L] normal matrix.
    """
    vertices = normalize_sphere_vertices(vertices_unit)
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"features must be [C,V], got {features.shape}")
    C, V = features.shape
    if vertices.shape[0] != V:
        raise ValueError(f"vertices/features mismatch: {vertices.shape[0]} vs {V}")

    L = (degree + 1) ** 2
    ytf = np.zeros((L, C), dtype=np.float64)
    y2 = np.zeros((L,), dtype=np.float64)

    if use_area_weights and faces is not None:
        w_all = compute_vertex_area_weights(vertices, faces)
    else:
        w_all = np.ones(V, dtype=np.float64)

    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        Y = real_spharm_basis_chunk(vertices[start:end], degree)  # [L, B]
        W = w_all[start:end][None, :]  # [1, B]
        F = features[:, start:end].T  # [B, C]
        ytf += (Y * W) @ F
        y2 += np.sum((Y * Y) * W, axis=1)

    coeffs = (ytf / np.maximum(y2[:, None], 1e-12)).T  # [C, L]
    return coeffs.astype(np.float32)


def project_features_to_spharm_chunked(
    vertices_unit: np.ndarray,
    features: np.ndarray,
    degree: int,
    faces: Optional[np.ndarray] = None,
    ridge: float = 1e-6,
    chunk_size: int = 4096,
    use_area_weights: bool = True,
) -> np.ndarray:
    """
    Weighted ridge least-squares projection of vertex features to SPHARM coefficients.

    This is more exact but very slow for degree >= 60 because it builds and
    solves a dense [L,L] system, where L=(degree+1)^2. Use --projection_mode fast
    for large-scale preprocessing.
    """
    vertices = normalize_sphere_vertices(vertices_unit)
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"features must be [C,V], got {features.shape}")
    C, V = features.shape
    if vertices.shape[0] != V:
        raise ValueError(f"vertices/features mismatch: {vertices.shape[0]} vs {V}")

    L = (degree + 1) ** 2
    xtx = np.zeros((L, L), dtype=np.float64)
    xtf = np.zeros((L, C), dtype=np.float64)

    if use_area_weights and faces is not None:
        w_all = compute_vertex_area_weights(vertices, faces)
    else:
        w_all = np.ones(V, dtype=np.float64)

    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        Y = real_spharm_basis_chunk(vertices[start:end], degree)  # [L, B]
        A = Y.T  # [B, L]
        F = features[:, start:end].T  # [B, C]
        sw = np.sqrt(w_all[start:end])[:, None]
        Aw = A * sw
        Fw = F * sw
        xtx += Aw.T @ Aw
        xtf += Aw.T @ Fw

    if ridge > 0:
        trace_scale = float(np.trace(xtx) / max(L, 1))
        xtx += (ridge * max(trace_scale, 1e-12)) * np.eye(L, dtype=np.float64)

    coeffs = np.linalg.solve(xtx, xtf).T  # [C, L]
    return coeffs.astype(np.float32)


def reconstruct_features_from_spharm_chunked(
    vertices_unit: np.ndarray,
    coeffs: np.ndarray,
    degree: int,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Reconstruct vertex maps from coefficients without storing full Y."""
    vertices = normalize_sphere_vertices(vertices_unit)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.ndim != 2:
        raise ValueError(f"coeffs must be [C,L], got {coeffs.shape}")
    C, L = coeffs.shape
    expected_L = (degree + 1) ** 2
    if L != expected_L:
        raise ValueError(f"coeff L mismatch: got {L}, expected {expected_L}")

    V = vertices.shape[0]
    out = np.zeros((C, V), dtype=np.float32)
    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        Y = real_spharm_basis_chunk(vertices[start:end], degree)  # [L, B]
        out[:, start:end] = (coeffs @ Y).astype(np.float32)
    return out


# ---------------------------------------------------------------------
# FreeSurfer preparation
# ---------------------------------------------------------------------
def find_freesurfer_surf_dirs(fs_root: str) -> List[Path]:
    root = Path(fs_root)
    if root.name == "surf" and root.is_dir():
        return [root]
    dirs = sorted([p for p in root.rglob("surf") if p.is_dir()])
    return dirs


def parse_subject_session_from_surf_dir(surf_dir: Path) -> Tuple[str, str]:
    # Expected: <root>/<subject>/<session>/surf
    session = surf_dir.parent.name if surf_dir.parent is not None else "unknown_session"
    subject = surf_dir.parent.parent.name if surf_dir.parent.parent is not None else surf_dir.parent.name
    return subject, session


def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def _canonical_csv_key(name: str) -> str:
    """Normalize CSV column names such as 'Image Data ID' -> 'image_data_id'."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _row_get(row: Dict[str, Any], aliases: Sequence[str], default: Any = "") -> Any:
    """Get a value from a normalized CSV row using multiple possible aliases."""
    for key in aliases:
        ckey = _canonical_csv_key(key)
        if ckey in row and str(row[ckey]).strip() != "":
            return row[ckey]
    return default


def _safe_float(value: Any, default: float = -1.0) -> float:
    try:
        if isinstance(value, np.ndarray):
            value = value.item() if value.shape == () else value.tolist()
        s = str(value).strip()
        if s == "" or s.lower() in {"nan", "none", "null", "unknown"}:
            return float(default)
        return float(s)
    except Exception:
        return float(default)


def load_metadata_csv(path: str) -> Dict[Any, Any]:
    """
    Optional demographics CSV metadata.

    This function is designed for ADNI-like CSV files with columns like:
      Image Data ID, Subject, Group, Sex, Age, Visit, Modality, Description, Acq Date, Days

    It also accepts lowercase/simple aliases:
      subject, session, image_data_id, group, sex, age, days, diagnosis

    FreeSurfer folder structure is expected to be:
      <fs_root>/<Subject>/<Image Data ID>/surf
    so metadata is matched mainly by (Subject, Image Data ID). If a subject has exactly
    one row, subject-only fallback is also allowed. If a subject has multiple rows and
    session/image id does not match, missing metadata defaults are used instead of using
    the wrong visit.
    """
    meta: Dict[Any, Any] = {"__rows_by_subject__": {}}
    if not path:
        return meta

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return meta
        for raw in reader:
            row_norm = {_canonical_csv_key(k): ("" if v is None else str(v).strip()) for k, v in raw.items()}

            subject = str(_row_get(row_norm, ["subject", "subject_id"], "")).strip()
            image_data_id = str(_row_get(row_norm, ["image_data_id", "image_data", "image_id", "session", "session_id"], "")).strip()
            if not subject:
                continue

            group = str(_row_get(row_norm, ["group", "diagnosis", "dx"], "unknown")).strip() or "unknown"
            sex = str(_row_get(row_norm, ["sex", "gender"], "unknown")).strip() or "unknown"
            age = _safe_float(_row_get(row_norm, ["age"], -1.0), -1.0)
            days = _safe_float(_row_get(row_norm, ["days", "day"], -1.0), -1.0)
            visit = str(_row_get(row_norm, ["visit", "viscode", "visit_code"], "unknown")).strip() or "unknown"
            acq_date = str(_row_get(row_norm, ["acq_date", "acquisition_date", "scan_date", "date"], "unknown")).strip() or "unknown"
            modality = str(_row_get(row_norm, ["modality"], "unknown")).strip() or "unknown"
            description = str(_row_get(row_norm, ["description", "desc"], "unknown")).strip() or "unknown"

            clean = dict(row_norm)
            clean.update(
                {
                    "subject": subject,
                    "subject_id": subject,
                    "session": image_data_id,
                    "session_id": image_data_id,
                    "image_data_id": image_data_id,
                    "group": group,
                    "diagnosis": group,  # backward-compatible alias
                    "sex": sex,
                    "age": age,
                    "days": days,
                    "visit": visit,
                    "acq_date": acq_date,
                    "modality": modality,
                    "description": description,
                }
            )
            if image_data_id:
                meta[(subject, image_data_id)] = clean
            meta["__rows_by_subject__"].setdefault(subject, []).append(clean)

    # Subject-only fallback only when there is one row for that subject.
    for subject, rows in meta.get("__rows_by_subject__", {}).items():
        if len(rows) == 1:
            meta[(subject, "")] = rows[0]

    n_rows = sum(len(v) for v in meta.get("__rows_by_subject__", {}).values())
    logger.info("loaded metadata rows=%d from %s", n_rows, path)
    return meta


def get_metadata_row(metadata: Dict[Any, Any], subject: str, session: str) -> Dict[str, Any]:
    """Return the metadata row that best matches (subject, session)."""
    if not metadata:
        return {}
    row = metadata.get((subject, session))
    if row is not None:
        return row
    row = metadata.get((subject, ""))
    if row is not None:
        return row
    return {}


def get_meta_value(
    metadata: Dict[Any, Any],
    subject: str,
    session: str,
    key: str,
    default: Any,
) -> Any:
    row = get_metadata_row(metadata, subject, session)
    return row.get(key, default)


def read_freesurfer_morph(path: Path) -> np.ndarray:
    require_nibabel()
    arr = fsio.read_morph_data(str(path)).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def read_freesurfer_sphere(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    require_nibabel()
    vertices, faces = fsio.read_geometry(str(path))
    return normalize_sphere_vertices(vertices), np.asarray(faces, dtype=np.int64)


def prepare_one_freesurfer_hemi(
    surf_dir: Path,
    hemi: str,
    out_path: Path,
    degree: int,
    sphere_file_template: str,
    feature_file_templates: Dict[str, str],
    ridge: float,
    chunk_size: int,
    use_area_weights: bool,
    projection_mode: str,
    metadata: Dict[Tuple[str, str], Dict[str, Any]],
) -> None:
    subject, session = parse_subject_session_from_surf_dir(surf_dir)
    sphere_path = surf_dir / sphere_file_template.format(hemi=hemi)
    if not sphere_path.exists():
        raise FileNotFoundError(f"Missing sphere file: {sphere_path}")

    vertices, faces = read_freesurfer_sphere(sphere_path)
    feature_arrays: List[np.ndarray] = []
    source_files: Dict[str, str] = {"sphere": str(sphere_path)}

    for feat_name in FEATURE_NAMES:
        template = feature_file_templates[feat_name]
        feat_path = surf_dir / template.format(hemi=hemi)
        if not feat_path.exists():
            raise FileNotFoundError(f"Missing feature file for {feat_name}: {feat_path}")
        feat = read_freesurfer_morph(feat_path)
        if feat.shape[0] != vertices.shape[0]:
            raise ValueError(
                f"Feature vertex count mismatch for {feat_path}: {feat.shape[0]} vs sphere {vertices.shape[0]}"
            )
        feature_arrays.append(feat)
        source_files[feat_name] = str(feat_path)

    features = np.stack(feature_arrays, axis=0)  # [3,V]
    if projection_mode == "exact":
        coeffs = project_features_to_spharm_chunked(
            vertices_unit=vertices,
            features=features,
            degree=degree,
            faces=faces,
            ridge=ridge,
            chunk_size=chunk_size,
            use_area_weights=use_area_weights,
        )
    elif projection_mode == "fast":
        coeffs = project_features_to_spharm_fast_chunked(
            vertices_unit=vertices,
            features=features,
            degree=degree,
            faces=faces,
            chunk_size=chunk_size,
            use_area_weights=use_area_weights,
        )
    else:
        raise ValueError(f"Unknown projection_mode={projection_mode}. Use fast or exact.")

    meta_row = get_metadata_row(metadata, subject, session)
    image_data_id = str(meta_row.get("image_data_id", session))
    group = str(meta_row.get("group", meta_row.get("diagnosis", "unknown")))
    sex = str(meta_row.get("sex", "unknown"))
    age_float = _safe_float(meta_row.get("age", -1.0), -1.0)
    days_float = _safe_float(meta_row.get("days", -1.0), -1.0)
    visit = str(meta_row.get("visit", "unknown"))
    acq_date = str(meta_row.get("acq_date", "unknown"))

    hemi_int = 0 if hemi == "lh" else 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        coeffs=coeffs.astype(np.float32),
        feature_names=np.asarray(FEATURE_NAMES),
        degree=np.asarray(degree, dtype=np.int64),
        hemi=np.asarray(hemi_int, dtype=np.int64),
        hemi_name=np.asarray(hemi),
        subject_id=np.asarray(subject),
        session_id=np.asarray(session),
        image_data_id=np.asarray(image_data_id),
        group=np.asarray(str(group)),
        diagnosis=np.asarray(str(group)),  # backward-compatible alias
        sex=np.asarray(str(sex)),
        age=np.asarray(age_float, dtype=np.float32),
        days=np.asarray(days_float, dtype=np.float32),
        visit=np.asarray(str(visit)),
        acq_date=np.asarray(str(acq_date)),
        surf_dir=np.asarray(str(surf_dir)),
        source_files=np.asarray(json.dumps(source_files)),
        n_vertices=np.asarray(vertices.shape[0], dtype=np.int64),
    )


def split_surf_dirs_by_subject(
    surf_dirs: List[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[set, set, set]:
    """Subject-level train/val/test split to avoid leakage across visits/hemispheres."""
    subjects = sorted({parse_subject_session_from_surf_dir(p)[0] for p in surf_dirs})
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    if n == 0:
        return set(), set(), set()
    if n == 1:
        return set(subjects), set(), set()

    train_ratio = float(train_ratio)
    val_ratio = float(val_ratio)
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("val_ratio must be >=0 and train_ratio + val_ratio < 1")

    n_train = max(1, int(round(n * train_ratio)))
    n_val = int(round(n * val_ratio)) if val_ratio > 0 else 0
    n_train = min(n_train, n)
    n_val = min(n_val, max(0, n - n_train))
    n_test = n - n_train - n_val

    # For n>=3, keep at least one validation and one test subject when possible.
    if n >= 3 and val_ratio > 0 and n_val == 0:
        n_val = 1
        n_train = max(1, n_train - 1)
        n_test = n - n_train - n_val
    if n >= 3 and n_test == 0:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        elif n_val > 0:
            n_val -= 1

    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train : n_train + n_val])
    test_subjects = set(subjects[n_train + n_val :])
    return train_subjects, val_subjects, test_subjects


def mode_prepare_fs(args: argparse.Namespace) -> None:
    metadata = load_metadata_csv(args.metadata_csv)
    surf_dirs = find_freesurfer_surf_dirs(args.fs_root)
    if args.max_subjects > 0:
        surf_dirs = surf_dirs[: args.max_subjects]
    if len(surf_dirs) == 0:
        raise RuntimeError(f"No surf directories found under {args.fs_root}")

    train_subjects, val_subjects, test_subjects = split_surf_dirs_by_subject(
        surf_dirs, args.train_ratio, args.val_ratio, args.seed
    )
    logger.info(
        "found %d surf dirs, train subjects=%d, val subjects=%d, test subjects=%d",
        len(surf_dirs), len(train_subjects), len(val_subjects), len(test_subjects)
    )

    feature_file_templates = {
        "CTh": args.cth_file,
        "K1": args.k1_file,
        "K2": args.k2_file,
    }

    n_saved = 0
    n_skipped = 0
    for surf_dir in surf_dirs:
        subject, session = parse_subject_session_from_surf_dir(surf_dir)
        if subject in train_subjects:
            split = "train"
        elif subject in val_subjects:
            split = "val"
        else:
            split = "test"
        for hemi in args.hemis.split(","):
            hemi = hemi.strip()
            if hemi not in ("lh", "rh"):
                continue
            out_name = f"{safe_filename(subject)}_{safe_filename(session)}_{hemi}_l{args.degree}.npz"
            out_path = Path(args.out_dir) / split / out_name
            if out_path.exists() and not args.overwrite:
                logger.info("skip existing: %s", out_path)
                n_skipped += 1
                continue
            try:
                logger.info("preparing %s %s -> %s", surf_dir, hemi, out_path)
                prepare_one_freesurfer_hemi(
                    surf_dir=surf_dir,
                    hemi=hemi,
                    out_path=out_path,
                    degree=args.degree,
                    sphere_file_template=args.sphere_file,
                    feature_file_templates=feature_file_templates,
                    ridge=args.ridge,
                    chunk_size=args.chunk_size,
                    use_area_weights=not args.no_area_weights,
                    projection_mode=args.projection_mode,
                    metadata=metadata,
                )
                n_saved += 1
            except Exception as exc:
                n_skipped += 1
                logger.warning("failed to prepare %s %s: %s", surf_dir, hemi, exc)
                if args.strict:
                    raise

    logger.info("prepare done: saved=%d skipped/failed=%d out_dir=%s", n_saved, n_skipped, args.out_dir)


# ---------------------------------------------------------------------
# Dataset and conditioning
# ---------------------------------------------------------------------
def parse_condition_keys(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def as_str(x: Any) -> str:
    if isinstance(x, np.ndarray):
        return str(x.item()) if x.shape == () else str(x.tolist())
    return str(x)


class BrainSPHARMDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        degree: int,
        condition_keys: Sequence[str],
        calc_norm: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        cond_schema: Optional[Dict[str, Any]] = None,
        std_floor_abs: float = 1e-5,
        std_floor_percentile: float = 1.0,
    ) -> None:
        self.data_dir = data_dir
        self.split = split
        self.degree = degree
        self.seq_length = (degree + 1) ** 2
        self.condition_keys = list(condition_keys)
        self.files = sorted(glob.glob(os.path.join(data_dir, split, "*.npz")))
        if len(self.files) == 0:
            raise RuntimeError(f"No prepared npz files found: {data_dir}/{split}/*.npz")

        coeffs: List[np.ndarray] = []
        metas: List[Dict[str, Any]] = []
        for p in self.files:
            d = np.load(p, allow_pickle=True)
            x = d["coeffs"].astype(np.float32)
            if x.shape != (3, self.seq_length):
                raise ValueError(f"Bad coeff shape in {p}: {x.shape}, expected {(3, self.seq_length)}")
            coeffs.append(x)
            metas.append(
                {
                    "path": p,
                    "subject_id": as_str(d.get("subject_id", "unknown")),
                    "session_id": as_str(d.get("session_id", "unknown")),
                    "image_data_id": as_str(d.get("image_data_id", d.get("session_id", "unknown"))),
                    "hemi": int(np.asarray(d.get("hemi", 0)).item()),
                    "age": _safe_float(d.get("age", -1.0), -1.0),
                    "days": _safe_float(d.get("days", -1.0), -1.0),
                    "sex": as_str(d.get("sex", "unknown")),
                    "group": as_str(d.get("group", d.get("diagnosis", "unknown"))),
                    "diagnosis": as_str(d.get("diagnosis", d.get("group", "unknown"))),
                    "visit": as_str(d.get("visit", "unknown")),
                    "acq_date": as_str(d.get("acq_date", "unknown")),
                    "surf_dir": as_str(d.get("surf_dir", "")),
                }
            )
        self.raw_coeffs = np.stack(coeffs, axis=0).astype(np.float32)  # [N,3,L]
        self.metas = metas

        if calc_norm:
            self.mean = self.raw_coeffs.mean(axis=0, keepdims=True).astype(np.float32)
            raw_std = self.raw_coeffs.std(axis=0, keepdims=True).astype(np.float32)
            floor = max(float(std_floor_abs), float(np.percentile(raw_std, std_floor_percentile)))
            self.std = np.maximum(raw_std, floor).astype(np.float32)
            logger.info("[%s] coeff norm: mean=%s std_floor=%.6e raw_std_min=%.6e", split, self.mean.shape, floor, float(raw_std.min()))
        else:
            if mean is None or std is None:
                raise ValueError("mean/std required when calc_norm=False")
            self.mean = mean.astype(np.float32)
            self.std = std.astype(np.float32)

        self.coeffs = ((self.raw_coeffs - self.mean) / self.std).astype(np.float32)

        if cond_schema is None:
            self.cond_schema = self._build_cond_schema(self.metas, self.condition_keys)
        else:
            self.cond_schema = cond_schema
        self.cond_dim = int(self.cond_schema.get("cond_dim", 0))
        self.cond_vecs = np.stack([self._meta_to_cond_vec(m, self.cond_schema) for m in self.metas], axis=0).astype(np.float32)

    @staticmethod
    def _canonical_condition_key(key: str) -> str:
        key = str(key).strip().lower()
        alias = {
            "subject": "subject_id",
            "subj": "subject_id",
            "subjectid": "subject_id",
            "subject_id": "subject_id",
            "group": "group",
            "diagnosis": "group",
            "dx": "group",
            "sex": "sex",
            "gender": "sex",
            "age": "age",
            "days": "days",
            "day": "days",
            "hemi": "hemi",
            "hemisphere": "hemi",
        }
        return alias.get(key, key)

    @staticmethod
    def _build_cond_schema(metas: List[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, Any]:
        canonical_keys: List[str] = []
        for k in keys:
            ck = BrainSPHARMDataset._canonical_condition_key(k)
            if ck not in canonical_keys:
                canonical_keys.append(ck)

        schema: Dict[str, Any] = {"keys": canonical_keys, "items": [], "cond_dim": 0}
        for key in canonical_keys:
            if key in ("age", "days"):
                vals = np.asarray([_safe_float(m.get(key, -1.0), -1.0) for m in metas if _safe_float(m.get(key, -1.0), -1.0) >= 0], dtype=np.float32)
                if vals.size == 0:
                    mean, std = 0.0, 1.0
                else:
                    mean, std = float(vals.mean()), float(vals.std() + 1e-6)
                schema["items"].append({"type": "continuous", "key": key, "mean": mean, "std": std, "dim": 1})
                schema["cond_dim"] += 1
            elif key == "hemi":
                schema["items"].append({"type": "categorical", "key": "hemi", "values": ["0", "1"], "dim": 2})
                schema["cond_dim"] += 2
            elif key in ("sex", "group", "subject_id"):
                values = sorted({str(m.get(key, "unknown")) for m in metas if str(m.get(key, "unknown")) != ""})
                if "unknown" not in values:
                    values.append("unknown")
                schema["items"].append({"type": "categorical", "key": key, "values": values, "dim": len(values)})
                schema["cond_dim"] += len(values)
            else:
                logger.warning("unknown condition key ignored: %s", key)
        return schema

    @staticmethod
    def _meta_to_cond_vec(meta: Dict[str, Any], schema: Dict[str, Any]) -> np.ndarray:
        parts: List[np.ndarray] = []
        for item in schema.get("items", []):
            key = item["key"]
            if item["type"] == "continuous":
                val = _safe_float(meta.get(key, -1.0), -1.0)
                if val < 0:
                    val = float(item["mean"])
                norm = (val - float(item["mean"])) / float(item["std"])
                parts.append(np.asarray([norm], dtype=np.float32))
            elif item["type"] == "categorical":
                values = [str(v) for v in item["values"]]
                if key == "hemi":
                    val = str(int(meta.get("hemi", 0)))
                else:
                    val = str(meta.get(key, "unknown"))
                if val == "":
                    val = "unknown"
                if val not in values:
                    val = "unknown" if "unknown" in values else values[0]
                onehot = np.zeros(len(values), dtype=np.float32)
                onehot[values.index(val)] = 1.0
                parts.append(onehot)
        if len(parts) == 0:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(parts, axis=0)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "coeffs": torch.from_numpy(self.coeffs[idx]),
            "cond": torch.from_numpy(self.cond_vecs[idx]),
            "index": torch.tensor(idx, dtype=torch.long),
        }


def collate_brain(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "coeffs": torch.stack([b["coeffs"] for b in batch], dim=0),
        "cond": torch.stack([b["cond"] for b in batch], dim=0),
        "index": torch.stack([b["index"] for b in batch], dim=0),
    }


def denormalize_coeffs(x_norm: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    mean_t = torch.from_numpy(mean).to(device=x_norm.device, dtype=x_norm.dtype)
    std_t = torch.from_numpy(std).to(device=x_norm.device, dtype=x_norm.dtype)
    return x_norm * std_t + mean_t


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
def num_groups_for_groupnorm(channels: int) -> int:
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb_scale = math.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb_scale)
        emb = x.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class FiLMResBlock1D(nn.Module):
    def __init__(self, channels: int, cond_dim: int, dilation: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        groups = num_groups_for_groupnorm(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.cond_proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        scale_shift = self.cond_proj(cond_emb).unsqueeze(-1)
        scale, shift = torch.chunk(scale_shift, 2, dim=1)
        h = self.norm1(x)
        h = h * (1.0 + scale) + shift
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return x + h


class BrainSPHARMResConvNet(nn.Module):
    """
    1D residual network over SPHARM coefficient index.

    x_t has feature_channels=3 for [CTh, K1, K2].
    For auxiliary cross-feature learning, the network also receives:
      known_values: [B,3,L]
      known_mask:   [B,3,L]
    Input to the first conv is concat([x_t, known_values, known_mask]) => 9 channels.
    During final generation known_values=known_mask=0.
    """

    def __init__(
        self,
        degree: int,
        feature_channels: int = 3,
        cond_input_dim: int = 0,
        channels: int = 256,
        depth: int = 18,
        cond_dim: int = 512,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.degree = degree
        self.seq_length = (degree + 1) ** 2
        self.feature_channels = feature_channels
        self.cond_input_dim = cond_input_dim
        self.register_buffer("degree_idx", spharm_degree_index(degree), persistent=False)

        self.in_conv = nn.Conv1d(feature_channels * 3, channels, kernel_size=1)
        self.degree_emb = nn.Embedding(degree + 1, channels)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(256),
            nn.Linear(256, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        if cond_input_dim > 0:
            self.condition_mlp = nn.Sequential(
                nn.Linear(cond_input_dim, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )
        else:
            self.condition_mlp = None

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        dilations_base = [1, 2, 4, 8, 16, 32, 64, 32, 16, 8, 4, 2]
        dilations = [dilations_base[i % len(dilations_base)] for i in range(depth)]
        self.blocks = nn.ModuleList(
            [FiLMResBlock1D(channels, cond_dim, dilation=d, dropout=dropout) for d in dilations]
        )
        self.out_norm = nn.GroupNorm(num_groups_for_groupnorm(channels), channels)
        self.out_conv = nn.Conv1d(channels, feature_channels, kernel_size=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        cond_vec: Optional[torch.Tensor] = None,
        known_values: Optional[torch.Tensor] = None,
        known_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, L = x.shape
        if C != self.feature_channels or L != self.seq_length:
            raise ValueError(f"Expected x [B,{self.feature_channels},{self.seq_length}], got {tuple(x.shape)}")
        if known_values is None:
            known_values = torch.zeros_like(x)
        if known_mask is None:
            known_mask = torch.zeros_like(x)

        t_emb = self.time_mlp(time)
        if self.condition_mlp is not None and cond_vec is not None and cond_vec.shape[-1] > 0:
            c_emb = self.condition_mlp(cond_vec.float())
        else:
            c_emb = torch.zeros_like(t_emb)
        cond_emb = self.cond_mlp(torch.cat([t_emb, c_emb], dim=-1))

        h_in = torch.cat([x, known_values, known_mask], dim=1)
        h = self.in_conv(h_in)
        deg_emb = self.degree_emb(self.degree_idx.to(x.device)).transpose(0, 1).unsqueeze(0)
        h = h + deg_emb
        for block in self.blocks:
            h = block(h, cond_emb)
        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


# ---------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------
def cosine_beta_schedule(timesteps: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-8, max_beta).float()


class BrainSPHARMDiffusion(nn.Module):
    def __init__(
        self,
        model: BrainSPHARMResConvNet,
        degree: int,
        num_timesteps: int = 1000,
        cond_drop_prob: float = 0.10,
        low_freq_boost: float = 4.0,
        x0_loss_weight: float = 0.25,
        aux_channel_mask_prob: float = 0.30,
        known_channel_loss_weight: float = 0.10,
        high_t_prob: float = 0.35,
    ) -> None:
        super().__init__()
        self.model = model
        self.degree = degree
        self.seq_length = (degree + 1) ** 2
        self.feature_channels = model.feature_channels
        self.num_timesteps = num_timesteps
        self.cond_drop_prob = cond_drop_prob
        self.x0_loss_weight = x0_loss_weight
        self.aux_channel_mask_prob = aux_channel_mask_prob
        self.known_channel_loss_weight = known_channel_loss_weight
        self.high_t_prob = high_t_prob

        betas = cosine_beta_schedule(num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("loss_weights", spharm_loss_weights(degree, low_freq_boost=low_freq_boost))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        s = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return a * x_start + s * noise

    def get_v_target(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        s = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return a * noise - s * x_start

    def v_to_x0_eps(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        s = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        pred_x0 = a * x_t - s * v
        pred_eps = s * x_t + a * v
        return pred_x0, pred_eps

    def make_known_channel_mask(self, x_start: torch.Tensor) -> torch.Tensor:
        B, C, L = x_start.shape
        mask = torch.zeros(B, C, 1, device=x_start.device, dtype=x_start.dtype)
        if self.aux_channel_mask_prob <= 0:
            return mask.expand(-1, -1, L)
        for b in range(B):
            if torch.rand((), device=x_start.device) < self.aux_channel_mask_prob:
                # Give one or two clean feature channels as auxiliary context.
                n_known = 1 if torch.rand((), device=x_start.device) < 0.5 else 2
                perm = torch.randperm(C, device=x_start.device)[:n_known]
                mask[b, perm, 0] = 1.0
        return mask.expand(-1, -1, L)

    def forward(self, x_start: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        B = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=x_start.device).long()
        if self.high_t_prob > 0:
            high_mask = torch.rand(B, device=x_start.device) < self.high_t_prob
            if high_mask.any():
                t_high = torch.randint(
                    int(self.num_timesteps * 0.55), self.num_timesteps,
                    (int(high_mask.sum().item()),), device=x_start.device
                ).long()
                t[high_mask] = t_high

        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        v_target = self.get_v_target(x_start, t, noise)

        known_mask = self.make_known_channel_mask(x_start)
        known_values = x_start * known_mask
        loss_channel_weight = torch.where(
            known_mask > 0,
            torch.full_like(known_mask, self.known_channel_loss_weight),
            torch.ones_like(known_mask),
        )

        train_cond = cond_vec
        if self.cond_drop_prob > 0 and cond_vec.shape[-1] > 0:
            drop = (torch.rand(B, device=x_start.device) < self.cond_drop_prob).float().view(B, 1)
            train_cond = cond_vec * (1.0 - drop)

        pred_v = self.model(x_t, t, cond_vec=train_cond, known_values=known_values, known_mask=known_mask)
        pred_x0, _ = self.v_to_x0_eps(x_t, t, pred_v)

        v_loss = F.mse_loss(pred_v, v_target, reduction="none")
        v_loss = (v_loss * self.loss_weights * loss_channel_weight).mean()
        x0_loss = F.smooth_l1_loss(pred_x0, x_start, reduction="none", beta=0.5)
        x0_loss = (x0_loss * self.loss_weights * loss_channel_weight).mean()
        return v_loss + self.x0_loss_weight * x0_loss

    def _predict_v_cfg(
        self,
        model: BrainSPHARMResConvNet,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_vec: torch.Tensor,
        cfg_scale: float,
        known_values: torch.Tensor,
        known_mask: torch.Tensor,
    ) -> torch.Tensor:
        if cfg_scale <= 1.0001 or cond_vec.shape[-1] == 0:
            return model(x, t, cond_vec=cond_vec, known_values=known_values, known_mask=known_mask)
        null_cond = torch.zeros_like(cond_vec)
        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        cond_in = torch.cat([null_cond, cond_vec], dim=0)
        kv_in = torch.cat([known_values, known_values], dim=0)
        km_in = torch.cat([known_mask, known_mask], dim=0)
        v_uncond, v_cond = model(x_in, t_in, cond_vec=cond_in, known_values=kv_in, known_mask=km_in).chunk(2, dim=0)
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    @torch.no_grad()
    def sample(
        self,
        cond_vec: torch.Tensor,
        model: Optional[BrainSPHARMResConvNet] = None,
        steps: int = 120,
        cfg_scale: float = 2.0,
        eta: float = 0.0,
        clip_x0: float = 4.0,
        lowpass_cut_l: Optional[int] = 58,
        lowpass_tau: float = 12.0,
        known_values: Optional[torch.Tensor] = None,
        known_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if model is None:
            model = self.model
        model.eval()
        device = next(model.parameters()).device
        cond_vec = cond_vec.to(device)
        B = cond_vec.shape[0]
        x = torch.randn(B, self.feature_channels, self.seq_length, device=device)
        if known_values is None:
            known_values = torch.zeros_like(x)
        else:
            known_values = known_values.to(device)
        if known_mask is None:
            known_mask = torch.zeros_like(x)
        else:
            known_mask = known_mask.to(device)

        steps = int(min(max(2, steps), self.num_timesteps))
        time_indices = torch.linspace(self.num_timesteps - 1, 0, steps, device=device).long()
        time_indices = torch.unique_consecutive(time_indices)
        lowpass = spharm_lowpass_mask(self.degree, lowpass_cut_l, lowpass_tau).to(device)

        for i, t_int in enumerate(time_indices):
            t = torch.full((B,), int(t_int.item()), device=device, dtype=torch.long)
            v = self._predict_v_cfg(model, x, t, cond_vec, cfg_scale, known_values, known_mask)
            pred_x0, pred_eps = self.v_to_x0_eps(x, t, v)
            if clip_x0 is not None and clip_x0 > 0:
                pred_x0 = pred_x0.clamp(-clip_x0, clip_x0)
            pred_x0 = pred_x0 * lowpass
            # Keep known channels fixed when partial-conditioning is used.
            pred_x0 = pred_x0 * (1.0 - known_mask) + known_values * known_mask

            if i == len(time_indices) - 1:
                x = pred_x0
                break
            next_t_int = int(time_indices[i + 1].item())
            alpha_t = self.alphas_cumprod[int(t_int.item())]
            alpha_next = self.alphas_cumprod[next_t_int]
            if eta <= 0:
                x = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1.0 - alpha_next) * pred_eps
            else:
                sigma = eta * torch.sqrt((1.0 - alpha_next) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_next))
                sigma = torch.clamp(sigma, min=0.0)
                noise = torch.randn_like(x)
                direction = torch.sqrt(torch.clamp(1.0 - alpha_next - sigma**2, min=0.0)) * pred_eps
                x = torch.sqrt(alpha_next) * pred_x0 + direction + sigma * noise
            x = x * (1.0 - known_mask) + known_values * known_mask
        return x


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, ema_v in self.module.state_dict().items():
            model_v = msd[k].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.copy_(ema_v * self.decay + model_v * (1.0 - self.decay))
            else:
                ema_v.copy_(model_v)


# ---------------------------------------------------------------------
# Train / sample helpers
# ---------------------------------------------------------------------
@dataclass
class TrainConfig:
    data_dir: str
    degree: int = 60
    condition_keys: str = "subject,group,sex,age,days,hemi"
    val_split: str = "val"
    epochs: int = 1000
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-4
    num_timesteps: int = 1000
    sample_steps: int = 120
    channels: int = 256
    depth: int = 18
    cond_dim: int = 512
    dropout: float = 0.05
    cond_drop_prob: float = 0.10
    low_freq_boost: float = 4.0
    x0_loss_weight: float = 0.25
    aux_channel_mask_prob: float = 0.30
    known_channel_loss_weight: float = 0.10
    high_t_prob: float = 0.35
    ema_decay: float = 0.9995
    grad_clip: float = 1.0
    num_workers: int = 4
    std_floor_abs: float = 1e-5
    std_floor_percentile: float = 1.0
    cfg_scale: float = 2.0
    clip_x0: float = 4.0
    lowpass_cut_l: int = 58
    lowpass_tau: float = 12.0
    validation_every: int = 25
    sample_every: int = 25
    save_every: int = 50
    samples_per_epoch: int = 8
    output_dir: str = "./brain_spharm_runs"
    resume: str = ""
    seed: int = 42
    amp: bool = True


def save_checkpoint(
    path: str,
    diffusion: BrainSPHARMDiffusion,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_dataset: BrainSPHARMDataset,
    cfg: TrainConfig,
    epoch: int,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "config": asdict(cfg),
            "model_state": diffusion.model.state_dict(),
            "diffusion_state": diffusion.state_dict(),
            "ema_state": ema.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "mean": train_dataset.mean,
            "std": train_dataset.std,
            "cond_schema": train_dataset.cond_schema,
            "feature_names": FEATURE_NAMES,
        },
        path,
    )
    logger.info("checkpoint saved: %s", path)


@torch.no_grad()
def validate_coeff_mae(
    diffusion: BrainSPHARMDiffusion,
    sample_model: BrainSPHARMResConvNet,
    val_loader: DataLoader,
) -> float:
    device = next(sample_model.parameters()).device
    sample_model.eval()
    total = 0.0
    count = 0
    for batch in val_loader:
        x = batch["coeffs"].to(device)
        cond = batch["cond"].to(device)
        B = x.shape[0]
        t = torch.full((B,), diffusion.num_timesteps // 2, device=device, dtype=torch.long)
        noise = torch.randn_like(x)
        x_t = diffusion.q_sample(x, t, noise)
        known = torch.zeros_like(x)
        known_mask = torch.zeros_like(x)
        pred_v = sample_model(x_t, t, cond_vec=cond, known_values=known, known_mask=known_mask)
        pred_x0, _ = diffusion.v_to_x0_eps(x_t, t, pred_v)
        total += F.l1_loss(pred_x0, x, reduction="sum").item()
        count += x.numel()
    mae = total / max(count, 1)
    logger.info("validation normalized coefficient MAE=%.8f", mae)
    return mae


@torch.no_grad()
def save_epoch_samples(
    diffusion: BrainSPHARMDiffusion,
    sample_model: BrainSPHARMResConvNet,
    dataset: BrainSPHARMDataset,
    epoch: int,
    out_dir: str,
    n: int,
    sample_steps: int,
    cfg_scale: float,
    lowpass_cut_l: Optional[int],
    lowpass_tau: float,
    clip_x0: float,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    device = next(sample_model.parameters()).device
    n = min(n, len(dataset))
    idxs = np.linspace(0, len(dataset) - 1, n, dtype=int)
    cond = torch.from_numpy(dataset.cond_vecs[idxs]).float().to(device)
    x_norm = diffusion.sample(
        cond_vec=cond,
        model=sample_model,
        steps=sample_steps,
        cfg_scale=cfg_scale,
        eta=0.0,
        clip_x0=clip_x0,
        lowpass_cut_l=lowpass_cut_l,
        lowpass_tau=lowpass_tau,
    )
    x_orig = denormalize_coeffs(x_norm, dataset.mean, dataset.std).cpu().numpy()
    for j, idx in enumerate(idxs):
        meta = dataset.metas[int(idx)]
        save_path = os.path.join(out_dir, f"epoch{epoch:04d}_{j:03d}_{meta['subject_id']}_{meta['session_id']}_hemi{meta['hemi']}.npz")
        np.savez_compressed(
            save_path,
            coeffs=x_orig[j].astype(np.float32),
            feature_names=np.asarray(FEATURE_NAMES),
            degree=np.asarray(dataset.degree),
            cond_vec=dataset.cond_vecs[int(idx)],
            cond_schema=np.asarray(json.dumps(dataset.cond_schema)),
            source_subject=np.asarray(meta["subject_id"]),
            source_session=np.asarray(meta["session_id"]),
            image_data_id=np.asarray(meta.get("image_data_id", meta["session_id"])),
            group=np.asarray(meta.get("group", "unknown")),
            sex=np.asarray(meta.get("sex", "unknown")),
            days=np.asarray(meta.get("days", -1.0), dtype=np.float32),
            hemi=np.asarray(meta["hemi"], dtype=np.int64),
            age=np.asarray(meta["age"], dtype=np.float32),
        )
    logger.info("saved epoch samples: epoch=%d n=%d dir=%s", epoch, n, out_dir)


def mode_train(args: argparse.Namespace) -> None:
    cfg = TrainConfig(
        data_dir=args.data_dir,
        degree=args.degree,
        condition_keys=args.condition_keys,
        val_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_timesteps=args.num_timesteps,
        sample_steps=args.sample_steps,
        channels=args.channels,
        depth=args.depth,
        cond_dim=args.cond_dim,
        dropout=args.dropout,
        cond_drop_prob=args.cond_drop_prob,
        low_freq_boost=args.low_freq_boost,
        x0_loss_weight=args.x0_loss_weight,
        aux_channel_mask_prob=args.aux_channel_mask_prob,
        known_channel_loss_weight=args.known_channel_loss_weight,
        high_t_prob=args.high_t_prob,
        ema_decay=args.ema_decay,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        std_floor_abs=args.std_floor_abs,
        std_floor_percentile=args.std_floor_percentile,
        cfg_scale=args.cfg_scale,
        clip_x0=args.clip_x0,
        lowpass_cut_l=args.lowpass_cut_l,
        lowpass_tau=args.lowpass_tau,
        validation_every=args.validation_every,
        sample_every=args.sample_every,
        save_every=args.save_every,
        samples_per_epoch=args.samples_per_epoch,
        output_dir=args.output_dir,
        resume=args.resume,
        seed=args.seed,
        amp=not args.no_amp,
    )
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s config=%s", device, cfg)

    condition_keys = parse_condition_keys(cfg.condition_keys)
    train_dataset = BrainSPHARMDataset(
        cfg.data_dir,
        "train",
        degree=cfg.degree,
        condition_keys=condition_keys,
        calc_norm=True,
        std_floor_abs=cfg.std_floor_abs,
        std_floor_percentile=cfg.std_floor_percentile,
    )
    val_split = cfg.val_split
    if not glob.glob(os.path.join(cfg.data_dir, val_split, "*.npz")):
        logger.warning("validation split '%s' is empty or missing; falling back to test split", val_split)
        val_split = "test"
    val_dataset = BrainSPHARMDataset(
        cfg.data_dir,
        val_split,
        degree=cfg.degree,
        condition_keys=condition_keys,
        calc_norm=False,
        mean=train_dataset.mean,
        std=train_dataset.std,
        cond_schema=train_dataset.cond_schema,
        std_floor_abs=cfg.std_floor_abs,
        std_floor_percentile=cfg.std_floor_percentile,
    )
    logger.info("condition schema: %s", train_dataset.cond_schema)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_brain,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_brain,
    )

    model = BrainSPHARMResConvNet(
        degree=cfg.degree,
        feature_channels=3,
        cond_input_dim=train_dataset.cond_dim,
        channels=cfg.channels,
        depth=cfg.depth,
        cond_dim=cfg.cond_dim,
        dropout=cfg.dropout,
    ).to(device)
    diffusion = BrainSPHARMDiffusion(
        model=model,
        degree=cfg.degree,
        num_timesteps=cfg.num_timesteps,
        cond_drop_prob=cfg.cond_drop_prob,
        low_freq_boost=cfg.low_freq_boost,
        x0_loss_weight=cfg.x0_loss_weight,
        aux_channel_mask_prob=cfg.aux_channel_mask_prob,
        known_channel_loss_weight=cfg.known_channel_loss_weight,
        high_t_prob=cfg.high_t_prob,
    ).to(device)
    ema = ModelEMA(diffusion.model, decay=cfg.ema_decay)
    ema.module.to(device)

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.05)
    scaler = GradScaler(enabled=cfg.amp and device.type == "cuda")

    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
    sample_dir = os.path.join(cfg.output_dir, "generated_coeffs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    start_epoch = 1
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        diffusion.model.load_state_dict(ckpt["model_state"], strict=True)
        if "ema_state" in ckpt:
            ema.module.load_state_dict(ckpt["ema_state"], strict=True)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        logger.info("resumed from %s at epoch %d", cfg.resume, start_epoch)

    lowpass_cut_l = None if cfg.lowpass_cut_l < 0 else cfg.lowpass_cut_l
    logger.info("training start")
    for epoch in range(start_epoch, cfg.epochs + 1):
        diffusion.train()
        running = 0.0
        nb = 0
        for batch in train_loader:
            x = batch["coeffs"].to(device, non_blocking=True)
            cond = batch["cond"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.amp and device.type == "cuda"):
                loss = diffusion(x, cond)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(diffusion.model)
            running += float(loss.item())
            nb += 1

        scheduler.step()
        avg = running / max(nb, 1)
        logger.info("epoch [%d/%d] loss=%.8f lr=%.3e", epoch, cfg.epochs, avg, scheduler.get_last_lr()[0])

        if epoch % cfg.validation_every == 0:
            validate_coeff_mae(diffusion, ema.module, val_loader)
        if epoch % cfg.sample_every == 0:
            save_epoch_samples(
                diffusion, ema.module, val_dataset, epoch, sample_dir, cfg.samples_per_epoch,
                cfg.sample_steps, cfg.cfg_scale, lowpass_cut_l, cfg.lowpass_tau, cfg.clip_x0,
            )
        if epoch % cfg.save_every == 0:
            save_checkpoint(os.path.join(ckpt_dir, f"epoch{epoch:04d}.pt"), diffusion, ema, optimizer, scheduler, train_dataset, cfg, epoch)

    save_checkpoint(os.path.join(ckpt_dir, "final.pt"), diffusion, ema, optimizer, scheduler, train_dataset, cfg, cfg.epochs)
    logger.info("training complete")


def build_cond_vec_from_args(cond_schema: Dict[str, Any], args: argparse.Namespace, n: int) -> np.ndarray:
    metas = []
    for _ in range(n):
        metas.append(
            {
                "subject_id": str(args.subject),
                "group": str(args.group),
                "diagnosis": str(args.group),
                "sex": str(args.sex),
                "age": float(args.age),
                "days": float(args.days),
                "hemi": int(args.hemi),
            }
        )
    vecs = [BrainSPHARMDataset._meta_to_cond_vec(m, cond_schema) for m in metas]
    return np.stack(vecs, axis=0).astype(np.float32)


def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[BrainSPHARMDiffusion, BrainSPHARMResConvNet, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt["config"]
    degree = int(cfg_dict["degree"])
    cond_schema = ckpt.get("cond_schema", {"cond_dim": 0, "items": []})
    cond_dim_in = int(cond_schema.get("cond_dim", 0))
    model = BrainSPHARMResConvNet(
        degree=degree,
        feature_channels=3,
        cond_input_dim=cond_dim_in,
        channels=int(cfg_dict.get("channels", 256)),
        depth=int(cfg_dict.get("depth", 18)),
        cond_dim=int(cfg_dict.get("cond_dim", 512)),
        dropout=float(cfg_dict.get("dropout", 0.05)),
    ).to(device)
    diffusion = BrainSPHARMDiffusion(
        model=model,
        degree=degree,
        num_timesteps=int(cfg_dict.get("num_timesteps", 1000)),
        cond_drop_prob=float(cfg_dict.get("cond_drop_prob", 0.10)),
        low_freq_boost=float(cfg_dict.get("low_freq_boost", 4.0)),
        x0_loss_weight=float(cfg_dict.get("x0_loss_weight", 0.25)),
        aux_channel_mask_prob=float(cfg_dict.get("aux_channel_mask_prob", 0.0)),
        known_channel_loss_weight=float(cfg_dict.get("known_channel_loss_weight", 0.10)),
        high_t_prob=float(cfg_dict.get("high_t_prob", 0.0)),
    ).to(device)
    if "ema_state" in ckpt:
        model.load_state_dict(ckpt["ema_state"], strict=True)
    else:
        model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    diffusion.eval()
    return diffusion, model, ckpt


def mode_sample(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    diffusion, model, ckpt = load_model_from_checkpoint(args.checkpoint, device)
    cond_schema = ckpt.get("cond_schema", {"cond_dim": 0, "items": []})
    cond_vec = torch.from_numpy(build_cond_vec_from_args(cond_schema, args, args.n)).float().to(device)
    lowpass_cut_l = None if args.lowpass_cut_l < 0 else args.lowpass_cut_l
    x_norm = diffusion.sample(
        cond_vec=cond_vec,
        model=model,
        steps=args.sample_steps,
        cfg_scale=args.cfg_scale,
        eta=args.eta,
        clip_x0=args.clip_x0,
        lowpass_cut_l=lowpass_cut_l,
        lowpass_tau=args.lowpass_tau,
    )
    x_orig = denormalize_coeffs(x_norm, ckpt["mean"], ckpt["std"]).cpu().numpy()
    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(args.n):
        path = os.path.join(args.out_dir, f"sample_{i:03d}.npz")
        np.savez_compressed(
            path,
            coeffs=x_orig[i].astype(np.float32),
            feature_names=np.asarray(FEATURE_NAMES),
            degree=np.asarray(int(ckpt["config"]["degree"]), dtype=np.int64),
            subject_id=np.asarray(str(args.subject)),
            group=np.asarray(str(args.group)),
            diagnosis=np.asarray(str(args.group)),
            sex=np.asarray(str(args.sex)),
            age=np.asarray(float(args.age), dtype=np.float32),
            days=np.asarray(float(args.days), dtype=np.float32),
            hemi=np.asarray(int(args.hemi), dtype=np.int64),
            cond_vec=cond_vec[i].cpu().numpy(),
            cond_schema=np.asarray(json.dumps(cond_schema)),
        )
    logger.info("saved %d samples to %s", args.n, args.out_dir)


def mode_reconstruct(args: argparse.Namespace) -> None:
    require_nibabel()
    coeff_npz = np.load(args.coeff_npz, allow_pickle=True)
    coeffs = coeff_npz["coeffs"].astype(np.float32)
    degree = int(args.degree if args.degree >= 0 else np.asarray(coeff_npz.get("degree", 60)).item())

    if args.surf_dir:
        sphere_path = Path(args.surf_dir) / args.sphere_file.format(hemi=args.hemi)
    elif args.sphere_path:
        sphere_path = Path(args.sphere_path)
    else:
        raise ValueError("Provide either --surf_dir or --sphere_path")
    vertices, faces = read_freesurfer_sphere(sphere_path)
    maps = reconstruct_features_from_spharm_chunked(vertices, coeffs, degree, chunk_size=args.chunk_size)

    out_npz = args.out_npz
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(
        out_npz,
        maps=maps.astype(np.float32),
        CTh=maps[0].astype(np.float32),
        K1=maps[1].astype(np.float32),
        K2=maps[2].astype(np.float32),
        feature_names=np.asarray(FEATURE_NAMES),
        sphere_path=np.asarray(str(sphere_path)),
        coeff_npz=np.asarray(str(args.coeff_npz)),
        degree=np.asarray(degree, dtype=np.int64),
    )
    logger.info("saved reconstructed vertex maps: %s", out_npz)

    if args.write_morph_dir:
        out_dir = Path(args.write_morph_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fsio.write_morph_data(str(out_dir / f"{args.hemi}.generated.thickness"), maps[0].astype(np.float32))
        fsio.write_morph_data(str(out_dir / f"{args.hemi}.generated.smoothwm.K1.crv"), maps[1].astype(np.float32))
        fsio.write_morph_data(str(out_dir / f"{args.hemi}.generated.smoothwm.K2.crv"), maps[2].astype(np.float32))
        logger.info("wrote FreeSurfer morph files to: %s", out_dir)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def add_prepare_fs_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("prepare_fs", help="Convert FreeSurfer surf folders to SPHARM coefficient npz files")
    p.add_argument("--fs_root", type=str, required=True, help="FreeSurfer root or a single surf directory")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--degree", type=int, default=60)
    p.add_argument("--hemis", type=str, default="lh,rh")
    p.add_argument("--sphere_file", type=str, default="{hemi}.sphere.reg", help="Usually {hemi}.sphere.reg")
    p.add_argument("--cth_file", type=str, default=DEFAULT_FEATURE_FILES["CTh"])
    p.add_argument("--k1_file", type=str, default=DEFAULT_FEATURE_FILES["K1"])
    p.add_argument("--k2_file", type=str, default=DEFAULT_FEATURE_FILES["K2"])
    p.add_argument("--metadata_csv", type=str, default="", help="Optional ADNI CSV with Image Data ID, Subject, Group, Sex, Age, Days")
    p.add_argument("--train_ratio", type=float, default=0.80)
    p.add_argument("--val_ratio", type=float, default=0.10, help="Subject-level validation ratio; test ratio is 1-train-val")
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--projection_mode", type=str, default="fast", choices=["fast", "exact"], help="fast=diagonal quadrature projection, exact=full ridge least-squares but slow")
    p.add_argument("--chunk_size", type=int, default=8192)
    p.add_argument("--no_area_weights", action="store_true")
    p.add_argument("--max_subjects", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--strict", action="store_true", help="Raise error on first failed subject instead of skipping")
    p.set_defaults(func=mode_prepare_fs)


def add_train_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("train", help="Train brain SPHARM coefficient diffusion")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--degree", type=int, default=60)
    p.add_argument("--condition_keys", type=str, default="subject,group,sex,age,days,hemi", help="comma list: subject,group,sex,age,days,hemi")
    p.add_argument("--val_split", type=str, default="val", help="Validation split folder name; falls back to test if missing")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_timesteps", type=int, default=1000)
    p.add_argument("--sample_steps", type=int, default=120)
    p.add_argument("--channels", type=int, default=256)
    p.add_argument("--depth", type=int, default=18)
    p.add_argument("--cond_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--cond_drop_prob", type=float, default=0.10)
    p.add_argument("--low_freq_boost", type=float, default=4.0)
    p.add_argument("--x0_loss_weight", type=float, default=0.25)
    p.add_argument("--aux_channel_mask_prob", type=float, default=0.30)
    p.add_argument("--known_channel_loss_weight", type=float, default=0.10)
    p.add_argument("--high_t_prob", type=float, default=0.35)
    p.add_argument("--ema_decay", type=float, default=0.9995)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--std_floor_abs", type=float, default=1e-5)
    p.add_argument("--std_floor_percentile", type=float, default=1.0)
    p.add_argument("--cfg_scale", type=float, default=2.0)
    p.add_argument("--clip_x0", type=float, default=4.0)
    p.add_argument("--lowpass_cut_l", type=int, default=58, help="-1 disables low-pass")
    p.add_argument("--lowpass_tau", type=float, default=12.0)
    p.add_argument("--validation_every", type=int, default=25)
    p.add_argument("--sample_every", type=int, default=25)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--samples_per_epoch", type=int, default=8)
    p.add_argument("--output_dir", type=str, default="./brain_spharm_runs")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true")
    p.set_defaults(func=mode_train)


def add_sample_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("sample", help="Sample generated CTh/K1/K2 coefficient npz files")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--subject", type=str, default="unknown")
    p.add_argument("--group", type=str, default="unknown", help="e.g. CN, MCI, AD")
    p.add_argument("--sex", type=str, default="unknown", help="e.g. M or F")
    p.add_argument("--age", type=float, default=-1.0)
    p.add_argument("--days", type=float, default=-1.0)
    p.add_argument("--hemi", type=int, default=0, help="0=lh, 1=rh")
    p.add_argument("--sample_steps", type=int, default=120)
    p.add_argument("--cfg_scale", type=float, default=2.0)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--clip_x0", type=float, default=4.0)
    p.add_argument("--lowpass_cut_l", type=int, default=58)
    p.add_argument("--lowpass_tau", type=float, default=12.0)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=mode_sample)


def add_reconstruct_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("reconstruct", help="Reconstruct generated coefficients to vertex maps on a FreeSurfer sphere")
    p.add_argument("--coeff_npz", type=str, required=True)
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--degree", type=int, default=-1, help="Use coeff file degree if -1")
    p.add_argument("--surf_dir", type=str, default="")
    p.add_argument("--sphere_path", type=str, default="")
    p.add_argument("--sphere_file", type=str, default="{hemi}.sphere.reg")
    p.add_argument("--hemi", type=str, default="lh")
    p.add_argument("--chunk_size", type=int, default=4096)
    p.add_argument("--write_morph_dir", type=str, default="", help="Optional output dir for FreeSurfer morph files")
    p.set_defaults(func=mode_reconstruct)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FreeSurfer brain SPHARM coefficient diffusion")
    sub = p.add_subparsers(dest="mode", required=True)
    add_prepare_fs_parser(sub)
    add_train_parser(sub)
    add_sample_parser(sub)
    add_reconstruct_parser(sub)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
