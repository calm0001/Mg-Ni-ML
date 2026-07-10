from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    ParameterGrid,
    RandomizedSearchCV,
    train_test_split,
    cross_val_score,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor

    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False
    CatBoostRegressor = None

try:
    import shap

    HAS_SHAP = True
except Exception:
    HAS_SHAP = False
    shap = None

try:
    from PIL import Image

    HAS_PIL = True
except Exception:
    HAS_PIL = False
    Image = None


RANDOM_STATE = 42

# Element: atomic_radius_pm, Pauling electronegativity, valence_e,
# atomic_volume_A3, atomic_mass, atomic_number.
ELEMENT_PROPS = {
    "H": (53, 2.20, 1, 14.4, 1.008, 1),
    "Li": (152, 0.98, 1, 21.3, 6.941, 3),
    "Be": (112, 1.57, 2, 8.0, 9.012, 4),
    "B": (87, 2.04, 3, 7.24, 10.811, 5),
    "C": (77, 2.55, 4, 8.71, 12.011, 6),
    "N": (75, 3.04, 5, 17.3, 14.007, 7),
    "O": (73, 3.44, 6, 14.0, 15.999, 8),
    "F": (71, 3.98, 7, 17.1, 18.998, 9),
    "Na": (186, 0.93, 1, 37.7, 22.990, 11),
    "Mg": (160, 1.31, 2, 23.0, 24.305, 12),
    "Al": (143, 1.61, 3, 16.6, 26.982, 13),
    "Si": (111, 1.90, 4, 21.2, 28.086, 14),
    "P": (110, 2.19, 5, 22.3, 30.974, 15),
    "S": (104, 2.58, 6, 24.5, 32.065, 16),
    "K": (227, 0.82, 1, 71.3, 39.098, 19),
    "Ca": (197, 1.00, 2, 43.7, 40.078, 20),
    "Sc": (162, 1.36, 3, 25.0, 44.956, 21),
    "Ti": (147, 1.54, 4, 14.1, 47.867, 22),
    "V": (134, 1.63, 5, 13.1, 50.942, 23),
    "Cr": (128, 1.66, 6, 12.0, 51.996, 24),
    "Mn": (127, 1.55, 2, 12.0, 54.938, 25),
    "Fe": (126, 1.83, 2, 11.8, 55.845, 26),
    "Co": (125, 1.88, 2, 11.1, 58.933, 27),
    "Ni": (124, 1.91, 2, 10.9, 58.693, 28),
    "Cu": (128, 1.90, 1, 11.8, 63.546, 29),
    "Zn": (134, 1.65, 2, 13.9, 65.380, 30),
    "Ga": (136, 1.81, 3, 18.9, 69.723, 31),
    "Ge": (122, 2.01, 4, 22.6, 72.640, 32),
    "Se": (120, 2.55, 6, 25.8, 78.960, 34),
    "Rb": (248, 0.82, 1, 87.0, 85.468, 37),
    "Sr": (215, 0.95, 2, 55.3, 87.620, 38),
    "Y": (180, 1.22, 3, 19.9, 88.906, 39),
    "Zr": (160, 1.33, 4, 23.3, 91.224, 40),
    "Nb": (146, 1.60, 5, 18.0, 92.906, 41),
    "Mo": (139, 2.16, 6, 15.6, 95.950, 42),
    "Ru": (134, 2.20, 6, 13.1, 101.07, 44),
    "Rh": (134, 2.28, 6, 13.1, 102.906, 45),
    "Pd": (137, 2.20, 2, 14.7, 106.420, 46),
    "Ag": (144, 1.93, 1, 17.1, 107.868, 47),
    "In": (167, 1.78, 3, 26.1, 114.818, 49),
    "Sn": (140, 1.96, 4, 27.3, 118.710, 50),
    "Sb": (140, 2.05, 5, 29.9, 121.760, 51),
    "Te": (138, 2.10, 6, 32.1, 127.600, 52),
    "Cs": (265, 0.79, 1, 110.0, 132.905, 55),
    "Ba": (222, 0.89, 2, 62.5, 137.327, 56),
    "La": (187, 1.10, 3, 22.5, 138.905, 57),
    "Ce": (182, 1.12, 3, 20.7, 140.116, 58),
    "Pr": (182, 1.13, 3, 20.8, 140.908, 59),
    "Nd": (181, 1.14, 3, 20.6, 144.242, 60),
    "Sm": (180, 1.17, 3, 20.0, 150.360, 62),
    "Eu": (180, 1.20, 3, 28.9, 151.964, 63),
    "Gd": (180, 1.20, 3, 19.9, 157.250, 64),
    "Tb": (177, 1.20, 3, 19.3, 158.925, 65),
    "Dy": (178, 1.22, 3, 19.0, 162.500, 66),
    "Ho": (176, 1.23, 3, 18.7, 164.930, 67),
    "Er": (176, 1.24, 3, 18.4, 167.259, 68),
    "Tm": (176, 1.25, 3, 18.1, 168.934, 69),
    "Yb": (176, 1.10, 3, 24.8, 173.045, 70),
    "Lu": (174, 1.27, 3, 17.8, 174.967, 71),
    "Hf": (159, 1.30, 4, 22.3, 178.490, 72),
    "Ta": (146, 1.50, 5, 18.0, 180.948, 73),
    "W": (139, 2.36, 6, 15.8, 183.840, 74),
    "Re": (137, 1.90, 7, 14.7, 186.207, 75),
    "Os": (135, 2.20, 6, 14.0, 190.230, 76),
    "Ir": (136, 2.20, 4, 14.1, 192.217, 77),
    "Pt": (139, 2.28, 2, 15.1, 195.084, 78),
    "Au": (144, 2.54, 1, 17.0, 196.967, 79),
    "Pb": (175, 2.33, 4, 30.3, 207.200, 82),
    "Bi": (156, 2.02, 5, 35.4, 208.980, 83),
    "Th": (179, 1.30, 4, 32.0, 232.038, 90),
    "U": (156, 1.38, 6, 20.8, 238.029, 92),
    "Mm": (183, 1.12, 3, 21.0, 140.000, 58),
}

DATA_ELEMENT_COLUMNS = ["Mg", "Ni", "Co", "La", "Ce", "Pr", "Nd", "Sm", "Y", "Al", "In"]
ACTIVE_ELEMENT_POOL = DATA_ELEMENT_COLUMNS + ["Ti", "Zr"]
PHYSICAL_DESCRIPTOR_KEYWORDS = [
    "mixing_entropy",
    "size_mismatch",
    "RE_TM_ratio",
    "Mg_fraction",
    "avg_atomic_radius",
    "delta_atomic_radius",
    "avg_electronegativity",
    "delta_electronegativity",
    "avg_atomic_volume",
    "delta_atomic_volume",
    "avg_atomic_mass",
    "avg_atomic_number",
    "VEC",
]

MODEL_COLORS = {
    "LR": "#4363d8",
    "SVR": "#e6194B",
    "DT": "#3cb44b",
    "RF": "#911eb4",
    "GBRT": "#f58231",
    "XGBoost": "#42d4f4",
    "ExtraTrees": "#7f7f7f",
    "CatBoost": "#ffb000",
    "LightGBM": "#009e73",
    "AdaBoost": "#cc79a7",
    "HistGBRT": "#56b4e9",
}

METHOD_REFERENCES = [
    ("Reference workflow", "https://doi.org/10.1016/j.ijhydene.2026.155639"),
    ("CatBoost", "https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html"),
    ("XGBoost", "https://doi.org/10.1145/2939672.2939785"),
    ("LightGBM", "https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html"),
    ("ExtraTrees", "https://doi.org/10.1007/s10994-006-6226-1"),
    ("Mg-containing alloy screening", "https://doi.org/10.1016/j.est.2023.107720"),
]


@dataclass(frozen=True)
class TargetSpec:
    key: str
    target_col: str
    label: str
    process_features: tuple[str, ...]
    unit: str
    maximize: bool = True


TARGET_SPECS = OrderedDict(
    [
        (
            "capacity",
            TargetSpec(
                key="capacity",
                target_col="H_storage_wt%",
                label="H_storage_wt%",
                process_features=("temperature", "pressure"),
                unit="wt%",
                maximize=True,
            ),
        ),
        (
            "pressure",
            TargetSpec(
                key="pressure",
                target_col="pressure",
                label="Pressure_bar",
                process_features=("H_storage_wt%", "temperature"),
                unit="bar",
                maximize=False,
            ),
        ),
        (
            "temperature",
            TargetSpec(
                key="temperature",
                target_col="temperature",
                label="Temperature_K",
                process_features=("H_storage_wt%", "pressure"),
                unit="K",
                maximize=False,
            ),
        ),
    ]
)


def safe_name(name: str) -> str:
    return name.replace("%", "pct").replace("/", "_").replace(" ", "_")


def write_method_references(output_dir: Path) -> None:
    lines = ["# Method references", ""]
    for name, url in METHOD_REFERENCES:
        lines.append(f"- {name}: {url}")
    (output_dir / "method_references.md").write_text("\n".join(lines), encoding="utf-8")


def standardize_dataset(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {
        "Ab_T": "temperature",
        "Ab_P": "pressure",
        "Ab_max": "H_storage_wt%",
    }
    df = df.rename(columns=rename_map)

    required = DATA_ELEMENT_COLUMNS + ["temperature", "pressure", "H_storage_wt%"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Mg", "Ni", "temperature", "pressure", "H_storage_wt%"])
    df = df[(df["Mg"] > 0) & (df["Ni"] > 0)]
    df = df[(df["temperature"] > 0) & (df["pressure"] > 0) & (df["H_storage_wt%"] > 0)]
    if max_rows is not None and max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=RANDOM_STATE).sort_index()
    return df.reset_index(drop=True)


def ensure_element_columns(df: pd.DataFrame, element_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for elem in element_cols:
        if elem not in out.columns:
            out[elem] = 0.0
        out[elem] = pd.to_numeric(out[elem], errors="coerce").fillna(0.0)
    return out


def compute_material_features(df: pd.DataFrame, element_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = ensure_element_columns(df, element_cols)
    valid = [e for e in element_cols if e in ELEMENT_PROPS and e != "H"]
    if not valid:
        raise ValueError("No valid element columns found for feature engineering.")

    comp = out[valid].fillna(0).to_numpy(dtype=float)
    row_sums = comp.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    frac = comp / row_sums

    props = np.array([ELEMENT_PROPS[e] for e in valid], dtype=float)
    radii = props[:, 0]
    en = props[:, 1]
    val_e = props[:, 2]
    at_vol = props[:, 3]
    at_mass = props[:, 4]
    at_num = props[:, 5]

    avg_r = frac @ radii
    avg_en = frac @ en
    avg_vol = frac @ at_vol

    out["avg_atomic_radius"] = avg_r
    out["delta_atomic_radius"] = np.sqrt(np.sum(frac * (radii - avg_r.reshape(-1, 1)) ** 2, axis=1))
    out["avg_electronegativity"] = avg_en
    out["delta_electronegativity"] = np.sqrt(np.sum(frac * (en - avg_en.reshape(-1, 1)) ** 2, axis=1))
    out["VEC"] = frac @ val_e
    out["avg_atomic_volume"] = avg_vol
    out["delta_atomic_volume"] = np.sqrt(np.sum(frac * (at_vol - avg_vol.reshape(-1, 1)) ** 2, axis=1))
    out["avg_atomic_mass"] = frac @ at_mass
    out["avg_atomic_number"] = frac @ at_num
    out["n_elements"] = (comp > 0.001).sum(axis=1)

    frac_safe = np.where(frac > 1e-10, frac, 1e-10)
    out["mixing_entropy"] = -8.314 * np.sum(np.where(frac > 1e-10, frac * np.log(frac_safe), 0), axis=1)
    out["mixing_entropy"] = out["mixing_entropy"].clip(lower=0)

    r_safe = np.where(avg_r.reshape(-1, 1) > 0, avg_r.reshape(-1, 1), 1.0)
    out["size_mismatch"] = np.sqrt(np.sum(frac * (1 - radii / r_safe) ** 2, axis=1)) * 100

    re_elems = [e for e in valid if e in {"La", "Ce", "Pr", "Nd", "Sm", "Y", "Gd", "Tb", "Dy", "Ho", "Er", "Mm", "Sc", "Eu", "Tm", "Yb", "Lu"}]
    tm_elems = [e for e in valid if e in {"Ni", "Co", "Mn", "Fe", "Cu", "Cr", "V", "Ti", "Zr", "Nb", "Mo", "Pd", "Pt", "Ru", "Rh", "Ir", "Ag", "Au", "Zn"}]
    if re_elems and tm_elems:
        re_sum = out[re_elems].sum(axis=1)
        tm_sum = out[tm_elems].sum(axis=1)
        out["RE_TM_ratio"] = (re_sum / (tm_sum + 0.01)).clip(0, 100)
    else:
        out["RE_TM_ratio"] = 0.0

    out["Mg_fraction"] = out["Mg"] / (row_sums.flatten() + 1e-8) if "Mg" in valid else 0.0

    engineered = [
        "avg_atomic_radius",
        "delta_atomic_radius",
        "avg_electronegativity",
        "delta_electronegativity",
        "VEC",
        "avg_atomic_volume",
        "delta_atomic_volume",
        "avg_atomic_mass",
        "avg_atomic_number",
        "n_elements",
        "mixing_entropy",
        "size_mismatch",
        "RE_TM_ratio",
        "Mg_fraction",
    ]
    return out, engineered


def build_feature_frame(
    df: pd.DataFrame,
    spec: TargetSpec,
    element_cols: list[str],
    require_target: bool = True,
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    data = ensure_element_columns(df, element_cols)
    data, engineered = compute_material_features(data, element_cols)

    process_cols = [c for c in spec.process_features if c in data.columns]
    for col in process_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].fillna(data[col].median())

    feature_names = process_cols + engineered
    feature_names = list(dict.fromkeys(feature_names))

    if require_target:
        data[spec.target_col] = pd.to_numeric(data[spec.target_col], errors="coerce")
        data = data.dropna(subset=feature_names + [spec.target_col])
        data = data[data[spec.target_col] > 0]
    else:
        data = data.dropna(subset=feature_names)

    for col in feature_names:
        data = data[np.isfinite(data[col])]
    if require_target:
        data = data[np.isfinite(data[spec.target_col])]

    groups = {
        "process": process_cols,
        "engineered": engineered,
        "elements": element_cols,
        "physical": [f for f in engineered if any(k == f for k in PHYSICAL_DESCRIPTOR_KEYWORDS)],
    }
    return data.reset_index(drop=True), feature_names, groups


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mask = np.abs(y_true) > 1e-8
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else np.nan
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        slope = np.polyfit(y_true, y_pred, 1)[0]
        corr = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        slope = np.nan
        corr = np.nan
    chi = 1.0 / slope if np.isfinite(slope) and abs(slope) > 1e-8 else np.inf
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape, "chi": chi, "Corr": corr}


def has_param(estimator, param: str) -> bool:
    try:
        return param in estimator.get_params()
    except Exception:
        return False


def set_seed_if_supported(estimator, seed: int):
    model = clone(estimator)
    if has_param(model, "random_state"):
        model.set_params(random_state=seed)
    elif has_param(model, "random_seed"):
        model.set_params(random_seed=seed)
    return model


def get_models_and_params(quick: bool = False) -> tuple[OrderedDict, dict[str, dict], dict[str, str]]:
    models = OrderedDict()
    params: dict[str, dict] = {}
    notes: dict[str, str] = {}

    models["LR"] = Ridge()
    params["LR"] = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100]}

    models["SVR"] = SVR()
    params["SVR"] = {
        "C": [0.1, 1, 10, 50, 100, 200],
        "epsilon": [0.001, 0.01, 0.05, 0.1, 0.2],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
        "kernel": ["rbf"],
    }

    models["DT"] = DecisionTreeRegressor(random_state=RANDOM_STATE)
    params["DT"] = {
        "max_depth": [3, 5, 8, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", None],
    }

    rf_estimators = [50, 100] if quick else [100, 200, 300, 500, 800]
    models["RF"] = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    params["RF"] = {
        "n_estimators": rf_estimators,
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }

    gbdt_estimators = [50, 100] if quick else [100, 200, 300, 500, 800]
    models["GBRT"] = GradientBoostingRegressor(random_state=RANDOM_STATE)
    params["GBRT"] = {
        "n_estimators": gbdt_estimators,
        "learning_rate": [0.005, 0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 4, 5, 7, 10],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_samples_split": [2, 5, 10],
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            verbosity=0,
            n_jobs=-1,
        )
        params["XGBoost"] = {
            "n_estimators": gbdt_estimators,
            "learning_rate": [0.005, 0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 4, 5, 7, 10],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.001, 0.01, 0.1, 1],
            "reg_lambda": [0.5, 1, 1.5, 2, 5],
            "min_child_weight": [1, 3, 5, 7],
        }
    else:
        models["XGBoost"] = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
        params["XGBoost"] = {"max_iter": [100, 200], "learning_rate": [0.03, 0.05, 0.1], "max_leaf_nodes": [15, 31, 63]}
        notes["XGBoost"] = "Fallback: xgboost is not installed, using sklearn HistGradientBoostingRegressor."

    models["ExtraTrees"] = ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    params["ExtraTrees"] = {
        "n_estimators": rf_estimators,
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }

    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostRegressor(
            random_seed=RANDOM_STATE,
            loss_function="RMSE",
            verbose=False,
            allow_writing_files=False,
        )
        params["CatBoost"] = {
            "iterations": [100, 200, 300, 500] if not quick else [50, 100],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "depth": [4, 6, 8, 10],
            "l2_leaf_reg": [1, 3, 5, 10],
        }
    else:
        models["AdaBoost"] = AdaBoostRegressor(random_state=RANDOM_STATE)
        params["AdaBoost"] = {
            "n_estimators": [50, 100, 200] if not quick else [25, 50],
            "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        }
        notes["AdaBoost"] = "Fallback: catboost is not installed; CatBoost slot is replaced by AdaBoost."

    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        params["LightGBM"] = {
            "n_estimators": gbdt_estimators,
            "learning_rate": [0.005, 0.01, 0.05, 0.1],
            "num_leaves": [15, 31, 63, 127],
            "max_depth": [-1, 5, 10, 15],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        }
    else:
        models["HistGBRT"] = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
        params["HistGBRT"] = {
            "max_iter": [100, 200, 300] if not quick else [50, 100],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "max_leaf_nodes": [15, 31, 63],
            "l2_regularization": [0.0, 0.01, 0.1],
        }
        notes["HistGBRT"] = "Fallback: lightgbm is not installed; LightGBM slot is replaced by sklearn HistGradientBoostingRegressor."

    return models, params, notes


def randomized_search_or_fit(
    name: str,
    model,
    grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_splitter,
    n_iter: int,
):
    if grid:
        max_combos = len(ParameterGrid(grid))
        search_iter = max(1, min(n_iter, max_combos))
        search = RandomizedSearchCV(
            model,
            grid,
            n_iter=search_iter,
            cv=cv_splitter,
            scoring="r2",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        print(f"  Best params for {name}: {search.best_params_}")
        return search.best_estimator_, search.best_params_
    fitted = clone(model)
    fitted.fit(X_train, y_train)
    return fitted, {}


def train_and_evaluate(
    models: OrderedDict,
    param_grids: dict[str, dict],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_iter: int,
    cv: int,
) -> OrderedDict:
    results = OrderedDict()
    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        best_model, best_params = randomized_search_or_fit(
            name,
            model,
            param_grids.get(name, {}),
            X_train,
            y_train,
            cv_splitter,
            n_iter,
        )
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        train_metrics = calc_metrics(y_train, y_train_pred)
        test_metrics = calc_metrics(y_test, y_test_pred)
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_splitter, scoring="r2")
        results[name] = {
            "model": best_model,
            "best_params": best_params,
            "y_train_pred": y_train_pred,
            "y_test_pred": y_test_pred,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "cv_scores": cv_scores,
        }
        print(
            f"  Train R2={train_metrics['R2']:.4f}, RMSE={train_metrics['RMSE']:.4f}; "
            f"Test R2={test_metrics['R2']:.4f}, RMSE={test_metrics['RMSE']:.4f}, "
            f"MAE={test_metrics['MAE']:.4f}, MAPE={test_metrics['MAPE']:.2f}%"
        )
    return results


def results_to_frame(results: OrderedDict) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        tr = res["train_metrics"]
        te = res["test_metrics"]
        cv = res["cv_scores"]
        rows.append(
            {
                "Model": name,
                "Train_R2": tr["R2"],
                "Train_RMSE": tr["RMSE"],
                "Train_MAE": tr["MAE"],
                "Train_MAPE": tr["MAPE"],
                "Test_R2": te["R2"],
                "Test_RMSE": te["RMSE"],
                "Test_MAE": te["MAE"],
                "Test_MAPE": te["MAPE"],
                "Test_chi": te["chi"],
                "Test_Corr": te["Corr"],
                "CV_R2_mean": float(np.mean(cv)),
                "CV_R2_std": float(np.std(cv)),
                "Best_Params": json.dumps(res["best_params"], ensure_ascii=False, default=str),
            }
        )
    return pd.DataFrame(rows)


def print_results_table(results: OrderedDict) -> None:
    frame = results_to_frame(results)
    print("\nModel comparison:")
    print(
        frame[
            ["Model", "Train_R2", "Test_R2", "Test_RMSE", "Test_MAE", "Test_MAPE", "CV_R2_mean", "CV_R2_std"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    best = frame.sort_values("Test_R2", ascending=False).iloc[0]
    print(f"\nBest model: {best['Model']} (Test R2={best['Test_R2']:.4f})")


def plot_correlation_heatmap(feature_df: pd.DataFrame, y: np.ndarray, target_label: str, output_dir: Path) -> None:
    corr_df = feature_df.copy()
    corr_df[target_label] = y
    corr = corr_df.corr(numeric_only=True).abs()
    n = len(corr)
    fig_size = max(9, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.82))
    sns.heatmap(corr, cmap="YlGnBu", vmin=0, vmax=1, square=True, ax=ax, cbar_kws={"label": "|Pearson r|"})
    fs = max(6, min(10, 120 // max(n, 1)))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fs)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fs)
    ax.set_title(f"Feature correlation heatmap - {target_label}", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(output_dir / f"Fig_Correlation_Heatmap_{safe_name(target_label)}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"Fig_Correlation_Heatmap_{safe_name(target_label)}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_paper_style_scatter_error(
    results: OrderedDict,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_label: str,
    output_dir: Path,
) -> None:
    model_names = list(results.keys())
    n = len(model_names)
    fig, axes = plt.subplots(n, 2, figsize=(13, max(3.2 * n, 8)))
    if n == 1:
        axes = np.array([axes])

    for idx, name in enumerate(model_names):
        res = results[name]
        y_pred = res["y_test_pred"]
        y_train_pred = res["y_train_pred"]
        resid = y_pred - y_test
        color = MODEL_COLORS.get(name, "#4c72b0")

        ax_sc = axes[idx, 0]
        ax_sc.scatter(y_train, y_train_pred, c="#2c3e50", alpha=0.35, s=12, label="Training set")
        ax_sc.scatter(y_test, y_pred, c=color, alpha=0.75, s=20, edgecolors="black", linewidth=0.3, label="Testing set")
        all_v = np.concatenate([y_train, y_test, y_train_pred, y_pred])
        pad = max(0.05 * (all_v.max() - all_v.min()), 0.1)
        lims = [all_v.min() - pad, all_v.max() + pad]
        ax_sc.plot(lims, lims, "k-", lw=1.2)
        ax_sc.plot(lims, [v * 1.2 for v in lims], "k--", lw=0.8, alpha=0.6)
        ax_sc.plot(lims, [v * 0.8 for v in lims], "k--", lw=0.8, alpha=0.6)
        ax_sc.set_xlim(lims)
        ax_sc.set_ylim(lims)
        ax_sc.set_xlabel(f"Experimental {target_label}")
        ax_sc.set_ylabel(f"Predicted {target_label}")
        ax_sc.set_aspect("equal", adjustable="box")
        tr = res["train_metrics"]
        te = res["test_metrics"]
        txt = (
            f"Train R2={tr['R2']:.4f}\nTrain RMSE={tr['RMSE']:.4f}\n"
            f"Test R2={te['R2']:.4f}\nTest RMSE={te['RMSE']:.4f}\n"
            f"MAE={te['MAE']:.4f}\nMAPE={te['MAPE']:.2f}%"
        )
        ax_sc.text(
            0.03,
            0.97,
            txt,
            transform=ax_sc.transAxes,
            fontsize=8,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.92),
        )
        ax_sc.set_title(f"({idx + 1}) {name}", fontweight="bold", loc="left")
        ax_sc.legend(loc="lower right", fontsize=8)

        ax_err = axes[idx, 1]
        sample_idx = np.arange(len(resid))
        try:
            xy = np.vstack([sample_idx, resid])
            kde = gaussian_kde(xy)
            y_pad = max(np.std(resid) * 2, 1e-6)
            xx, yy = np.mgrid[0 : max(len(resid) - 1, 1) : 120j, resid.min() - y_pad : resid.max() + y_pad : 120j]
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax_err.contourf(xx, yy, zz, levels=18, cmap="viridis", alpha=0.85)
        except Exception:
            pass
        ax_err.scatter(sample_idx, resid, c=color, s=10, alpha=0.6)
        ax_err.axhline(0, color="red", ls="--", lw=1.2)
        ax_err.set_xlabel("No. of samples")
        ax_err.set_ylabel("Prediction error")
        ax_err.set_title(f"Error distribution - {name}", fontsize=10)

    fig.suptitle(f"Performance of ML models for {target_label}", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(output_dir / f"Fig_PaperStyle_{safe_name(target_label)}.png", dpi=300)
    fig.savefig(output_dir / f"Fig_PaperStyle_{safe_name(target_label)}.pdf")
    plt.close(fig)


def plot_radar_comparison(results: OrderedDict, target_label: str, output_dir: Path) -> None:
    metrics = ["R2", "RMSE", "MAE", "MAPE"]
    ylabels = ["R2", "RMSE", "MAE", "MAPE (%)"]
    model_names = list(results.keys())
    n = len(model_names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_plot = angles + [angles[0]]

    fig, axes = plt.subplots(1, 4, subplot_kw=dict(polar=True), figsize=(16, 4.5))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.85, bottom=0.18, wspace=0.42)
    for i, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        ax = axes[i]
        train_vals = np.array([results[m]["train_metrics"][metric] for m in model_names], dtype=float)
        test_vals = np.array([results[m]["test_metrics"][metric] for m in model_names], dtype=float)
        all_vals = np.concatenate([train_vals, test_vals])
        lo = max(0.0, np.nanmin(all_vals) * 0.85)
        hi = np.nanmax(all_vals) * 1.1 if np.nanmax(all_vals) > 0 else 1.0
        ticks = mticker.MaxNLocator(nbins=4).tick_values(lo, hi)
        ticks = ticks[ticks >= 0]
        if len(ticks) < 2:
            ticks = np.linspace(lo, hi, 4)
        ax.plot(angles_plot, np.append(train_vals, train_vals[0]), color="#2a7a2a", lw=1.5, marker="o", label="Training set")
        ax.fill(angles_plot, np.append(train_vals, train_vals[0]), color="#2a7a2a", alpha=0.12)
        ax.plot(angles_plot, np.append(test_vals, test_vals[0]), color="#d62020", lw=1.5, ls="--", marker="o", label="Testing set")
        ax.fill(angles_plot, np.append(test_vals, test_vals[0]), color="#d62020", alpha=0.10)
        ax.set_xticks(angles)
        ax.set_xticklabels(model_names, fontsize=8)
        ax.set_ylim(ticks[0], ticks[-1])
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{v:.0f}%" if metric == "MAPE" else f"{v:.2g}" for v in ticks], fontsize=7, color="gray")
        ax.set_title(f"({chr(97 + i)}) {ylabel}", fontsize=10, pad=12)
        ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, edgecolor="black", bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(f"Radar comparison - {target_label}", fontsize=14, fontweight="bold")
    fig.savefig(output_dir / f"Fig_Radar_Comparison_{safe_name(target_label)}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"Fig_Radar_Comparison_{safe_name(target_label)}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_std_comparison(results: OrderedDict, y_train: np.ndarray, y_test: np.ndarray, target_label: str, output_dir: Path) -> None:
    model_names = list(results.keys())
    x = np.arange(len(model_names))
    width = 0.2
    sd_exp_train = np.std(y_train)
    sd_exp_test = np.std(y_test)
    sd_pred_train = np.array([np.std(results[m]["y_train_pred"]) for m in model_names])
    sd_pred_test = np.array([np.std(results[m]["y_test_pred"]) for m in model_names])

    fig, ax = plt.subplots(figsize=(max(11, len(model_names) * 0.9), 6))
    ax.bar(x - 1.5 * width, np.full(len(model_names), sd_exp_train), width, label="SD_exp Train", color="#2d6a4f", edgecolor="black")
    ax.bar(x - 0.5 * width, sd_pred_train, width, label="SD_pred Train", color="#95d5b2", edgecolor="black")
    ax.bar(x + 0.5 * width, np.full(len(model_names), sd_exp_test), width, label="SD_exp Test", color="#d62828", edgecolor="black")
    ax.bar(x + 1.5 * width, sd_pred_test, width, label="SD_pred Test", color="#f4a261", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right", fontweight="bold")
    ax.set_ylabel(f"Standard deviation ({target_label})")
    ax.set_xlabel("Model")
    ax.set_title(f"Standard deviation comparison: experimental vs. predicted - {target_label}", fontsize=12, fontweight="bold")
    ax.legend(ncol=2, fontsize=9, frameon=True, edgecolor="black")
    ax.grid(axis="y", alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(output_dir / f"Fig_SD_Comparison_{safe_name(target_label)}.png", dpi=300)
    fig.savefig(output_dir / f"Fig_SD_Comparison_{safe_name(target_label)}.pdf")
    plt.close(fig)


def plot_feature_importance(results: OrderedDict, feature_names: list[str], target_label: str, output_dir: Path) -> None:
    items = [(name, res["model"]) for name, res in results.items() if hasattr(res["model"], "feature_importances_")]
    if not items:
        return
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(max(5 * n, 8), 7))
    if n == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, items):
        imp = np.asarray(model.feature_importances_, dtype=float)
        top = np.argsort(imp)[-min(15, len(feature_names)) :]
        ax.barh(range(len(top)), imp[top], color=MODEL_COLORS.get(name, "#4c72b0"), edgecolor="black", lw=0.5)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([feature_names[i] for i in top], fontsize=8)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Importance")
    fig.suptitle(f"Feature importance - {target_label}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / f"Fig_Feature_Importance_{safe_name(target_label)}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"Fig_Feature_Importance_{safe_name(target_label)}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_shap_summary(best_name: str, best_model, X_train: np.ndarray, feature_names: list[str], target_label: str, output_dir: Path) -> None:
    if not HAS_SHAP:
        print("SHAP is not installed; skipping SHAP plots.")
        return
    tree_like = {"RF", "GBRT", "XGBoost", "DT", "ExtraTrees", "CatBoost", "LightGBM"}
    if best_name not in tree_like:
        print(f"Skipping SHAP for {best_name}; tree SHAP is reserved for tree-like models.")
        return
    sample_size = min(1000, X_train.shape[0])
    X_sample = X_train[:sample_size]
    try:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_sample)
    except Exception as exc:
        print(f"SHAP failed for {best_name}: {exc}")
        return
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=20)
    plt.title(f"SHAP feature importance - {target_label}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / f"Fig_SHAP_Beeswarm_{safe_name(target_label)}.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / f"Fig_SHAP_Beeswarm_{safe_name(target_label)}.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False, max_display=20)
    plt.title(f"Mean |SHAP| - {target_label}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / f"Fig_SHAP_Bar_{safe_name(target_label)}.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / f"Fig_SHAP_Bar_{safe_name(target_label)}.pdf", bbox_inches="tight")
    plt.close()


def select_ablation_features(
    all_data: pd.DataFrame,
    feature_names: list[str],
    groups: dict[str, list[str]],
    element_cols: list[str],
) -> OrderedDict[str, list[str]]:
    physical = set(groups.get("physical", []))
    process = set(groups.get("process", []))
    available_elements = [e for e in element_cols if e in all_data.columns]
    basic = available_elements + [f for f in ["n_elements", "Mg_fraction"] if f in all_data.columns]
    subsets = OrderedDict()
    subsets["Full"] = feature_names
    subsets["No_Physical_Descriptors"] = [f for f in feature_names if f not in physical]
    subsets["No_Explicit_Process"] = [f for f in feature_names if f not in process]
    subsets["Basic_Composition"] = basic
    return OrderedDict((k, v) for k, v in subsets.items() if len(v) > 0)


def run_ablation_study(
    all_data: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    groups: dict[str, list[str]],
    element_cols: list[str],
    best_model,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target_label: str,
    output_dir: Path,
) -> pd.DataFrame:
    subsets = select_ablation_features(all_data, feature_names, groups, element_cols)
    rows = []
    for subset_name, subset_features in subsets.items():
        X = all_data[subset_features].to_numpy(dtype=float)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train_n = scaler.fit_transform(X_train)
        X_test_n = scaler.transform(X_test)
        model = clone(best_model)
        model.fit(X_train_n, y_train)
        pred = model.predict(X_test_n)
        m = calc_metrics(y_test, pred)
        rows.append(
            {
                "Subset": subset_name,
                "Feature_Count": len(subset_features),
                "Features": "; ".join(subset_features),
                "R2": m["R2"],
                "RMSE": m["RMSE"],
                "MAE": m["MAE"],
                "MAPE": m["MAPE"],
            }
        )
        print(f"  Ablation {subset_name}: R2={m['R2']:.4f}, RMSE={m['RMSE']:.4f}")
    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(output_dir / f"ablation_results_{safe_name(target_label)}.csv", index=False, encoding="utf-8-sig")
    plot_ablation_results(ablation_df, target_label, output_dir)
    return ablation_df


def plot_ablation_results(ablation_df: pd.DataFrame, target_label: str, output_dir: Path) -> None:
    metrics = ["R2", "RMSE", "MAE", "MAPE"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        bars = ax.bar(ablation_df["Subset"], ablation_df[metric], color="#4c72b0", edgecolor="black", alpha=0.82)
        ax.set_title(metric, fontweight="bold")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25, ls="--")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Feature ablation study - {target_label}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / f"Fig_Ablation_{safe_name(target_label)}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"Fig_Ablation_{safe_name(target_label)}.pdf", bbox_inches="tight")
    plt.close(fig)


def composition_to_formula(row: pd.Series, element_cols: list[str], min_amount: float = 0.05) -> str:
    parts = []
    for elem in element_cols:
        val = float(row.get(elem, 0.0))
        if val >= min_amount:
            txt = f"{val:.2f}".rstrip("0").rstrip(".")
            parts.append(f"{elem}{txt}")
    return "".join(parts)


def generate_virtual_candidates(
    data: pd.DataFrame,
    n_candidates: int,
    element_cols: list[str],
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    base = ensure_element_columns(data, element_cols)
    base_idx = rng.choice(base.index.to_numpy(), size=n_candidates, replace=True)
    comp = base.loc[base_idx, element_cols].to_numpy(dtype=float)
    noise = rng.normal(loc=1.0, scale=0.08, size=comp.shape)
    comp = np.clip(comp * noise, 0, None)

    dopant_cols = [element_cols.index(e) for e in element_cols if e not in {"Mg", "Ni"}]
    for i in range(n_candidates):
        n_dopants = int(rng.integers(1, min(4, len(dopant_cols)) + 1))
        chosen = rng.choice(dopant_cols, size=n_dopants, replace=False)
        comp[i, chosen] += rng.uniform(0.05, 3.0, size=n_dopants)
        if "Ti" in element_cols and rng.random() < 0.35:
            comp[i, element_cols.index("Ti")] += rng.uniform(0.05, 2.0)
        if "Zr" in element_cols and rng.random() < 0.25:
            comp[i, element_cols.index("Zr")] += rng.uniform(0.05, 2.5)

    comp[:, element_cols.index("Mg")] = np.maximum(comp[:, element_cols.index("Mg")], 1e-4)
    comp[:, element_cols.index("Ni")] = np.maximum(comp[:, element_cols.index("Ni")], 1e-4)
    row_sum = comp.sum(axis=1, keepdims=True)
    comp = comp / np.where(row_sum == 0, 1, row_sum) * 100.0

    candidates = pd.DataFrame(comp, columns=element_cols)
    for col in ["temperature", "pressure"]:
        q01, q99 = data[col].quantile([0.01, 0.99])
        values = data[col].to_numpy(dtype=float)
        sampled = rng.choice(values, size=n_candidates, replace=True)
        perturbed = sampled + rng.normal(0, max(data[col].std() * 0.05, 1e-6), size=n_candidates)
        candidates[col] = np.clip(perturbed, q01, q99)
    candidates["Formula"] = candidates.apply(lambda row: composition_to_formula(row, element_cols), axis=1)
    return candidates


def choose_uncertainty_base_model(results: OrderedDict):
    preference = ["CatBoost", "XGBoost", "LightGBM", "RF", "ExtraTrees", "GBRT"]
    for name in preference:
        if name in results:
            return name, results[name]["model"]
    best = max(results, key=lambda k: results[k]["test_metrics"]["R2"])
    return best, results[best]["model"]


def train_uncertainty_ensemble(
    base_model,
    X: np.ndarray,
    y: np.ndarray,
    n_members: int,
    random_state: int = RANDOM_STATE,
) -> dict:
    X_core, X_val, y_core, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_core_n = scaler.fit_transform(X_core)
    X_val_n = scaler.transform(X_val)
    rng = np.random.default_rng(random_state)
    models = []
    val_preds = []
    val_rmse = []
    for member in range(n_members):
        idx = rng.choice(np.arange(len(X_core_n)), size=len(X_core_n), replace=True)
        model = set_seed_if_supported(base_model, random_state + member + 1)
        model.fit(X_core_n[idx], y_core[idx])
        pred = model.predict(X_val_n)
        models.append(model)
        val_preds.append(pred)
        val_rmse.append(math.sqrt(mean_squared_error(y_val, pred)))
    val_rmse = np.asarray(val_rmse, dtype=float)
    weights = 1.0 / (val_rmse + 1e-8)
    weights = weights / weights.sum()
    val_pred_matrix = np.vstack(val_preds)
    val_mean = weights @ val_pred_matrix
    val_disagreement = np.sqrt(np.sum(weights[:, None] * (val_pred_matrix - val_mean) ** 2, axis=0))
    val_uncertainty = np.sqrt((weights @ val_rmse) ** 2 + 0.5 * val_disagreement**2)
    val_abs_error = np.abs(y_val - val_mean)
    uncertainty_error_corr = np.corrcoef(val_abs_error, val_uncertainty)[0, 1] if len(y_val) > 2 else np.nan
    return {
        "models": models,
        "weights": weights,
        "scaler": scaler,
        "val_rmse": val_rmse,
        "val_abs_error": val_abs_error,
        "val_uncertainty": val_uncertainty,
        "uncertainty_error_corr": uncertainty_error_corr,
    }


def predict_uncertainty(ensemble: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_n = ensemble["scaler"].transform(X)
    pred_matrix = np.vstack([model.predict(X_n) for model in ensemble["models"]])
    weights = ensemble["weights"]
    mean = weights @ pred_matrix
    disagreement = np.sqrt(np.sum(weights[:, None] * (pred_matrix - mean) ** 2, axis=0))
    intra = float(weights @ ensemble["val_rmse"])
    sigma = np.sqrt(intra**2 + 0.5 * disagreement**2)
    return mean, sigma


def run_active_learning_screening(
    source_data: pd.DataFrame,
    capacity_feature_data: pd.DataFrame,
    feature_names: list[str],
    capacity_results: OrderedDict,
    n_candidates: int,
    rounds: int,
    query_size: int,
    ensemble_members: int,
    output_dir: Path,
) -> pd.DataFrame:
    print("\nUCB-based virtual screening for capacity ...")
    element_cols = ACTIVE_ELEMENT_POOL
    candidate_raw = generate_virtual_candidates(source_data, n_candidates, element_cols)
    spec = TARGET_SPECS["capacity"]
    candidate_features, _, _ = build_feature_frame(candidate_raw, spec, element_cols, require_target=False)

    train_feature_data, _, _ = build_feature_frame(source_data, spec, element_cols, require_target=True)
    X_train_all = train_feature_data[feature_names].to_numpy(dtype=float)
    y_train_all = train_feature_data[spec.target_col].to_numpy(dtype=float)
    X_candidates = candidate_features[feature_names].to_numpy(dtype=float)

    base_name, base_model = choose_uncertainty_base_model(capacity_results)
    print(f"  Bayesian-style uncertainty ensemble base learner: {base_name}")
    ensemble = train_uncertainty_ensemble(base_model, X_train_all, y_train_all, ensemble_members)
    mu, sigma = predict_uncertainty(ensemble, X_candidates)

    candidate_features["Predicted_Ab_max"] = mu
    candidate_features["Uncertainty"] = sigma
    candidate_features["UCB_BaseModel"] = base_name
    candidate_features["Formula"] = candidate_raw["Formula"].values

    selected = np.zeros(len(candidate_features), dtype=bool)
    round_records = []
    k_values = np.linspace(0.9, 0.5, rounds)
    query_size = min(query_size, len(candidate_features))
    for round_id, k in enumerate(k_values, start=1):
        scores = candidate_features["Predicted_Ab_max"].to_numpy() + k * candidate_features["Uncertainty"].to_numpy()
        scores[selected] = -np.inf
        take = min(query_size, int((~selected).sum()))
        if take <= 0:
            break
        idx = np.argsort(scores)[-take:][::-1]
        selected[idx] = True
        round_block = candidate_features.iloc[idx].copy()
        round_block["Active_Round"] = round_id
        round_block["Exploration_k"] = k
        round_block["Acquisition_UCB"] = scores[idx]
        round_records.append(round_block)
        print(f"  Round {round_id}: selected {take} candidates, k={k:.2f}, best UCB={scores[idx[0]]:.4f}")

    screened = pd.concat(round_records, ignore_index=True)
    screened = screened.sort_values("Acquisition_UCB", ascending=False).reset_index(drop=True)
    screened.insert(0, "Rank", np.arange(1, len(screened) + 1))
    out_cols = (
        ["Rank", "Active_Round", "Exploration_k", "Formula"]
        + element_cols
        + ["temperature", "pressure", "Predicted_Ab_max", "Uncertainty", "Acquisition_UCB", "UCB_BaseModel"]
    )
    screened[out_cols].to_csv(output_dir / "active_learning_screened_candidates_capacity.csv", index=False, encoding="utf-8-sig")

    diagnostics = {
        "base_model": base_name,
        "ensemble_members": ensemble_members,
        "validation_member_rmse": ensemble["val_rmse"].tolist(),
        "validation_weights": ensemble["weights"].tolist(),
        "validation_error_uncertainty_corr": float(ensemble["uncertainty_error_corr"]),
        "candidate_count": int(n_candidates),
        "selected_count": int(len(screened)),
    }
    (output_dir / "active_learning_uncertainty_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    plot_active_learning(screened, ensemble, output_dir)
    return screened


def plot_active_learning(screened: pd.DataFrame, ensemble: dict, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(screened["Predicted_Ab_max"], bins=40, color="#4c72b0", alpha=0.78, edgecolor="black")
    ax.set_xlabel("Predicted hydrogen storage capacity (wt%)")
    ax.set_ylabel("Selected candidate count")
    ax.set_title("UCB-prioritized candidate distribution", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "Fig_ActiveLearning_Candidate_Distribution.png", dpi=300)
    fig.savefig(output_dir / "Fig_ActiveLearning_Candidate_Distribution.pdf")
    plt.close(fig)

    top = screened.head(20).copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["Formula"], top["Acquisition_UCB"], color="#009e73", edgecolor="black")
    ax.set_xlabel("UCB acquisition score")
    ax.set_title("Top UCB-prioritized candidates", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "Fig_ActiveLearning_Top20_Candidates.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "Fig_ActiveLearning_Top20_Candidates.pdf", bbox_inches="tight")
    plt.close(fig)

    if np.isfinite(ensemble["uncertainty_error_corr"]):
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.scatter(ensemble["val_uncertainty"], ensemble["val_abs_error"], s=18, alpha=0.65, c="#d55e00", edgecolors="black", linewidth=0.25)
        ax.set_xlabel("Predicted uncertainty")
        ax.set_ylabel("Validation absolute error")
        ax.set_title(f"Error-uncertainty correlation r={ensemble['uncertainty_error_corr']:.3f}", fontweight="bold")
        ax.grid(alpha=0.25, ls="--")
        fig.tight_layout()
    fig.savefig(output_dir / "Fig_Uncertainty_Error_Correlation.png", dpi=300)
    fig.savefig(output_dir / "Fig_Uncertainty_Error_Correlation.pdf")
    plt.close(fig)


def metric_csv_path(output_dir: Path, spec: TargetSpec) -> Path:
    return output_dir / spec.key / f"model_comparison_{safe_name(spec.label)}.csv"


def feature_csv_path(output_dir: Path, spec: TargetSpec) -> Path:
    return output_dir / spec.key / f"processed_features_{safe_name(spec.label)}.csv"


def ablation_csv_path(output_dir: Path, spec: TargetSpec) -> Path:
    return output_dir / spec.key / f"ablation_results_{safe_name(spec.label)}.csv"


def prediction_csv_path(output_dir: Path, spec: TargetSpec) -> Path:
    return output_dir / spec.key / f"predictions_{safe_name(spec.label)}.csv"


def save_prediction_frame(
    results: OrderedDict,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    spec: TargetSpec,
    target_dir: Path,
) -> pd.DataFrame:
    rows = []
    for model_name, res in results.items():
        for split, sample_idx, actual, predicted in (
            ("Train", train_idx, y_train, res["y_train_pred"]),
            ("Test", test_idx, y_test, res["y_test_pred"]),
        ):
            for row_id, y_true, y_pred in zip(sample_idx, actual, predicted):
                rows.append(
                    {
                        "Target": spec.label,
                        "Model": model_name,
                        "Split": split,
                        "Sample_Index": int(row_id),
                        "Experimental": float(y_true),
                        "Predicted": float(y_pred),
                        "Residual": float(y_pred - y_true),
                        "Absolute_Error": float(abs(y_pred - y_true)),
                    }
                )
    prediction_df = pd.DataFrame(rows)
    prediction_df.to_csv(prediction_csv_path(target_dir.parent, spec), index=False, encoding="utf-8-sig")
    return prediction_df


def load_target_metrics(output_dir: Path, target_keys: list[str]) -> pd.DataFrame:
    frames = []
    for key in target_keys:
        spec = TARGET_SPECS[key]
        path = metric_csv_path(output_dir, spec)
        if path.exists():
            df = pd.read_csv(path)
            df.insert(0, "Target", spec.label)
            df.insert(1, "Unit", spec.unit)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_publication_feature_frame(output_dir: Path, spec: TargetSpec, source_data: pd.DataFrame | None = None) -> pd.DataFrame | None:
    path = feature_csv_path(output_dir, spec)
    if path.exists():
        return pd.read_csv(path)
    if source_data is None:
        return None
    feature_data, _, _ = build_feature_frame(source_data, spec, ACTIVE_ELEMENT_POOL, require_target=True)
    return feature_data


def make_publication_dir(output_dir: Path) -> Path:
    fig_dir = output_dir / "MSEB_submission_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def save_pubfig(fig: plt.Figure, fig_dir: Path, stem: str, dpi: int = 500) -> None:
    fig.savefig(fig_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.tiff", dpi=dpi, bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_publication_workflow(output_dir: Path, n_rows: int | None = None) -> None:
    fig_dir = make_publication_dir(output_dir)
    fig, ax = plt.subplots(figsize=(14.2, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    sample_text = f"{n_rows} Mg–Ni entries" if n_rows else "Mg–Ni dataset"
    boxes = [
        (0.035, 0.60, 0.205, 0.23, f"Data curation\n{sample_text}\ncapacity / pressure /\ntemperature", "#e8f3f8"),
        (0.275, 0.60, 0.205, 0.23, "Descriptors\ncomposition + process\natomic features\nthermodynamic features", "#eef7e8"),
        (0.515, 0.60, 0.205, 0.23, "Nine ML models\nLR, SVR, DT, RF\nGBRT, XGBoost\nET, CatBoost, LGBM", "#fff3df"),
        (0.755, 0.60, 0.205, 0.23, "Evaluation\nR2 / RMSE\nMAE / MAPE\nCV + test split", "#f7e8f0"),
        (0.105, 0.20, 0.245, 0.21, "Interpretability\nfeature importance\nSHAP-ready outputs", "#f5f5f5"),
        (0.390, 0.20, 0.245, 0.21, "Ablation study\nfull descriptors\nvs reduced subsets", "#edf0ff"),
        (0.675, 0.20, 0.245, 0.21, "Uncertainty-guided\nscreening\nUCB candidate ranking", "#e9f7f3"),
    ]
    for x, y, w, h, text, color in boxes:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.018",
            linewidth=1.3,
            edgecolor="#333333",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.8, fontweight="bold", linespacing=1.28)

    arrows = [
        ((0.240, 0.715), (0.275, 0.715)),
        ((0.480, 0.715), (0.515, 0.715)),
        ((0.720, 0.715), (0.755, 0.715)),
        ((0.600, 0.600), (0.228, 0.410)),
        ((0.618, 0.600), (0.512, 0.410)),
        ((0.636, 0.600), (0.798, 0.410)),
    ]
    for start, end in arrows:
        arrow = patches.FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color="#333333")
        ax.add_patch(arrow)

    ax.text(0.5, 0.94, "Integrated machine-learning workflow for Mg–Ni hydrogen-storage alloys", ha="center", fontsize=15, fontweight="bold")
    ax.text(
        0.5,
        0.04,
        "The workflow follows multi-property prediction, descriptor ablation, and uncertainty-aware virtual screening practices.",
        ha="center",
        fontsize=10,
        color="#333333",
    )
    save_pubfig(fig, fig_dir, "Figure_01_Workflow")


def plot_publication_correlation_panels(
    output_dir: Path,
    target_keys: list[str],
    source_data: pd.DataFrame | None = None,
) -> None:
    fig_dir = make_publication_dir(output_dir)
    frames = []
    for key in target_keys:
        spec = TARGET_SPECS[key]
        feature_data = load_publication_feature_frame(output_dir, spec, source_data)
        if feature_data is None or spec.target_col not in feature_data.columns:
            continue
        numeric_cols = [c for c in feature_data.columns if pd.api.types.is_numeric_dtype(feature_data[c])]
        cols = [c for c in numeric_cols if c != spec.target_col] + [spec.target_col]
        corr = feature_data[cols].corr(numeric_only=True).abs()
        frames.append((spec, corr))

    if not frames:
        print("Publication correlation panels skipped: no processed feature files were found.")
        return

    fig, axes = plt.subplots(1, len(frames), figsize=(6.8 * len(frames), 6.4))
    if len(frames) == 1:
        axes = [axes]
    for i, (spec, corr) in enumerate(frames):
        ax = axes[i]
        sns.heatmap(
            corr,
            ax=ax,
            cmap="YlGnBu",
            vmin=0,
            vmax=1,
            square=True,
            cbar=i == len(frames) - 1,
            cbar_kws={"label": "|Pearson r|", "shrink": 0.72},
        )
        ax.set_title(f"({chr(97 + i)}) {spec.label}", fontsize=12, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right", fontsize=6)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
    fig.suptitle("Correlation heatmaps for multi-property Mg–Ni descriptors", fontsize=15, fontweight="bold")
    save_pubfig(fig, fig_dir, "Figure_02_Correlation_Heatmaps")


def plot_publication_model_comparison(output_dir: Path, target_keys: list[str]) -> None:
    fig_dir = make_publication_dir(output_dir)
    metrics = load_target_metrics(output_dir, target_keys)
    if metrics.empty:
        print("Publication model comparison skipped: no model comparison CSV files were found.")
        return

    metric_specs = [
        ("Test_R2", "R2", "higher is better"),
        ("Test_RMSE", "RMSE", "lower is better"),
        ("Test_MAE", "MAE", "lower is better"),
        ("Test_MAPE", "MAPE (%)", "lower is better"),
    ]
    targets = [TARGET_SPECS[k].label for k in target_keys if TARGET_SPECS[k].label in set(metrics["Target"])]
    fig, axes = plt.subplots(len(metric_specs), len(targets), figsize=(16, 15))
    if len(targets) == 1:
        axes = axes.reshape(len(metric_specs), 1)

    for c, target in enumerate(targets):
        target_df = metrics[metrics["Target"] == target].copy()
        model_order = target_df["Model"].tolist()
        colors = [MODEL_COLORS.get(model, "#4c72b0") for model in model_order]
        for r, (metric_col, metric_label, subtitle) in enumerate(metric_specs):
            ax = axes[r, c]
            vals = target_df[metric_col].to_numpy(dtype=float)
            bars = ax.bar(np.arange(len(model_order)), vals, color=colors, edgecolor="black", linewidth=0.55)
            best_idx = int(np.argmax(vals) if metric_col == "Test_R2" else np.argmin(vals))
            bars[best_idx].set_linewidth(1.8)
            bars[best_idx].set_edgecolor("#111111")
            ax.set_xticks(np.arange(len(model_order)))
            ax.set_xticklabels(model_order, rotation=38, ha="right", fontsize=7.4)
            ax.set_ylabel(metric_label)
            ax.set_title(f"{target} - {metric_label}\n{subtitle}", fontsize=9.5, fontweight="bold")
            if metric_col == "Test_R2":
                ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.28, ls="--")
            ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Testing performance comparison of nine machine-learning models", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_pubfig(fig, fig_dir, "Figure_03_Model_Performance")


def best_model_for_target(output_dir: Path, spec: TargetSpec) -> str | None:
    summary_path = output_dir / "summary_best_models.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        hit = summary[summary["Target"] == spec.label]
        if not hit.empty:
            return str(hit.iloc[0]["Best_Model"])
    metrics_path = metric_csv_path(output_dir, spec)
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if not metrics.empty:
            return str(metrics.sort_values("Test_R2", ascending=False).iloc[0]["Model"])
    return None


def plot_publication_best_prediction_panels(
    output_dir: Path,
    target_keys: list[str],
    run_outputs: dict | None = None,
) -> None:
    fig_dir = make_publication_dir(output_dir)
    panels = []
    for i, key in enumerate(target_keys):
        spec = TARGET_SPECS[key]
        best_name = best_model_for_target(output_dir, spec)
        pred_df = None
        metrics_df = pd.read_csv(metric_csv_path(output_dir, spec)) if metric_csv_path(output_dir, spec).exists() else pd.DataFrame()
        if run_outputs and key in run_outputs:
            info = run_outputs[key]
            rows = []
            feature_data = info["feature_data"]
            y_train = feature_data.iloc[info["train_idx"]][spec.target_col].to_numpy(dtype=float)
            y_test = feature_data.iloc[info["test_idx"]][spec.target_col].to_numpy(dtype=float)
            for model_name, res in info["results"].items():
                rows.extend(
                    {
                        "Model": model_name,
                        "Split": "Train",
                        "Experimental": float(actual),
                        "Predicted": float(pred),
                    }
                    for actual, pred in zip(y_train, res["y_train_pred"])
                )
                rows.extend(
                    {
                        "Model": model_name,
                        "Split": "Test",
                        "Experimental": float(actual),
                        "Predicted": float(pred),
                    }
                    for actual, pred in zip(y_test, res["y_test_pred"])
                )
            pred_df = pd.DataFrame(rows)
        else:
            pred_path = prediction_csv_path(output_dir, spec)
            if pred_path.exists():
                pred_df = pd.read_csv(pred_path)

        if pred_df is None or pred_df.empty or best_name is None:
            print(f"Publication best-model parity panel skipped for {spec.label}: prediction data were not found.")
            continue

        best_df = pred_df[pred_df["Model"] == best_name].copy()
        if best_df.empty:
            print(f"Publication best-model parity panel skipped for {spec.label}: {best_name} predictions were not found.")
            continue
        metric_row = pd.Series(dtype=float)
        if not metrics_df.empty:
            hit = metrics_df[metrics_df["Model"] == best_name]
            if not hit.empty:
                metric_row = hit.iloc[0]
        panels.append((spec, best_name, best_df, metric_row))

    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5.3 * len(panels), 5.1))
    if len(panels) == 1:
        axes = [axes]
    for idx, (spec, best_name, best_df, metric_row) in enumerate(panels):
        ax = axes[idx]
        train_df = best_df[best_df["Split"] == "Train"]
        test_df = best_df[best_df["Split"] == "Test"]
        color = MODEL_COLORS.get(best_name, "#4c72b0")
        ax.scatter(train_df["Experimental"], train_df["Predicted"], s=8, c="#8b959b", alpha=0.22, label="Training")
        ax.scatter(test_df["Experimental"], test_df["Predicted"], s=18, c=color, alpha=0.78, edgecolors="black", linewidth=0.18, label="Testing")
        vals = pd.concat([best_df["Experimental"], best_df["Predicted"]]).to_numpy(dtype=float)
        pad = max((np.nanmax(vals) - np.nanmin(vals)) * 0.07, 0.1)
        lims = [np.nanmin(vals) - pad, np.nanmax(vals) + pad]
        ax.plot(lims, lims, "k-", lw=1.05, label="Ideal")
        ax.plot(lims, [v * 1.2 for v in lims], "k--", lw=0.7, alpha=0.55)
        ax.plot(lims, [v * 0.8 for v in lims], "k--", lw=0.7, alpha=0.55)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Experimental {spec.label}", fontsize=9)
        ax.set_ylabel(f"Predicted {spec.label}", fontsize=9)
        ax.set_title(f"({chr(97 + idx)}) {spec.label}: {best_name}", fontsize=11, fontweight="bold", loc="left")
        ax.grid(alpha=0.24, ls="--")
        ax.tick_params(labelsize=8.2)
        if not metric_row.empty:
            text = (
                f"R2 = {metric_row['Test_R2']:.4f}\n"
                f"RMSE = {metric_row['Test_RMSE']:.4g}\n"
                f"MAE = {metric_row['Test_MAE']:.4g}\n"
                f"MAPE = {metric_row['Test_MAPE']:.2f}%"
            )
            ax.text(
                0.04,
                0.96,
                text,
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#777777", alpha=0.90),
            )
        if idx == 0:
            ax.legend(loc="lower right", fontsize=8, frameon=True, edgecolor="black")
    fig.suptitle("Best-model parity plots for Mg–Ni hydrogen-storage properties", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_pubfig(fig, fig_dir, "Fig4_BestModel_Parity_Plots")


def plot_prediction_scatter_grid(
    pred_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    spec: TargetSpec,
    fig_dir: Path,
    stem: str,
) -> None:
    model_order = [name for name in MODEL_COLORS if name in set(pred_df["Model"])]
    model_order.extend([name for name in pred_df["Model"].unique() if name not in model_order])
    model_order = model_order[:9]
    fig, axes = plt.subplots(3, 3, figsize=(12.2, 11.2))
    axes = axes.ravel()
    for idx, model_name in enumerate(model_order):
        ax = axes[idx]
        model_df = pred_df[pred_df["Model"] == model_name]
        train_df = model_df[model_df["Split"] == "Train"]
        test_df = model_df[model_df["Split"] == "Test"]
        color = MODEL_COLORS.get(model_name, "#4c72b0")
        ax.scatter(train_df["Experimental"], train_df["Predicted"], s=7, c="#6f7f86", alpha=0.25, label="Training set")
        ax.scatter(test_df["Experimental"], test_df["Predicted"], s=11, c=color, alpha=0.72, edgecolors="black", linewidth=0.15, label="Testing set")
        vals = pd.concat([model_df["Experimental"], model_df["Predicted"]]).to_numpy(dtype=float)
        pad = max((np.nanmax(vals) - np.nanmin(vals)) * 0.07, 0.1)
        lims = [np.nanmin(vals) - pad, np.nanmax(vals) + pad]
        ax.plot(lims, lims, "k-", lw=1.0)
        ax.plot(lims, [v * 1.2 for v in lims], "k--", lw=0.65, alpha=0.55)
        ax.plot(lims, [v * 0.8 for v in lims], "k--", lw=0.65, alpha=0.55)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"({idx + 1}) {model_name}", fontsize=10, fontweight="bold", loc="left")
        ax.set_xlabel(f"Experimental {spec.label}", fontsize=8)
        ax.set_ylabel(f"Predicted {spec.label}", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.22, ls="--")
        if not metrics_df.empty:
            hit = metrics_df[metrics_df["Model"] == model_name]
            if not hit.empty:
                row = hit.iloc[0]
                text = f"R2={row['Test_R2']:.4f}\nRMSE={row['Test_RMSE']:.4g}\nMAE={row['Test_MAE']:.4g}\nMAPE={row['Test_MAPE']:.2f}%"
                ax.text(
                    0.04,
                    0.96,
                    text,
                    transform=ax.transAxes,
                    va="top",
                    fontsize=7.2,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#777777", alpha=0.88),
                )
        if idx == 0:
            ax.legend(loc="lower right", fontsize=7, frameon=True, edgecolor="black")
    for ax in axes[len(model_order) :]:
        ax.axis("off")
    fig.suptitle(f"Predicted versus experimental values for {spec.label}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_pubfig(fig, fig_dir, stem)


def make_scatter_grid_from_paper_style(src_path: Path, png_path: Path, pdf_path: Path) -> None:
    if not HAS_PIL:
        return
    with Image.open(src_path) as image:
        image = image.convert("RGB")
        w, h = image.size
        x0 = int(w * 0.120)
        x1 = int(w * 0.405)
        top = int(h * 0.045)
        bottom = int(h * 0.987)
        row_h = (bottom - top) / 9.0
        panels = []
        for idx in range(9):
            y0 = int(top + idx * row_h)
            y1 = int(top + (idx + 1) * row_h)
            crop = image.crop((x0, max(0, y0 - int(row_h * 0.03)), x1, min(h, y1 + int(row_h * 0.03))))
            panels.append(crop)

        panel_w = max(p.width for p in panels)
        panel_h = max(p.height for p in panels)
        pad_x = 70
        pad_y = 70
        canvas = Image.new("RGB", (panel_w * 3 + pad_x * 4, panel_h * 3 + pad_y * 4), "white")
        for idx, panel in enumerate(panels):
            row = idx // 3
            col = idx % 3
            x = pad_x + col * (panel_w + pad_x)
            y = pad_y + row * (panel_h + pad_y)
            canvas.paste(panel, (x, y))
        canvas.save(png_path, dpi=(300, 300))
        canvas.save(pdf_path, "PDF", resolution=300)


def plot_publication_prediction_scatter_matrices(output_dir: Path, target_keys: list[str]) -> None:
    """Build the final-manuscript Fig. 4-Fig. 6 3 x 3 scatter matrices."""
    fig_dir = make_publication_dir(output_dir)
    stems = {
        "capacity": "Figure_04_Capacity_Prediction_Error",
        "pressure": "Figure_05_Pressure_Prediction_Error",
        "temperature": "Figure_06_Temperature_Prediction_Error",
    }
    for key in target_keys:
        if key not in stems:
            continue
        spec = TARGET_SPECS[key]
        pred_path = prediction_csv_path(output_dir, spec)
        metrics_path = metric_csv_path(output_dir, spec)
        if pred_path.exists() and metrics_path.exists():
            pred_df = pd.read_csv(pred_path)
            metrics_df = pd.read_csv(metrics_path)
            plot_prediction_scatter_grid(pred_df, metrics_df, spec, fig_dir, stems[key])
            continue

        paper_style = output_dir / spec.key / f"Fig_PaperStyle_{safe_name(spec.label)}.png"
        if paper_style.exists() and HAS_PIL:
            make_scatter_grid_from_paper_style(
                paper_style,
                fig_dir / f"{stems[key]}.png",
                fig_dir / f"{stems[key]}.pdf",
            )
        else:
            print(f"Final manuscript scatter matrix skipped for {spec.label}: prediction data were not found.")


def plot_publication_ablation_summary(output_dir: Path, target_keys: list[str]) -> None:
    fig_dir = make_publication_dir(output_dir)
    frames = []
    for key in target_keys:
        spec = TARGET_SPECS[key]
        path = ablation_csv_path(output_dir, spec)
        if path.exists():
            df = pd.read_csv(path)
            df.insert(0, "Target", spec.label)
            frames.append((spec, df))
    if not frames:
        print("Publication ablation summary skipped: no ablation CSV files were found.")
        return

    fig, axes = plt.subplots(1, len(frames), figsize=(5.8 * len(frames), 4.9), sharey=True)
    if len(frames) == 1:
        axes = [axes]
    for i, (spec, df) in enumerate(frames):
        ax = axes[i]
        labels = df["Subset"].str.replace("_", " ", regex=False)
        colors = ["#111111" if v == "Full" else "#6aaed6" for v in df["Subset"]]
        ax.bar(np.arange(len(df)), df["R2"], color=colors, edgecolor="black", linewidth=0.7)
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Test R2")
        ax.set_title(f"({chr(97 + i)}) {spec.label}", fontweight="bold")
        ax.grid(axis="y", alpha=0.25, ls="--")
        for x, value in enumerate(df["R2"]):
            ax.text(x, min(value + 0.025, 1.02), f"{value:.3f}", ha="center", fontsize=8)
    fig.suptitle("Feature ablation study across target properties", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_pubfig(fig, fig_dir, "Figure_07_Ablation_Study")


def plot_publication_active_learning_summary(output_dir: Path) -> None:
    fig_dir = make_publication_dir(output_dir)
    path = output_dir / "capacity" / "active_learning_screened_candidates_capacity.csv"
    if not path.exists():
        print("Publication UCB screening summary skipped: screening CSV was not found.")
        return

    screened = pd.read_csv(path)
    if screened.empty:
        return
    plot_df = screened.sample(n=min(len(screened), 5000), random_state=RANDOM_STATE) if len(screened) > 5000 else screened
    top20 = screened.nsmallest(20, "Rank").copy()

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    sc = axes[0].scatter(
        plot_df["Predicted_Ab_max"],
        plot_df["Uncertainty"],
        c=plot_df["Acquisition_UCB"],
        cmap="viridis",
        s=12,
        alpha=0.72,
        edgecolors="none",
    )
    axes[0].set_xlabel("Predicted capacity (wt%)")
    axes[0].set_ylabel("Ensemble uncertainty")
    axes[0].set_title("(a) UCB screening map", fontweight="bold")
    axes[0].grid(alpha=0.25, ls="--")
    cbar = fig.colorbar(sc, ax=axes[0], shrink=0.84)
    cbar.set_label("UCB score")

    axes[1].barh(top20["Formula"][::-1], top20["Acquisition_UCB"][::-1], color="#009e73", edgecolor="black", linewidth=0.45)
    axes[1].set_xlabel("UCB score")
    axes[1].set_title("(b) Top-20 prioritized candidates", fontweight="bold")
    axes[1].tick_params(axis="y", labelsize=7)
    axes[1].grid(axis="x", alpha=0.25, ls="--")

    round_counts = screened.groupby("Active_Round")["Rank"].count().sort_index()
    axes[2].bar(round_counts.index.astype(str), round_counts.values, color="#f0a202", edgecolor="black", linewidth=0.6)
    axes[2].set_xlabel("Screening round")
    axes[2].set_ylabel("Selected candidates")
    axes[2].set_title("(c) Round-wise retained candidates", fontweight="bold")
    axes[2].grid(axis="y", alpha=0.25, ls="--")

    fig.suptitle("Uncertainty-aware active-learning results", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_pubfig(fig, fig_dir, "Figure_08_Active_Learning")

    element_cols = [c for c in ACTIVE_ELEMENT_POOL if c in screened.columns]
    top10 = screened.nsmallest(10, "Rank").copy()
    comp = top10.set_index("Formula")[element_cols]
    comp = comp.loc[:, comp.sum(axis=0) > 0]
    if not comp.empty:
        fig, ax = plt.subplots(figsize=(11.5, max(4.5, 0.42 * len(comp))))
        sns.heatmap(comp, cmap="YlOrRd", linewidths=0.35, linecolor="white", cbar_kws={"label": "Atomic fraction (%)"}, ax=ax)
        ax.set_xlabel("Alloying element")
        ax.set_ylabel("Candidate formula")
        ax.set_title("Composition heatmap for top active-learning candidates", fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", labelrotation=0)
        save_pubfig(fig, fig_dir, "Figure_09_Candidate_Composition")


def plot_publication_shap_beeswarm_summary(output_dir: Path, target_keys: list[str]) -> None:
    fig_dir = make_publication_dir(output_dir)
    panels = []
    for key in target_keys:
        spec = TARGET_SPECS[key]
        path = output_dir / spec.key / f"Fig_SHAP_Beeswarm_{safe_name(spec.label)}.png"
        if path.exists():
            panels.append((spec.label, path))

    if not panels:
        print("Publication SHAP beeswarm summary skipped: no target-level SHAP figures were found.")
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(10.6, 5.9 * len(panels)))
    if len(panels) == 1:
        axes = [axes]
    for idx, (label, path) in enumerate(panels):
        ax = axes[idx]
        image = plt.imread(path)
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"({chr(97 + idx)}) {label}", fontsize=12, fontweight="bold", loc="left", pad=8)
    fig.suptitle("SHAP summary bee swarm plot of the best model", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    save_pubfig(fig, fig_dir, "Figure_10_SHAP_Beeswarm_BestModels")


def generate_publication_outputs(
    output_dir: Path,
    target_keys: list[str],
    source_data: pd.DataFrame | None = None,
    run_outputs: dict | None = None,
) -> None:
    n_rows = len(source_data) if source_data is not None else None
    plot_publication_workflow(output_dir, n_rows=n_rows)
    plot_publication_correlation_panels(output_dir, target_keys, source_data)
    plot_publication_model_comparison(output_dir, target_keys)
    plot_publication_prediction_scatter_matrices(output_dir, target_keys)
    plot_publication_ablation_summary(output_dir, target_keys)
    plot_publication_active_learning_summary(output_dir)
    plot_publication_shap_beeswarm_summary(output_dir, target_keys)


def run_target(
    source_data: pd.DataFrame,
    spec: TargetSpec,
    args,
    output_dir: Path,
    models: OrderedDict,
    param_grids: dict[str, dict],
) -> dict:
    target_dir = output_dir / spec.key
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 80}\nTarget: {spec.label}\n{'=' * 80}")

    element_cols = ACTIVE_ELEMENT_POOL
    feature_data, feature_names, groups = build_feature_frame(source_data, spec, element_cols, require_target=True)
    X = feature_data[feature_names].to_numpy(dtype=float)
    y = feature_data[spec.target_col].to_numpy(dtype=float)
    indices = np.arange(len(feature_data))
    train_idx, test_idx = train_test_split(indices, test_size=args.test_size, random_state=RANDOM_STATE)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_n = scaler.fit_transform(X_train)
    X_test_n = scaler.transform(X_test)
    print(f"Rows: {len(feature_data)}, train: {len(train_idx)}, test: {len(test_idx)}")
    print(f"Feature count: {len(feature_names)}")
    print(f"Features: {feature_names}")

    results = train_and_evaluate(models, param_grids, X_train_n, X_test_n, y_train, y_test, args.n_iter, args.cv)
    print_results_table(results)
    metrics = results_to_frame(results)
    metrics.to_csv(target_dir / f"model_comparison_{safe_name(spec.label)}.csv", index=False, encoding="utf-8-sig")
    feature_data.to_csv(target_dir / f"processed_features_{safe_name(spec.label)}.csv", index=False, encoding="utf-8-sig")
    save_prediction_frame(results, y_train, y_test, train_idx, test_idx, spec, target_dir)

    if not args.no_plots:
        plot_correlation_heatmap(feature_data[feature_names], y, spec.label, target_dir)
        plot_paper_style_scatter_error(results, y_train, y_test, spec.label, target_dir)
        plot_radar_comparison(results, spec.label, target_dir)
        plot_std_comparison(results, y_train, y_test, spec.label, target_dir)
        plot_feature_importance(results, feature_names, spec.label, target_dir)
        best_name = max(results, key=lambda k: results[k]["test_metrics"]["R2"])
        if not args.no_shap:
            plot_shap_summary(best_name, results[best_name]["model"], X_train_n, feature_names, spec.label, target_dir)

    ablation_df = None
    if args.run_ablation:
        best_name = max(results, key=lambda k: results[k]["test_metrics"]["R2"])
        print(f"\nAblation study uses best model for {spec.label}: {best_name}")
        ablation_df = run_ablation_study(
            feature_data,
            y,
            feature_names,
            groups,
            element_cols,
            results[best_name]["model"],
            train_idx,
            test_idx,
            spec.label,
            target_dir,
        )

    return {
        "spec": spec,
        "feature_data": feature_data,
        "feature_names": feature_names,
        "groups": groups,
        "results": results,
        "metrics": metrics,
        "ablation": ablation_df,
        "scaler": scaler,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrated Mg-Ni ML workflow for capacity, pressure, temperature, ablation, and uncertainty-guided screening."
    )
    parser.add_argument("--data", default="MgNi_Augmented_Data_Format.xlsx", help="Input MgNi dataset.")
    parser.add_argument("--output-dir", default="outputs/integrated_ml", help="Directory for all outputs.")
    parser.add_argument("--targets", default="all", help="Comma-separated targets: capacity,pressure,temperature or all.")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--cv", type=int, default=10)
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick development runs.")
    parser.add_argument("--run-ablation", action="store_true", default=True, help="Run feature ablation study.")
    parser.add_argument("--skip-ablation", dest="run_ablation", action="store_false")
    parser.add_argument("--run-active-learning", action="store_true", default=True, help="Run UCB-based virtual screening for capacity.")
    parser.add_argument("--skip-active-learning", dest="run_active_learning", action="store_false")
    parser.add_argument("--candidate-count", type=int, default=15000)
    parser.add_argument("--active-rounds", type=int, default=5)
    parser.add_argument("--query-size", type=int, default=3000)
    parser.add_argument("--ensemble-members", type=int, default=15)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument(
        "--publication-outputs",
        action="store_true",
        default=True,
        help="Generate paper-style combined result figures.",
    )
    parser.add_argument("--skip-publication-outputs", dest="publication_outputs", action="store_false")
    parser.add_argument(
        "--only-publication-outputs",
        action="store_true",
        help="Only build MSEB_submission_figures from an existing output directory.",
    )
    parser.add_argument("--quick", action="store_true", help="Small, fast smoke-test settings.")
    args = parser.parse_args()
    if args.quick:
        args.n_iter = min(args.n_iter, 2)
        args.cv = min(args.cv, 3)
        args.max_rows = args.max_rows or 500
        args.candidate_count = min(args.candidate_count, 500)
        args.query_size = min(args.query_size, 100)
        args.active_rounds = min(args.active_rounds, 3)
        args.ensemble_members = min(args.ensemble_members, 3)
        args.no_shap = True
    return args


def resolve_targets(target_text: str) -> list[str]:
    if target_text.strip().lower() == "all":
        return list(TARGET_SPECS.keys())
    keys = [x.strip().lower() for x in target_text.split(",") if x.strip()]
    invalid = [k for k in keys if k not in TARGET_SPECS]
    if invalid:
        raise ValueError(f"Invalid target keys: {invalid}. Use one of {list(TARGET_SPECS)} or all.")
    return keys


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = root / data_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_method_references(output_dir)

    print(f"Input data: {data_path}")
    print(f"Output directory: {output_dir}")
    target_keys = resolve_targets(args.targets)
    if args.only_publication_outputs:
        source_data = None
        if data_path.exists():
            source_data = standardize_dataset(data_path, max_rows=args.max_rows)
        generate_publication_outputs(output_dir, target_keys, source_data=source_data, run_outputs=None)
        print("\nPublication outputs generated from existing result files.")
        return

    data = standardize_dataset(data_path, max_rows=args.max_rows)
    data.to_csv(output_dir / "standardized_input_data.csv", index=False, encoding="utf-8-sig")
    print(f"Loaded valid Mg-Ni rows: {len(data)}")

    models, param_grids, model_notes = get_models_and_params(quick=args.quick)
    print(f"\nUsing {len(models)} models: {list(models.keys())}")
    if model_notes:
        for model_name, note in model_notes.items():
            print(f"  Note for {model_name}: {note}")
        (output_dir / "model_fallback_notes.json").write_text(json.dumps(model_notes, indent=2), encoding="utf-8")

    run_outputs = {}
    for target_key in target_keys:
        spec = TARGET_SPECS[target_key]
        run_outputs[target_key] = run_target(data, spec, args, output_dir, models, param_grids)

    if args.run_active_learning and "capacity" in run_outputs:
        cap = run_outputs["capacity"]
        run_active_learning_screening(
            data,
            cap["feature_data"],
            cap["feature_names"],
            cap["results"],
            args.candidate_count,
            args.active_rounds,
            args.query_size,
            args.ensemble_members,
            output_dir / "capacity",
        )
    elif args.run_active_learning:
        print("UCB screening is capacity-oriented; run with --targets capacity or --targets all to enable it.")

    summary = []
    for key, info in run_outputs.items():
        best = info["metrics"].sort_values("Test_R2", ascending=False).iloc[0]
        summary.append(
            {
                "Target": info["spec"].label,
                "Best_Model": best["Model"],
                "Test_R2": float(best["Test_R2"]),
                "Test_RMSE": float(best["Test_RMSE"]),
                "Test_MAE": float(best["Test_MAE"]),
                "Test_MAPE": float(best["Test_MAPE"]),
            }
        )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / "summary_best_models.csv", index=False, encoding="utf-8-sig")
    print("\nBest-model summary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if args.publication_outputs and not args.no_plots:
        generate_publication_outputs(output_dir, target_keys, source_data=data, run_outputs=run_outputs)
        print(f"\nPublication figures saved under: {output_dir / 'MSEB_submission_figures'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
