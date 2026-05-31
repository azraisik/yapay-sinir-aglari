from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
OUTPUTS_DIR = BASE_DIR / "outputs"

RISK_CONFIG_FILE = "sla_risk_model_config.json"
DURATION_CONFIG_FILE = "duration_model_config.json"
YSA_CONFIG_FILE = "ysa_model_config.json"
YSA_MODEL_FILE = "ysa_sla_risk_model.keras"
YSA_NUMPY_MODEL_FILE = "ysa_sla_risk_weights.npz"
YSA_PREPROCESSOR_FILE = "ysa_preprocessor.pkl"
YSA_SCALER_FILE = "ysa_scaler.pkl"
YSA_ENCODER_FILE = "ysa_encoder.pkl"
YSA_FEATURE_LIST_FILE = "ysa_feature_list.json"

YSA_MODEL_CANDIDATES = [
    YSA_NUMPY_MODEL_FILE,
    YSA_MODEL_FILE,
    "ysa_sla_risk_model.h5",
    "ysa_sla_risk_pipeline.pkl",
    "ysa_sla_risk_model.pkl",
    "ann_sla_risk_model.keras",
    "ann_sla_risk_model.h5",
]
YSA_NEURAL_MODEL_SUFFIXES = {".keras", ".h5", ".npz"}

YSA_PREPROCESSING_MISSING_MESSAGE = (
    "YSA tahmini için gerekli ön işleme dosyaları bulunamadı. "
    "Lütfen eğitimde kullanılan scaler/encoder/feature list dosyalarını ekleyin."
)

DEFAULT_RISK_THRESHOLDS = {"low": 0.40, "high": 0.70}
DEFAULT_SLA_LIMITS = {"High": 8, "Medium": 24, "Low": 48}

CASE_ID_COL = "Dosya No"
RISK_SCORE_COL = "SLA İhlal Risk Skoru"
RISK_PERCENT_COL = "SLA İhlal Riski (%)"
RISK_LEVEL_COL = "Risk Seviyesi"
DURATION_COL = "Tahmini Çözüm Süresi"
SLA_LIMIT_COL = "SLA Limiti"
SLA_STATUS_COL = "SLA Durumu"
SLA_DELTA_COL = "SLA Farkı"
ACTION_COL = "Önerilen Aksiyon"
REASON_COL = "Risk Nedenleri"

ACTION_EXPLANATIONS = {
    "Dosya sahibi sabitlenmeli": "Yeniden atama sayısı yüksek olduğu için dosyanın tek bir sorumlu üzerinden takip edilmesi önerilir.",
    "İş yükü yeniden dağıtılmalı": "İş yükü göstergesi yüksek olduğu için ekip kapasitesi yeniden değerlendirilmelidir.",
    "Yönetici kontrolüne alınmalı": "Escalation yaşandığı için dosyanın yönetici tarafından izlenmesi önerilir.",
    "Ek kaynak veya yönetici desteği önerilir": "Tahmini çözüm süresi SLA limitini aştığı için ek destek planlanmalıdır.",
    "Öncelikli müdahale gerekli": "SLA ihlal riski yüksek olduğu için dosyanın öncelikli olarak ele alınması önerilir.",
    "Yakından takip edilmeli": "Risk orta seviyede olduğu için dosyanın düzenli aralıklarla kontrol edilmesi önerilir.",
    "Normal süreçte izlenebilir": "Dosya mevcut göstergelere göre kritik risk taşımamaktadır.",
}


st.set_page_config(
    page_title="SLA Risk Takip Paneli",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --navy: #0f2747;
        --muted-navy: #2b4363;
        --border: #d8e0eb;
        --soft-bg: #f6f8fb;
        --card-bg: #ffffff;
        --text-muted: #65758b;
        --green-bg: #e7f6ed;
        --green-fg: #16703a;
        --orange-bg: #fff3d8;
        --orange-fg: #936200;
        --red-bg: #fde8e8;
        --red-fg: #a51d2d;
        --blue-bg: #e9f0fb;
        --blue-fg: #153e75;
    }
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1440px;
    }
    h1, h2, h3 {
        color: var(--navy);
        letter-spacing: 0;
    }
    .page-title {
        font-size: 1.72rem;
        font-weight: 700;
        color: var(--navy);
        margin-bottom: 0.12rem;
    }
    .page-subtitle {
        color: var(--text-muted);
        font-size: 0.94rem;
        margin-bottom: 0.95rem;
    }
    .metric-card {
        border: 1px solid var(--border);
        background: var(--card-bg);
        border-radius: 8px;
        padding: 0.92rem 1rem 0.82rem 1rem;
        min-height: 122px;
        box-shadow: 0 2px 8px rgba(16, 35, 63, 0.06);
        border-left: 4px solid #244c7a;
    }
    .metric-label {
        color: var(--text-muted);
        font-size: 0.74rem;
        font-weight: 700;
        margin-bottom: 0.48rem;
        text-transform: uppercase;
    }
    .metric-value {
        color: var(--navy);
        font-size: 1.72rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .metric-help {
        color: #7b8ca3;
        font-size: 0.78rem;
        margin-top: 0.42rem;
        line-height: 1.25;
    }
    .panel {
        border: 1px solid var(--border);
        background: var(--card-bg);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 5px rgba(16, 35, 63, 0.04);
    }
    .badge {
        display: inline-block;
        border-radius: 999px;
        padding: 0.22rem 0.58rem;
        font-size: 0.78rem;
        font-weight: 700;
        border: 1px solid transparent;
        white-space: nowrap;
    }
    .badge-low {
        background: var(--green-bg);
        color: var(--green-fg);
        border-color: #bfe7cb;
    }
    .badge-medium {
        background: var(--orange-bg);
        color: var(--orange-fg);
        border-color: #f3d487;
    }
    .badge-high {
        background: var(--red-bg);
        color: var(--red-fg);
        border-color: #f2b6b8;
    }
    .badge-neutral {
        background: var(--blue-bg);
        color: var(--blue-fg);
        border-color: #c7d7ef;
    }
    .detail-label {
        color: var(--text-muted);
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.18rem;
    }
    .detail-value {
        color: var(--navy);
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }
    .summary-card {
        border: 1px solid var(--border);
        background: white;
        border-radius: 8px;
        padding: 0.86rem 0.95rem;
        min-height: 98px;
        box-shadow: 0 1px 5px rgba(16, 35, 63, 0.04);
    }
    .summary-label {
        color: var(--text-muted);
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 0.38rem;
        text-transform: uppercase;
    }
    .summary-value {
        color: var(--navy);
        font-size: 1.12rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .field-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        border-bottom: 1px solid #eef2f7;
        padding: 0.44rem 0;
    }
    .field-row:last-child {
        border-bottom: none;
    }
    .field-name {
        color: var(--text-muted);
        font-size: 0.84rem;
    }
    .field-value {
        color: var(--navy);
        font-size: 0.9rem;
        font-weight: 650;
        text-align: right;
    }
    .action-note {
        background: #f5f7fb;
        border: 1px solid var(--border);
        border-radius: 8px;
        color: #33455f;
        font-size: 0.9rem;
        line-height: 1.45;
        padding: 0.75rem 0.82rem;
        margin-top: 0.55rem;
    }
    div[data-testid="stSidebar"] {
        background-color: var(--soft-bg);
        border-right: 1px solid var(--border);
    }
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3 {
        color: var(--navy);
    }
    div[data-testid="stSidebar"] .stCaptionContainer {
        color: var(--text-muted);
    }
    div[data-baseweb="tag"] {
        background-color: #e9eef5 !important;
        border-color: #c8d4e3 !important;
    }
    div[data-baseweb="tag"] *,
    div[data-baseweb="tag"] span,
    div[data-baseweb="tag"] div {
        color: var(--navy) !important;
    }
    div[data-baseweb="tag"] svg,
    div[data-baseweb="tag"] path {
        color: var(--muted-navy) !important;
        fill: var(--muted-navy) !important;
    }
    [data-testid="stMultiSelect"] [data-baseweb="tag"] {
        background-color: #e9eef5 !important;
        border-color: #c8d4e3 !important;
    }
    [data-testid="stMultiSelect"] [data-baseweb="tag"] * {
        color: var(--navy) !important;
    }
    [data-testid="stMultiSelect"] [data-baseweb="tag"] svg,
    [data-testid="stMultiSelect"] [data-baseweb="tag"] path {
        color: var(--muted-navy) !important;
        fill: var(--muted-navy) !important;
    }
    div[data-testid="stMetricValue"] {
        color: var(--navy);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def path_candidates(file_name: str | Path, primary_dir: Path) -> list[Path]:
    value = str(file_name).strip()
    if not value:
        return []

    raw_path = Path(value).expanduser()
    if raw_path.is_absolute():
        candidates = [
            raw_path,
            primary_dir / raw_path.name,
            BASE_DIR / raw_path.name,
        ]
    elif raw_path.parent == Path("."):
        candidates = [
            primary_dir / raw_path.name,
            BASE_DIR / raw_path.name,
        ]
    else:
        candidates = [
            BASE_DIR / raw_path,
            primary_dir / raw_path.name,
            primary_dir / raw_path,
            BASE_DIR / raw_path.name,
        ]

    resolved_base = BASE_DIR.resolve()
    safe_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if not raw_path.is_absolute():
            try:
                resolved.relative_to(resolved_base)
            except ValueError:
                continue
        key = str(resolved).casefold()
        if key not in seen:
            safe_candidates.append(resolved)
            seen.add(key)
    return safe_candidates


def resolve_file_path(
    file_name: str | Path,
    primary_dir: Path,
    required: bool = True,
) -> Path | None:
    candidates = path_candidates(file_name, primary_dir)
    for path in candidates:
        if path.exists():
            return path

    if required:
        name = Path(str(file_name).strip()).name or str(file_name)
        searched = ", ".join(str(path) for path in candidates)
        detail = f" Aranan konumlar: {searched}" if searched else ""
        raise FileNotFoundError(f"{name} bulunamadı.{detail}")
    return None


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path.name} bulunamadı.")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def load_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    risk_config_path = resolve_file_path(RISK_CONFIG_FILE, CONFIG_DIR)
    duration_config_path = resolve_file_path(DURATION_CONFIG_FILE, CONFIG_DIR)
    return (
        read_json(risk_config_path),
        read_json(duration_config_path),
    )


@st.cache_data(show_spinner=False)
def load_csv(file_name: str | Path) -> pd.DataFrame:
    path = resolve_file_path(file_name, DATA_DIR)
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_model(file_name: str | Path) -> Any:
    path = resolve_file_path(file_name, MODELS_DIR)
    return joblib.load(path)


def existing_file_name(file_names: list[str], primary_dir: Path) -> str | None:
    for file_name in file_names:
        if not file_name:
            continue
        path = resolve_file_path(file_name, primary_dir, required=False)
        if path is not None:
            return str(path)
    return None


@st.cache_resource(show_spinner=False)
def load_ysa_model(file_name: str | Path) -> Any:
    path = resolve_file_path(file_name, MODELS_DIR)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        layers = []
        with np.load(path) as data:
            layer_index = 0
            while f"W{layer_index}" in data and f"b{layer_index}" in data:
                layers.append((
                    np.asarray(data[f"W{layer_index}"], dtype=np.float32),
                    np.asarray(data[f"b{layer_index}"], dtype=np.float32),
                ))
                layer_index += 1
        if not layers:
            raise ValueError("YSA NumPy ağırlık dosyasında katman bilgisi bulunamadı.")
        return {"model_type": "numpy_dense_sigmoid", "layers": layers}
    if suffix in {".keras", ".h5"}:
        try:
            from tensorflow.keras.models import load_model as keras_load_model
        except Exception as exc:
            raise RuntimeError(
                "YSA modeli Keras formatında, ancak TensorFlow/Keras yüklenemedi."
            ) from exc
        return keras_load_model(path)
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_ysa_joblib(file_name: str | Path) -> Any:
    path = resolve_file_path(file_name, MODELS_DIR)
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_ysa_feature_list_file(file_name: str | Path) -> list[str]:
    path = resolve_file_path(file_name, CONFIG_DIR, required=False)
    if path is None:
        return []
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, dict):
            for key in (
                "model_features",
                "input_features",
                "processed_features",
                "encoded_features",
                "one_hot_columns",
                "feature_names",
                "feature_names_out",
            ):
                items = value.get(key)
                if isinstance(items, list):
                    return [str(item) for item in items]
        return []
    if suffix in {".csv", ".txt"}:
        text = path.read_text(encoding="utf-8")
        return [item.strip() for item in text.replace("\n", ",").split(",") if item.strip()]
    return []


def normalize_name(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("ı", "i")
    )


def find_case_column(df: pd.DataFrame) -> str | None:
    candidates = {
        "case id",
        "caseid",
        "case no",
        "case number",
        "dosya no",
        "dosya numarasi",
        "dosya id",
        "ticket id",
        "incident id",
    }
    normalized = {normalize_name(col): col for col in df.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for col in df.columns:
        name = normalize_name(col)
        if ("case" in name and "id" in name) or ("dosya" in name and ("no" in name or "id" in name)):
            return col
    return None


def ensure_case_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    case_col = find_case_column(df)
    if case_col is None:
        df[CASE_ID_COL] = [f"Dosya-{idx:03d}" for idx in range(1, len(df) + 1)]
    elif case_col != CASE_ID_COL:
        df = df.rename(columns={case_col: CASE_ID_COL})
    df[CASE_ID_COL] = df[CASE_ID_COL].astype(str)
    return df


def get_list(config: dict[str, Any], key: str) -> list[str]:
    value = config.get(key, [])
    return value if isinstance(value, list) else []


def validate_columns(df: pd.DataFrame, columns: list[str], label: str) -> list[str]:
    return [col for col in columns if col not in df.columns]


def get_thresholds(config: dict[str, Any]) -> dict[str, float]:
    thresholds = config.get("risk_thresholds", {})
    if not isinstance(thresholds, dict):
        thresholds = {}
    low = float(thresholds.get("low", DEFAULT_RISK_THRESHOLDS["low"]))
    high = float(thresholds.get("high", DEFAULT_RISK_THRESHOLDS["high"]))
    if high <= low:
        return DEFAULT_RISK_THRESHOLDS.copy()
    return {"low": low, "high": high}


def get_sla_limits(config: dict[str, Any]) -> dict[str, float]:
    limits = config.get("sla_limits_hours", {})
    if not isinstance(limits, dict):
        limits = {}
    merged = DEFAULT_SLA_LIMITS.copy()
    for key, value in limits.items():
        try:
            merged[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return merged


def feature_by_name(features: list[str], target: str) -> str | None:
    target_name = normalize_name(target)
    for feature in features:
        if normalize_name(feature) == target_name:
            return feature
    return None


def config_list(config: dict[str, Any], keys: list[str]) -> list[str]:
    for key in keys:
        items = get_list(config, key)
        if items:
            return [str(item) for item in items]
    return []


def load_ysa_config(risk_config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    config_path = resolve_file_path(YSA_CONFIG_FILE, CONFIG_DIR, required=False)
    if config_path is not None:
        config = read_json(config_path)
        source = YSA_CONFIG_FILE
    else:
        config = dict(risk_config)
        config.pop("model_file", None)
        source = RISK_CONFIG_FILE

    model_features = get_list(config, "model_features") or get_list(risk_config, "model_features")
    categorical_features = get_list(config, "categorical_features") or get_list(
        risk_config, "categorical_features"
    )
    numeric_features = get_list(config, "numeric_features") or get_list(risk_config, "numeric_features")

    config["model_features"] = model_features
    config["categorical_features"] = categorical_features
    config["numeric_features"] = numeric_features
    config.setdefault("case_file", risk_config.get("case_file"))
    return config, source


def ysa_model_file_name(config: dict[str, Any]) -> str | None:
    configured = str(config.get("model_file", "")).strip()
    numpy_model = str(config.get("numpy_model_file", YSA_NUMPY_MODEL_FILE)).strip()
    candidates = [numpy_model]
    if configured:
        candidates.append(configured)
    candidates.extend(YSA_MODEL_CANDIDATES)
    return existing_file_name(candidates, MODELS_DIR)


def ysa_artifact_file(
    config: dict[str, Any],
    keys: list[str],
    default_file: str,
    primary_dir: Path = MODELS_DIR,
) -> str | None:
    candidates = [str(config.get(key, "")).strip() for key in keys]
    candidates.append(default_file)
    return existing_file_name([item for item in candidates if item], primary_dir)


def ysa_expected_features(config: dict[str, Any]) -> list[str]:
    features = config_list(
        config,
        [
            "input_features",
            "processed_features",
            "encoded_features",
            "one_hot_columns",
            "feature_names",
            "feature_names_out",
        ],
    )
    if features:
        return features

    feature_file = ysa_artifact_file(
        config,
        [
            "feature_list_file",
            "input_features_file",
            "processed_features_file",
            "one_hot_columns_file",
        ],
        YSA_FEATURE_LIST_FILE,
        CONFIG_DIR,
    )
    return load_ysa_feature_list_file(feature_file) if feature_file else []


def load_ysa_reference_data(config: dict[str, Any], risk_config: dict[str, Any]) -> pd.DataFrame:
    candidates = [
        str(config.get("case_file", "")).strip(),
        str(config.get("training_file", "")).strip(),
        str(risk_config.get("case_file", "")).strip(),
    ]
    file_name = existing_file_name([item for item in candidates if item], DATA_DIR)
    if not file_name:
        return pd.DataFrame()
    try:
        return load_csv(file_name)
    except Exception:
        return pd.DataFrame()


def unique_options(df: pd.DataFrame, column: str) -> list[Any]:
    if column not in df.columns:
        return []
    values = df[column].dropna().unique().tolist()
    return sorted(values, key=lambda value: str(value))


def numeric_default(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return 0.0
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(values.median())


def is_binary_feature(feature: str, df: pd.DataFrame) -> bool:
    normalized = normalize_name(feature)
    if normalized.startswith("is ") or normalized.startswith("has "):
        return True
    if feature in df.columns:
        values = pd.to_numeric(df[feature], errors="coerce").dropna().unique().tolist()
        return bool(values) and set(values).issubset({0, 1})
    return False


def manual_ysa_preprocess(
    raw_input: pd.DataFrame,
    config: dict[str, Any],
    expected_features: list[str],
) -> pd.DataFrame:
    if not expected_features:
        raise ValueError(YSA_PREPROCESSING_MISSING_MESSAGE)

    training_file = str(config.get("training_file", "")).strip()
    if not training_file:
        raise ValueError(YSA_PREPROCESSING_MISSING_MESSAGE)

    training_df = load_csv(training_file)
    model_features = get_list(config, "model_features")
    categorical_features = get_list(config, "categorical_features")
    numeric_features = get_list(config, "numeric_features")
    target = str(config.get("target", "")).strip()

    train_features = training_df[model_features].copy()
    if target and target in training_df.columns:
        try:
            from sklearn.model_selection import train_test_split

            y = training_df[target].astype(int)
            train_features, _test_features = train_test_split(
                train_features,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )
        except Exception:
            pass

    prepared = pd.DataFrame(0.0, index=raw_input.index, columns=expected_features)

    for feature in numeric_features:
        if feature not in prepared.columns:
            continue
        train_values = pd.to_numeric(train_features[feature], errors="coerce").dropna()
        mean = float(train_values.mean()) if not train_values.empty else 0.0
        scale = float(train_values.std(ddof=0)) if len(train_values) > 1 else 1.0
        if scale == 0:
            scale = 1.0
        if feature in raw_input.columns:
            input_values = pd.to_numeric(raw_input[feature], errors="coerce").fillna(mean)
        else:
            input_values = pd.Series(mean, index=raw_input.index)
        prepared[feature] = (input_values.astype(float) - mean) / scale

    for feature in categorical_features:
        if feature not in raw_input.columns:
            continue
        value = str(raw_input[feature].iloc[0])
        encoded_column = f"{feature}_{value}"
        if encoded_column in prepared.columns:
            prepared.loc[raw_input.index[0], encoded_column] = 1.0

    return prepared


def unpack_ysa_bundle(artifact: Any) -> tuple[Any, Any | None, list[str]]:
    if not isinstance(artifact, dict):
        return artifact, None, []

    model = artifact.get("model")
    if model is None:
        model = artifact.get("estimator")
    if model is None:
        model = artifact.get("classifier")

    pipeline = artifact.get("pipeline")
    preprocessor = artifact.get("preprocessor")
    if preprocessor is None:
        preprocessor = artifact.get("preprocessing")
    if preprocessor is None:
        preprocessor = artifact.get("transformer")
    if model is None and pipeline is not None and hasattr(pipeline, "predict"):
        model = pipeline
    elif preprocessor is None:
        preprocessor = pipeline
    features: list[str] = []
    for key in (
        "input_features",
        "processed_features",
        "encoded_features",
        "one_hot_columns",
        "feature_names",
        "feature_names_out",
    ):
        value = artifact.get(key)
        if isinstance(value, list):
            features = [str(item) for item in value]
            break
    return (model if model is not None else artifact), preprocessor, features


def to_model_array(values: Any) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    if isinstance(values, pd.DataFrame):
        values = values.to_numpy()
    return np.asarray(values)


def transformed_frame(values: Any, columns: list[str]) -> pd.DataFrame:
    array = to_model_array(values)
    if columns and array.ndim == 2 and array.shape[1] == len(columns):
        return pd.DataFrame(array, columns=columns)
    return pd.DataFrame(array)


def transformer_feature_names(transformer: Any, input_features: list[str]) -> list[str]:
    if hasattr(transformer, "get_feature_names_out"):
        try:
            return [str(item) for item in transformer.get_feature_names_out(input_features)]
        except TypeError:
            return [str(item) for item in transformer.get_feature_names_out()]
    return []


def align_expected_features(frame: pd.DataFrame, expected_features: list[str]) -> pd.DataFrame:
    if not expected_features:
        return frame
    missing = [feature for feature in expected_features if feature not in frame.columns]
    for feature in missing:
        frame[feature] = 0
    return frame[expected_features]


def keras_expected_width(model: Any) -> int | None:
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list) and shape:
        shape = shape[0]
    if isinstance(shape, tuple) and len(shape) >= 2 and shape[-1] is not None:
        return int(shape[-1])
    return None


def numpy_expected_width(model: Any) -> int | None:
    if not isinstance(model, dict) or model.get("model_type") != "numpy_dense_sigmoid":
        return None
    layers = model.get("layers", [])
    if not layers:
        return None
    first_weights, _first_bias = layers[0]
    if hasattr(first_weights, "shape") and len(first_weights.shape) == 2:
        return int(first_weights.shape[0])
    return None


def prepare_ysa_input(
    raw_input: pd.DataFrame,
    config: dict[str, Any],
    model_file: str,
    model: Any,
    bundled_preprocessor: Any | None,
    bundled_features: list[str],
) -> Any:
    suffix = Path(model_file).suffix.lower()
    model_features = get_list(config, "model_features")
    categorical_features = get_list(config, "categorical_features")
    numeric_features = get_list(config, "numeric_features")

    if suffix not in YSA_NEURAL_MODEL_SUFFIXES and bundled_preprocessor is None:
        return raw_input[model_features]

    expected_features = bundled_features or ysa_expected_features(config)

    preprocessor = bundled_preprocessor
    if preprocessor is None:
        preprocessor_file = ysa_artifact_file(
            config,
            ["preprocessor_file", "preprocessing_file", "transformer_file", "pipeline_file"],
            YSA_PREPROCESSOR_FILE,
        )
        if preprocessor_file:
            try:
                preprocessor = load_ysa_joblib(preprocessor_file)
            except Exception:
                preprocessor = None

    if preprocessor is not None:
        transformed = preprocessor.transform(raw_input[model_features])
        columns = expected_features or transformer_feature_names(preprocessor, model_features)
        prepared = transformed_frame(transformed, columns)
        if expected_features:
            prepared = align_expected_features(prepared, expected_features)
        values = to_model_array(prepared)
    else:
        try:
            prepared = manual_ysa_preprocess(raw_input, config, expected_features)
            values = to_model_array(prepared)
            expected_width = keras_expected_width(model) or numpy_expected_width(model)
            if expected_width is not None and values.ndim == 2 and values.shape[1] == expected_width:
                return values
        except Exception:
            pass

        encoder_file = ysa_artifact_file(
            config, ["encoder_file", "one_hot_encoder_file"], YSA_ENCODER_FILE
        )
        scaler_file = ysa_artifact_file(config, ["scaler_file", "standard_scaler_file"], YSA_SCALER_FILE)
        if (categorical_features and not encoder_file) or (numeric_features and not scaler_file) or not expected_features:
            raise ValueError(YSA_PREPROCESSING_MISSING_MESSAGE)

        pieces: list[pd.DataFrame] = []
        if numeric_features:
            scaler = load_ysa_joblib(str(scaler_file))
            numeric_values = raw_input[numeric_features].apply(pd.to_numeric, errors="coerce").fillna(0)
            pieces.append(pd.DataFrame(scaler.transform(numeric_values), columns=numeric_features))
        if categorical_features:
            encoder = load_ysa_joblib(str(encoder_file))
            encoded = encoder.transform(raw_input[categorical_features])
            encoded_names = transformer_feature_names(encoder, categorical_features)
            pieces.append(transformed_frame(encoded, encoded_names))

        prepared = pd.concat(pieces, axis=1) if pieces else pd.DataFrame()
        prepared = align_expected_features(prepared, expected_features)
        values = to_model_array(prepared)

    expected_width = keras_expected_width(model) or numpy_expected_width(model)
    if expected_width is not None and values.ndim == 2 and values.shape[1] != expected_width:
        raise ValueError(
            f"YSA modeli {expected_width} giriş bekliyor, hazırlanan veri {values.shape[1]} kolon içeriyor."
        )
    return values


def ysa_probability(model: Any, features: Any) -> float:
    if isinstance(model, dict) and model.get("model_type") == "numpy_dense_sigmoid":
        values = to_model_array(features).astype(np.float32)
        layers = model.get("layers", [])
        for layer_index, (weights, bias) in enumerate(layers):
            values = values @ weights + bias
            if layer_index < len(layers) - 1:
                values = np.maximum(values, 0)
            else:
                values = 1 / (1 + np.exp(-values))
        return float(np.ravel(values)[0])
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            classes = list(getattr(model, "classes_", []))
            class_index = classes.index(1) if 1 in classes else probabilities.shape[1] - 1
            return float(probabilities[0, class_index])
        return float(np.ravel(probabilities)[0])
    prediction = model.predict(features)
    return float(np.ravel(prediction)[0])


def ysa_result_level(score: float, thresholds: dict[str, float]) -> str:
    if score >= thresholds["high"]:
        return "Yüksek Risk"
    if score >= thresholds["low"]:
        return "Orta Risk"
    return "Düşük Risk"


def ysa_result_badge(level: str) -> str:
    if level.startswith("Yüksek"):
        return badge(level, "high")
    if level.startswith("Orta"):
        return badge(level, "medium")
    return badge(level, "low")


def ysa_comment(level: str, prediction_text: str) -> str:
    if level.startswith("Yüksek"):
        return "Bu dosya için SLA ihlal riski yüksek görünüyor; öncelikli takip önerilir."
    if level.startswith("Orta"):
        return "Bu dosya için risk orta seviyede; düzenli kontrol ve erken müdahale faydalı olur."
    if prediction_text == "İhlal Bekleniyor":
        return "Model ihlal bekliyor, ancak olasılık eşik altında; dosya yakın takipte tutulabilir."
    return "Bu dosya için mevcut girdilere göre SLA ihlali beklenmiyor."


def positive_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            classes = list(getattr(model, "classes_", []))
            class_index = classes.index(1) if 1 in classes else probabilities.shape[1] - 1
            return probabilities[:, class_index]
        return np.ravel(probabilities)
    if hasattr(model, "decision_function"):
        scores = np.ravel(model.decision_function(features))
        return 1 / (1 + np.exp(-scores))
    return np.ravel(model.predict(features)).astype(float)


def find_probability_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "SLA_Risk_Probability",
        "Risk_Probability",
        "Risk_Score",
        "SLA Risk Probability",
    ]
    normalized = {normalize_name(col): col for col in df.columns}
    for candidate in candidates:
        found = normalized.get(normalize_name(candidate))
        if found:
            return found
    for col in df.columns:
        name = normalize_name(col)
        if "risk" in name and ("probability" in name or "score" in name):
            return col
    return None


def risk_level(score: float, thresholds: dict[str, float]) -> str:
    if score >= thresholds["high"]:
        return "Yüksek"
    if score >= thresholds["low"]:
        return "Orta"
    return "Düşük"


def risk_level_from_percent(percent: float, thresholds: dict[str, float]) -> str:
    high_percent = thresholds["high"] * 100
    low_percent = thresholds["low"] * 100
    if percent >= high_percent:
        return "Yüksek"
    if percent >= low_percent:
        return "Orta"
    return "Düşük"


def sla_limit_for_priority(priority: Any, limits: dict[str, float]) -> float:
    priority_text = str(priority).strip().lower()
    for key, value in limits.items():
        if str(key).strip().lower() == priority_text:
            return float(value)
    return float(limits.get("Low", DEFAULT_SLA_LIMITS["Low"]))


def sla_status(delta_hours: float) -> str:
    if delta_hours < 0:
        return "Limit aşımı bekleniyor"
    if delta_hours <= 2:
        return "Limitine yakın"
    return "Limit içinde"


def quantile_threshold(df: pd.DataFrame, column: str, quantile: float, fallback: float) -> float:
    if column not in df.columns:
        return fallback
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return fallback
    return float(np.ceil(values.quantile(quantile)))


def process_thresholds(df: pd.DataFrame) -> dict[str, float]:
    return {
        "Step_count": quantile_threshold(df, "Step_count", 0.75, 8),
        "Reassignment_count": max(5, quantile_threshold(df, "Reassignment_count", 0.75, 5)),
        "Escalation_count": max(2, quantile_threshold(df, "Escalation_count", 0.80, 2)),
        "Workload_Index": quantile_threshold(df, "Workload_Index", 0.75, 40),
    }


def generate_action(row: pd.Series, thresholds: dict[str, float]) -> str:
    if row[SLA_DELTA_COL] < 0:
        return "Ek kaynak veya yönetici desteği önerilir"
    if row[RISK_LEVEL_COL] == "Yüksek":
        return "Öncelikli müdahale gerekli"
    if float(row.get("Escalation_count", 0)) >= thresholds["Escalation_count"]:
        return "Yönetici kontrolüne alınmalı"
    if float(row.get("Reassignment_count", 0)) >= thresholds["Reassignment_count"]:
        return "Dosya sahibi sabitlenmeli"
    if float(row.get("Workload_Index", 0)) >= thresholds["Workload_Index"]:
        return "İş yükü yeniden dağıtılmalı"
    if row[RISK_LEVEL_COL] == "Orta":
        return "Yakından takip edilmeli"
    return "Normal süreçte izlenebilir"


def action_explanation(action: str) -> str:
    return ACTION_EXPLANATIONS.get(action, "Dosya için operasyonel takip önerisi oluşturuldu.")


def risk_reasons(row: pd.Series, thresholds: dict[str, float]) -> list[str]:
    reasons: list[str] = []
    if float(row.get("Step_count", 0)) >= thresholds["Step_count"]:
        reasons.append("Adım sayısı yüksek")
    if float(row.get("Reassignment_count", 0)) >= thresholds["Reassignment_count"]:
        reasons.append("Yeniden atama sayısı yüksek")
    if float(row.get("Escalation_count", 0)) > 0:
        reasons.append("Escalation yaşanmış")
    if float(row.get("Workload_Index", 0)) >= thresholds["Workload_Index"]:
        reasons.append("İş yükü yüksek")
    if int(float(row.get("Is_Weekend", 0))) == 1:
        reasons.append("Hafta sonu açılmış dosya")
    if str(row.get("Priority", "")).strip().lower() == "high":
        reasons.append("Öncelik seviyesi yüksek")
    if row[SLA_DELTA_COL] < 0:
        reasons.append("Tahmini süre SLA limitini aşıyor")
    elif row[SLA_STATUS_COL] == "Limitine yakın":
        reasons.append("Tahmini süre SLA limitine yakın")
    if not reasons:
        reasons.append("Belirgin operasyonel risk göstergesi yok")
    return reasons


def badge(text: str, kind: str) -> str:
    return f"<span class='badge badge-{kind}'>{text}</span>"


def risk_badge(level: str) -> str:
    mapping = {"Düşük": "low", "Orta": "medium", "Yüksek": "high"}
    return badge(level, mapping.get(level, "neutral"))


def status_badge(status: str) -> str:
    if status == "Limit aşımı bekleniyor":
        return badge(status, "high")
    if status == "Limitine yakın":
        return badge(status, "medium")
    return badge(status, "low")


def style_risk(value: Any) -> str:
    if value == "Yüksek":
        return "background-color: #fde8e8; color: #a51d2d; font-weight: 700;"
    if value == "Orta":
        return "background-color: #fff3d8; color: #936200; font-weight: 700;"
    if value == "Düşük":
        return "background-color: #e7f6ed; color: #16703a; font-weight: 700;"
    return ""


def style_status(value: Any) -> str:
    if value == "Limit aşımı bekleniyor":
        return "background-color: #fde8e8; color: #a51d2d; font-weight: 700;"
    if value == "Limitine yakın":
        return "background-color: #fff3d8; color: #936200; font-weight: 700;"
    if value == "Limit içinde":
        return "background-color: #e7f6ed; color: #16703a; font-weight: 700;"
    return ""


def style_cells(styler: Any, func: Any, subset: list[str]) -> Any:
    if hasattr(styler, "map"):
        return styler.map(func, subset=subset)
    return styler.applymap(func, subset=subset)


@st.cache_data(show_spinner="Operasyon tablosu hazırlanıyor...")
def build_operations_table() -> tuple[pd.DataFrame, list[str]]:
    risk_config, duration_config = load_configs()
    notices: list[str] = []

    risk_case_file = risk_config.get("case_file")
    duration_case_file = duration_config.get("case_file")
    if not risk_case_file or not duration_case_file:
        raise ValueError("Config dosyalarında arayüz veri dosyası bilgisi bulunamadı.")

    risk_df = ensure_case_id(load_csv(risk_case_file))
    duration_df = ensure_case_id(load_csv(duration_case_file))

    risk_features = get_list(risk_config, "model_features")
    duration_features = get_list(duration_config, "model_features")

    missing_risk = validate_columns(risk_df, risk_features, "risk")
    missing_duration = validate_columns(duration_df, duration_features, "duration")
    if missing_risk or missing_duration:
        messages = []
        if missing_risk:
            messages.append(f"Risk verisinde eksik kolonlar: {', '.join(missing_risk)}")
        if missing_duration:
            messages.append(f"Süre verisinde eksik kolonlar: {', '.join(missing_duration)}")
        raise KeyError(" | ".join(messages))

    operations = risk_df.copy()

    try:
        risk_model = load_model(str(risk_config.get("model_file", "")))
        operations[RISK_SCORE_COL] = np.clip(positive_probability(risk_model, risk_df[risk_features]), 0, 1)
    except Exception:
        probability_col = find_probability_column(risk_df)
        if probability_col is None:
            raise RuntimeError("Risk modeli yüklenemedi ve hazır risk skoru kolonu bulunamadı.")
        operations[RISK_SCORE_COL] = pd.to_numeric(risk_df[probability_col], errors="coerce").fillna(0).clip(0, 1)
        notices.append("Risk tahmini için hazır skor çıktıları kullanılıyor.")

    duration_predictions = duration_df[[CASE_ID_COL]].copy()
    try:
        duration_model = load_model(str(duration_config.get("model_file", "")))
        duration_predictions[DURATION_COL] = duration_model.predict(duration_df[duration_features])
    except Exception:
        prediction_col = str(duration_config.get("prediction_column", ""))
        if prediction_col not in duration_df.columns:
            raise RuntimeError("Süre modeli yüklenemedi ve hazır süre tahmin kolonu bulunamadı.")
        duration_predictions[DURATION_COL] = pd.to_numeric(duration_df[prediction_col], errors="coerce")
        notices.append("Süre tahmini için hazır tahmin çıktıları kullanılıyor.")
    operations = operations.merge(duration_predictions, on=CASE_ID_COL, how="left")

    priority_col = feature_by_name(risk_features, "Priority") or "Priority"
    thresholds = get_thresholds(risk_config)
    limits = get_sla_limits(risk_config)

    operations[RISK_PERCENT_COL] = (operations[RISK_SCORE_COL] * 100).round(1)
    operations[RISK_LEVEL_COL] = operations[RISK_PERCENT_COL].apply(
        lambda value: risk_level_from_percent(float(value), thresholds)
    )
    operations[SLA_LIMIT_COL] = operations[priority_col].apply(lambda value: sla_limit_for_priority(value, limits))
    operations[SLA_DELTA_COL] = operations[SLA_LIMIT_COL] - operations[DURATION_COL]
    operations[SLA_STATUS_COL] = operations[SLA_DELTA_COL].apply(sla_status)

    operation_thresholds = process_thresholds(operations)
    operations[ACTION_COL] = operations.apply(lambda row: generate_action(row, operation_thresholds), axis=1)
    operations[REASON_COL] = operations.apply(lambda row: risk_reasons(row, operation_thresholds), axis=1)

    return operations, notices


def format_hours(value: float) -> str:
    return f"{value:.1f} saat"


def delta_text(value: float) -> str:
    if value < 0:
        return f"{abs(value):.1f} saat aşım bekleniyor"
    return f"{value:.1f} saat kaldı"


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()

    st.sidebar.markdown("### Filtreler")

    def apply_multiselect(column: str, label: str) -> None:
        nonlocal filtered
        if column not in filtered.columns:
            return
        options = sorted(filtered[column].dropna().astype(str).unique().tolist())
        selected = st.sidebar.multiselect(label, options, default=options)
        if selected:
            filtered = filtered[filtered[column].astype(str).isin(selected)]

    apply_multiselect("Priority", "Priority")
    apply_multiselect(RISK_LEVEL_COL, "Risk seviyesi")
    apply_multiselect("Issue Type", "Issue Type")
    apply_multiselect("Report Channel", "Report Channel")
    apply_multiselect(SLA_STATUS_COL, "SLA durumu")

    return filtered


def metric_card(label: str, value: str, helper: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_display_table(df: pd.DataFrame) -> pd.DataFrame:
    display_columns = [
        CASE_ID_COL,
        "Priority",
        "Issue Type",
        RISK_PERCENT_COL,
        RISK_LEVEL_COL,
        DURATION_COL,
        SLA_LIMIT_COL,
        SLA_STATUS_COL,
        ACTION_COL,
    ]
    available_columns = [col for col in display_columns if col in df.columns]
    table = df[available_columns].copy()
    if RISK_PERCENT_COL in table.columns:
        table[RISK_PERCENT_COL] = table[RISK_PERCENT_COL].round(1)
    if DURATION_COL in table.columns:
        table[DURATION_COL] = table[DURATION_COL].round(1)
    if SLA_LIMIT_COL in table.columns:
        table[SLA_LIMIT_COL] = table[SLA_LIMIT_COL].round(1)
    return table


def render_key_value_rows(fields: dict[str, Any]) -> None:
    for name, value in fields.items():
        left, right = st.columns([1.1, 1.4])
        with left:
            st.caption(str(name))
        with right:
            st.markdown(f"**{value}**")


def render_detail(selected_row: pd.Series) -> None:
    summary_cols = st.columns(4)
    with summary_cols[0]:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Risk Seviyesi</div>
                <div>{risk_badge(selected_row[RISK_LEVEL_COL])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with summary_cols[1]:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">SLA İhlal Riski</div>
                <div class="summary-value">%{selected_row[RISK_PERCENT_COL]:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with summary_cols[2]:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Tahmini Çözüm Süresi</div>
                <div class="summary-value">{format_hours(float(selected_row[DURATION_COL]))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with summary_cols[3]:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">SLA'ya Kalan / Aşan Süre</div>
                <div class="summary-value">{delta_text(float(selected_row[SLA_DELTA_COL]))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    info_left, info_right = st.columns([1, 1])
    with info_left:
        info_fields = {
            "Dosya No": selected_row.get(CASE_ID_COL, ""),
            "Priority": selected_row.get("Priority", ""),
            "Issue Type": selected_row.get("Issue Type", ""),
            "Report Channel": selected_row.get("Report Channel", ""),
            "Variant": selected_row.get("Variant", ""),
            "Step_count": selected_row.get("Step_count", ""),
            "Reassignment_count": selected_row.get("Reassignment_count", ""),
            "Escalation_count": selected_row.get("Escalation_count", ""),
            "Workload_Index": selected_row.get("Workload_Index", ""),
            "Open_Hour": selected_row.get("Open_Hour", ""),
            "Is_Weekend": selected_row.get("Is_Weekend", ""),
        }
        with st.container(border=True):
            st.markdown("#### Dosya Bilgileri")
            render_key_value_rows(info_fields)
    with info_right:
        reasons = selected_row[REASON_COL]
        if isinstance(reasons, str):
            reasons = [reasons]
        action = str(selected_row[ACTION_COL])
        operational_fields = {
            "SLA Limiti": format_hours(float(selected_row[SLA_LIMIT_COL])),
            "SLA Durumu": selected_row[SLA_STATUS_COL],
            "Önerilen Aksiyon": action,
        }
        with st.container(border=True):
            st.markdown("#### Operasyonel Değerlendirme")
            render_key_value_rows(operational_fields)
            st.info(action_explanation(action))
            st.markdown("**Riskin Olası Nedenleri**")
            for reason in reasons:
                st.markdown(f"- {str(reason)}")


def render_ysa_manual_prediction(risk_config: dict[str, Any]) -> None:
    ysa_config, config_source = load_ysa_config(risk_config)
    model_features = get_list(ysa_config, "model_features")
    categorical_features = get_list(ysa_config, "categorical_features")
    numeric_features = get_list(ysa_config, "numeric_features")
    thresholds = get_thresholds(risk_config)
    reference_df = load_ysa_reference_data(ysa_config, risk_config)
    model_file = ysa_model_file_name(ysa_config)

    st.markdown("### YSA Manuel Tahmin")
    st.caption(
        "Dosya özelliklerini girerek eğitilmiş Yapay Sinir Ağı modeliyle yalnızca SLA ihlal riski tahmini alın."
    )

    if config_source == RISK_CONFIG_FILE:
        st.info(
            "YSA için ayrı config bulunamadı; manuel alanlar mevcut risk modeli feature listesine göre hazırlandı."
        )

    if not model_features:
        st.warning("YSA tahmini için kullanılacak model_features bilgisi bulunamadı.")
        return

    if model_file is None:
        st.warning(
            f"YSA model dosyası bulunamadı. Varsayılan dosya adı: {YSA_MODEL_FILE}. "
            "Dosyayı eklediğinizde bu bölümden manuel tahmin alınabilir."
        )

    with st.form("ysa_manual_prediction_form"):
        st.markdown("#### Dosya Özellikleri")
        input_values: dict[str, Any] = {}
        columns = st.columns(2)
        for index, feature in enumerate(model_features):
            with columns[index % 2]:
                key = f"ysa_input_{index}_{feature}"
                if feature in categorical_features:
                    options = unique_options(reference_df, feature)
                    if options:
                        input_values[feature] = st.selectbox(feature, options=options, key=key)
                    else:
                        input_values[feature] = st.text_input(feature, key=key)
                elif is_binary_feature(feature, reference_df):
                    default_value = bool(int(round(numeric_default(reference_df, feature))))
                    input_values[feature] = int(st.checkbox(feature, value=default_value, key=key))
                elif feature in numeric_features or feature in reference_df.columns:
                    default_value = numeric_default(reference_df, feature)
                    input_values[feature] = st.number_input(
                        feature,
                        value=float(default_value),
                        step=1.0,
                        key=key,
                    )
                else:
                    input_values[feature] = st.text_input(feature, key=key)

        submitted = st.form_submit_button("YSA SLA Riskini Tahmin Et")

    if not submitted:
        return

    if model_file is None:
        st.warning("YSA modeli yüklenemediği için tahmin üretilemedi.")
        return

    raw_input = pd.DataFrame([input_values], columns=model_features)
    try:
        artifact = load_ysa_model(model_file)
        model, bundled_preprocessor, bundled_features = unpack_ysa_bundle(artifact)
        prepared_input = prepare_ysa_input(
            raw_input,
            ysa_config,
            model_file,
            model,
            bundled_preprocessor,
            bundled_features,
        )
        score = float(np.clip(ysa_probability(model, prepared_input), 0, 1))
    except ValueError as exc:
        st.error(str(exc))
        return
    except FileNotFoundError as exc:
        st.warning(f"YSA tahmini için gerekli dosya bulunamadı: {exc}")
        return
    except Exception as exc:
        st.error(f"YSA tahmini üretilirken bir hata oluştu: {exc}")
        return

    percent = score * 100
    level = ysa_result_level(score, thresholds)
    prediction_text = "İhlal Bekleniyor" if score >= 0.5 else "İhlal Beklenmiyor"
    comment = ysa_comment(level, prediction_text)

    st.markdown("#### Tahmin Sonucu")
    result_cols = st.columns(4)
    with result_cols[0]:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">YSA SLA İhlal Riski</div>
                <div class="summary-value">%{percent:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with result_cols[1]:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">YSA Risk Seviyesi</div>
                <div>{ysa_result_badge(level)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with result_cols[2]:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">YSA Tahmini</div>
                <div class="summary-value">{prediction_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with result_cols[3]:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Kısa Yorum</div>
                <div class="summary-value" style="font-size: 0.92rem; line-height: 1.35;">{comment}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.markdown('<div class="page-title">IT Servis Yönetimi SLA Risk Takip Paneli</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Analiz edilen servis dosyaları için SLA riski, tahmini çözüm süresi ve operasyonel aksiyon takibi.</div>',
        unsafe_allow_html=True,
    )

    try:
        operations, _notices = build_operations_table()
    except FileNotFoundError as exc:
        st.error(f"Gerekli dosya bulunamadı: {exc}")
        st.stop()
    except KeyError as exc:
        st.error(f"Gerekli kolonlar eksik. {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Panel hazırlanırken bir hata oluştu: {exc}")
        st.stop()

    risk_config, _duration_config = load_configs()

    st.sidebar.markdown("## SLA Risk Paneli")
    st.sidebar.caption("AI destekli operasyonel takip ekranı")
    filtered = filter_dataframe(operations)

    total_cases = len(filtered)
    high_risk = int((filtered[RISK_LEVEL_COL] == "Yüksek").sum()) if total_cases else 0
    avg_duration = float(filtered[DURATION_COL].mean()) if total_cases else 0.0
    avg_risk = float(filtered[RISK_SCORE_COL].mean() * 100) if total_cases else 0.0
    critical_cases = int((filtered.get("Priority", pd.Series(dtype=str)).astype(str).str.lower() == "high").sum())

    card_cols = st.columns(5)
    with card_cols[0]:
        metric_card("Analiz Edilen Dosya", f"{total_cases:,}".replace(",", "."), "Filtrelenen toplam kayıt")
    with card_cols[1]:
        metric_card("Yüksek Risk", f"{high_risk:,}".replace(",", "."), "Acil takip gerektiren dosyalar")
    with card_cols[2]:
        metric_card("Ort. Tahmini Süre", f"{avg_duration:.1f} sa", "Beklenen çözüm süresi")
    with card_cols[3]:
        metric_card("Ort. Risk Oranı", f"%{avg_risk:.1f}", "Ortalama SLA ihlal olasılığı")
    with card_cols[4]:
        metric_card("Kritik Öncelik", f"{critical_cases:,}".replace(",", "."), "High öncelikli kayıtlar")

    st.markdown("### Öncelikli Takip Listesi")
    if filtered.empty:
        st.info("Seçilen filtrelere uygun dosya bulunamadı.")
        st.divider()
        render_ysa_manual_prediction(risk_config)
        return

    max_display_rows = 1000
    sorted_table = filtered.assign(
        _sla_breach=(filtered[SLA_STATUS_COL] == "Limit aşımı bekleniyor").astype(int)
    ).sort_values(
        by=[RISK_SCORE_COL, "_sla_breach", DURATION_COL],
        ascending=[False, False, False],
    ).drop(columns=["_sla_breach"])
    visible_table = sorted_table.head(max_display_rows)
    if len(sorted_table) > max_display_rows:
        st.caption(
            f"Tablo performansı için en yüksek öncelikli ilk {max_display_rows} kayıt gösteriliyor. "
            "Daha dar sonuçlar için sol filtreleri kullanabilirsiniz."
        )

    display_table = make_display_table(visible_table)
    styled_table = display_table.style.format({
        RISK_PERCENT_COL: "{:.1f}",
        DURATION_COL: "{:.1f}",
        SLA_LIMIT_COL: "{:.1f}",
    })
    if RISK_LEVEL_COL in display_table.columns:
        styled_table = style_cells(styled_table, style_risk, [RISK_LEVEL_COL])
    if SLA_STATUS_COL in display_table.columns:
        styled_table = style_cells(styled_table, style_status, [SLA_STATUS_COL])

    st.dataframe(styled_table, use_container_width=True, hide_index=True, height=430)

    st.markdown("### Dosya Detayı")
    detail_options = visible_table[CASE_ID_COL].tolist()
    detail_labels = {
        row[CASE_ID_COL]: f"{row[CASE_ID_COL]} | Risk: %{row[RISK_PERCENT_COL]:.1f} | SLA: {row[SLA_STATUS_COL]}"
        for _, row in visible_table.iterrows()
    }
    selected_case = st.selectbox(
        "Detayını görüntülemek istediğiniz dosyayı seçin",
        options=detail_options,
        format_func=lambda case_id: detail_labels.get(case_id, str(case_id)),
    )

    selected_row = sorted_table.loc[sorted_table[CASE_ID_COL] == selected_case].iloc[0]
    render_detail(selected_row)

    st.divider()
    render_ysa_manual_prediction(risk_config)


if __name__ == "__main__":
    main()
