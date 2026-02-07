import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

st.set_page_config(
    page_title="Workload Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main { background-color: #f8f9fa; color: #212529; }
.stButton>button {
    width: 100%; border-radius: 5px; height: 3em;
    background-color: #007bff; color: white; border: none;
}
.stButton>button:hover { background-color: #0056b3; }
h1, h2, h3 { font-family: 'Inter', sans-serif; }
.forecast-card {
    background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
}
.forecast-card .label {
    font-size: 0.9rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}
.forecast-card .value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
    line-height: 1.2;
    font-variant-numeric: tabular-nums;
}
.forecast-card .metric {
    font-size: 0.85rem;
    color: #007bff;
    margin-top: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

LOOKBACK_WINDOW = 12
MODEL_DIR = "model"
SCALER_DIR = "scaler"
INSTANCE_CATEGORIES_FILE = "instance_categories.pkl"
SAMPLE_DIR = "sample"

SERVICES = {
    "FAAS": {"name": "Function as a Service", "metric": "QPS", "metric_col": "usage",
             "model_file": "faas_lstm.keras", "scaler_file": "faas_scaler.pkl",
             "csv_note": "Metrik: **QPS** (usage).",
             "instance_info": "**instance_id** format: `instance_1`, `instance_2`, … (total **226** instance)."},
    "PAAS": {"name": "Platform as a Service", "metric": "CPU Usage", "metric_col": "usage",
             "model_file": "paas_lstm.keras", "scaler_file": "paas_scaler.pkl",
             "csv_note": "Metrik: **CPU Usage** (usage).",
             "instance_info": "**instance_id** format: `instance_1` … `instance_93` (total **93** instance)."},
    "IAAS": {"name": "Infrastructure as a Service", "metric": "CPU Usage", "metric_col": "usage",
             "model_file": "iaas_lstm.keras", "scaler_file": "iaas_scaler.pkl",
             "csv_note": "Metrik: **CPU Usage** (usage).",
             "instance_info": "**instance_id** format: `instance_1`, … (total **426** instance)."},
    "RDS": {"name": "Relational Database Service", "metric": "QPS", "metric_col": "usage",
            "model_file": "rds_lstm.keras", "scaler_file": "rds_scaler.pkl",
            "csv_note": "Metrik: **QPS** (usage).",
            "instance_info": "**instance_id** format: `instance_1` … `instance_1000`, … (total **1113** instance)."},
}


def _load_instance_categories(service_key):
    cat_key = f"{service_key.lower()}_cat"
    for root in [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]:
        pkl_path = os.path.join(root, INSTANCE_CATEGORIES_FILE)
        if os.path.isfile(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    return pickle.load(f).get(cat_key)
            except Exception:
                pass
    fallback = os.path.join(SCALER_DIR, f"{service_key.lower()}_instance_categories.pkl")
    if os.path.isfile(fallback):
        try:
            return joblib.load(fallback)
        except Exception:
            pass
    return None


@st.cache_resource
def load_resources(service_key):
    config = SERVICES[service_key]
    try:
        model = tf.keras.models.load_model(os.path.join(MODEL_DIR, config["model_file"]))
    except Exception as e:
        st.error(f"Error loading model for {service_key}: {e}")
        return None, None, None
    try:
        scaler = joblib.load(os.path.join(SCALER_DIR, config["scaler_file"]))
    except Exception as e:
        st.error(f"Error loading scaler for {service_key}: {e}")
        return None, None, None
    return model, scaler, _load_instance_categories(service_key)


def _instance_code(instance_id, instance_categories):
    if instance_categories is None:
        return 0
    try:
        if hasattr(instance_categories, "categories"):
            return int(pd.Series([instance_id]).astype(instance_categories).cat.codes.iloc[0])
        return instance_categories.index(instance_id)
    except (ValueError, AttributeError, TypeError):
        return 0


def preprocess_input(data, scaler, n_features=None, instance_ids=None, instance_categories=None, instance_codes_direct=None):
    data = np.array(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_scale = getattr(scaler, "n_features_in_", 1)
    data_for_scale = data[:, :n_scale]
    scaled_usage = scaler.transform(data_for_scale)
    if len(scaled_usage) < LOOKBACK_WINDOW:
        return None, f"Insufficient data points. Need at least {LOOKBACK_WINDOW}, got {len(scaled_usage)}."

    seq_usage = scaled_usage[-LOOKBACK_WINDOW:].reshape(-1, 1)

    if n_features == 2:
        if instance_codes_direct is not None and len(instance_codes_direct) >= LOOKBACK_WINDOW:
            inst_codes = np.array(instance_codes_direct[-LOOKBACK_WINDOW:], dtype=np.float64).reshape(-1, 1)
        elif instance_ids is not None and instance_categories is not None and len(instance_ids) >= LOOKBACK_WINDOW:
            inst_codes = np.array(
                [_instance_code(inst, instance_categories) for inst in instance_ids[-LOOKBACK_WINDOW:]],
                dtype=np.float64
            ).reshape(-1, 1)
        else:
            inst_codes = None
        if inst_codes is not None:
            input_sequence = np.hstack([seq_usage, inst_codes])
        else:
            input_sequence = np.hstack([seq_usage] * n_features) if n_features else seq_usage
    else:
        input_sequence = np.hstack([seq_usage] * n_features) if n_features and seq_usage.shape[1] < n_features else seq_usage[:, :n_features] if n_features else seq_usage

    return input_sequence.reshape(1, LOOKBACK_WINDOW, -1).astype(np.float64), None


def inverse_transform_prediction(scaler, pred_scaled):
    n_scale = getattr(scaler, "n_features_in_", 1)
    return float(scaler.inverse_transform(pred_scaled[:, :n_scale])[0][0])


def _clip_to_window(final_value, data_points):
    if len(data_points) < LOOKBACK_WINDOW:
        return final_value, False
    last = data_points[-LOOKBACK_WINDOW:]
    low, high = max(0.0, float(np.min(last))), float(np.max(last))
    if final_value < low or final_value > high:
        return float(np.clip(final_value, low, high)), True
    return final_value, False


def _get_instance_col(df, use_instance):
    if not use_instance:
        return None
    if "instance_code" in df.columns and np.issubdtype(df["instance_code"].dtype, np.number):
        return "instance_code"
    if "instance_id" in df.columns:
        return "instance_id"
    if "instance" in df.columns:
        return "instance"
    return None


def _load_dataframe(selected_service, input_source):
    if input_source == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV (Time Series)", type=["csv"])
        if uploaded is None:
            return None
        try:
            return pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Error membaca file: {e}")
            return None

    sample_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), SAMPLE_DIR)
    if not os.path.isdir(sample_root):
        sample_root = SAMPLE_DIR
    prefix = f"sample_{selected_service.lower()}_"
    available = sorted(f for f in (os.listdir(sample_root) if os.path.isdir(sample_root) else [])
                       if f.startswith(prefix) and f.endswith(".csv"))
    if not available:
        st.warning(f"Tidak ada file sample di `{sample_root}` (contoh: {prefix}minimal.csv).")
        return None
    chosen_idx = st.selectbox("Pilih sample", range(len(available)), format_func=lambda i: available[i], key="sample_sel")
    try:
        df = pd.read_csv(os.path.join(sample_root, available[chosen_idx]))
        st.caption(f"Loaded: `{available[chosen_idx]}`")
        return df
    except Exception as e:
        st.error(f"Error membaca sample: {e}")
        return None


def main():
    st.sidebar.title("Bytedance-TS-Forecast")
    selected_service = st.sidebar.selectbox("Select Service", list(SERVICES.keys()))
    config = SERVICES[selected_service]

    st.title(f"{selected_service} Forecasting")
    st.markdown(f"**Metric:** {config['metric']} | **Service:** {config['name']}")

    model, scaler, instance_categories = load_resources(selected_service)
    if model is None or scaler is None:
        st.warning("Could not load model or scaler. Check file paths.")
        return

    n_features = int(model.input_shape[-1])
    use_instance = n_features == 2 and instance_categories is not None

    st.subheader("Input Data")
    input_source = st.radio("Sumber data", ["Upload CSV", "Gunakan sample dari folder sample/"], horizontal=True, key="input_src")
    df = _load_dataframe(selected_service, input_source)

    if df is None:
        st.info("Upload CSV atau pilih sample untuk forecast.")
        with st.expander("Kriteria CSV"):
            cfg = config
            st.markdown(f"**{selected_service}** — {cfg['name']} (metrik: {cfg['metric']})")
            st.markdown(f"""
            - **Kolom wajib:** **`{cfg['metric_col']}`** (numerik). Minimal **12 baris**.
            - **Opsional:** **`instance_code`** (integer) atau **`instance_id`**. {cfg['csv_note']} {cfg['instance_info']}
            """)
            st.table(pd.DataFrame({
                "timestamp": ["2024-01-01 00:00", "…"],
                cfg["metric_col"]: [42.3, "…"],
                "instance_code": [0, "…"],
            }))
        return

    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("CSV harus punya minimal satu kolom numerik (e.g. usage).")
            return
        usage_col = "usage" if "usage" in df.columns else numeric_cols[0]
        data_points = df[usage_col].values
        instance_col = _get_instance_col(df, use_instance)
        instance_codes_for_seq = None
        instance_ids_for_seq = None

        st.write(f"Loaded {len(data_points)} points dari kolom '{usage_col}'" + (f", instance: '{instance_col}'" if instance_col else ""))

        if instance_col:
            if instance_col == "instance_code":
                if len(df) < LOOKBACK_WINDOW:
                    st.warning(f"Perlu minimal {LOOKBACK_WINDOW} baris.")
                else:
                    data_points = df[usage_col].values
                    instance_codes_for_seq = np.asarray(df[instance_col].values, dtype=np.float64)
            else:
                instances = df[instance_col].dropna().unique().tolist()
                chosen = st.selectbox("Forecast for instance", instances, key="inst_sel") if len(instances) > 1 else instances[0]
                df_inst = df[df[instance_col] == chosen].sort_index()
                if len(df_inst) >= LOOKBACK_WINDOW:
                    data_points = df_inst[usage_col].values
                    instance_ids_for_seq = df_inst[instance_col].values
                else:
                    st.warning(f"Instance ini punya {len(df_inst)} baris; butuh minimal {LOOKBACK_WINDOW}.")
        elif use_instance:
            st.info("Tambahkan kolom **instance_code** atau **instance_id** + file kategori (lihat README).")

        chart_series = df.groupby(instance_col)[usage_col].mean() if instance_col and instance_col != "instance_code" and df[instance_col].nunique() > 1 else df[usage_col]
        st.line_chart(chart_series)

        if not st.button("Generate Forecast", type="primary"):
            return

        input_seq, error = preprocess_input(
            data_points, scaler, n_features,
            instance_ids=instance_ids_for_seq,
            instance_categories=instance_categories,
            instance_codes_direct=instance_codes_for_seq,
        )
        if error:
            st.error(error)
            return

        with st.spinner("Calculating forecast..."):
            pred_scaled = model.predict(input_seq)
            final_value = inverse_transform_prediction(scaler, pred_scaled)
            final_value, clipped = _clip_to_window(final_value, data_points)
            used_instance = (
                instance_codes_for_seq is not None
                or (instance_ids_for_seq is not None and instance_categories is not None)
            )

        st.divider()
        st.subheader("Forecast Result")
        if clipped:
            st.caption("Prediksi dibatasi ke rentang 12 nilai terakhir (min–max).")
        if final_value < 0 and not used_instance:
            st.warning(
                "Prediksi negatif: gunakan **instance_code** / **instance_id** dan file "
                f"**{selected_service.lower()}_instance_categories.pkl** (lihat README)."
            )
        display_value = f"{final_value:,.2f}"
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="forecast-card">
                <div class="label">Prediksi nilai berikutnya</div>
                <div class="value">{display_value}</div>
                <div class="metric">{config['metric']}</div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
