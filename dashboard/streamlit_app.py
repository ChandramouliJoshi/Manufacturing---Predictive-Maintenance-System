import streamlit as st
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
	import shap
	SHAP_AVAILABLE = True
except Exception:
	SHAP_AVAILABLE = False


st.set_page_config(page_title="FactoryGuard AI — Predictive Maintenance", layout="wide")

st.title("FactoryGuard AI — IoT Predictive Maintenance Engine")
st.markdown(
	"""
	FactoryGuard AI monitors robotic arm vibration, temperature, and pressure to predict critical failures 24 hours before they occur.
	Use this dashboard to review sample inputs, compute SHAP explanations, and validate model behavior for the factory floor.
	"""
)


MODEL_PATH = Path("models/production_failure_model.joblib")
SHAP_ARTIFACT = Path("models/shap_background.joblib")
SHAP_VALUES_ARTIFACT = Path("models/shap_values.joblib")
SHAP_SAMPLE_CSV = Path("models/shap_sample.csv")


@st.cache_resource
def load_model(path: Path):
	if not path.exists():
		return None, f"Missing model: {path}"
	try:
		return joblib.load(path), None
	except Exception as exc:
		return None, f"Model load failed: {exc}"


@st.cache_data
def load_shap_artifacts(path: Path):
	if not path.exists():
		return None, f"Missing SHAP artifacts: {path}"
	try:
		return joblib.load(path), None
	except Exception as exc:
		return None, f"SHAP artifact load failed: {exc}"


@st.cache_data
def load_shap_values(path: Path):
	if not path.exists():
		return None, None
	try:
		return joblib.load(path), None
	except Exception as exc:
		return None, f"Saved SHAP values load failed: {exc}"


def save_shap_values_artifact(shap_values, output_path: Path):
	output_path.parent.mkdir(parents=True, exist_ok=True)
	artifact = {
		"shap_explanation": shap_values,
		"values": shap_values.values,
		"base_values": shap_values.base_values,
		"data": shap_values.data,
		"feature_names": list(shap_values.feature_names),
	}
	joblib.dump(artifact, output_path)
	return output_path


model, model_error = load_model(MODEL_PATH)
artifacts, artifact_error = load_shap_artifacts(SHAP_ARTIFACT)

if model is None:
	st.error(f"{model_error} — run training first.")
	st.stop()

if artifacts is None:
	st.warning(f"{artifact_error}. Falling back to models/shap_sample.csv.")

background = artifacts.get("background") if artifacts else None
sample_df = artifacts.get("sample") if artifacts else None
feature_cols = artifacts.get("feature_columns") if artifacts else None

if sample_df is None and SHAP_SAMPLE_CSV.exists():
	sample_df = pd.read_csv(SHAP_SAMPLE_CSV)

if feature_cols is None and hasattr(model.named_steps.get("imputer"), "feature_names_in_"):
	feature_cols = list(model.named_steps["imputer"].feature_names_in_)

if background is None and sample_df is not None and feature_cols is not None:
	background = sample_df[feature_cols].head(min(50, len(sample_df)))

if background is None or sample_df is None or feature_cols is None:
	st.error("Dashboard inputs are incomplete. Regenerate SHAP artifacts with `python src/explainability.py`.")
	st.stop()

saved_shap_artifact, saved_shap_error = load_shap_values(SHAP_VALUES_ARTIFACT)
if saved_shap_error:
	st.warning(saved_shap_error)
saved_shap_values = None
if saved_shap_artifact is not None:
	saved_shap_values = saved_shap_artifact.get("shap_explanation")

st.sidebar.header("Options")
max_rows = max(1, min(20, len(sample_df)))
sample_count = st.sidebar.slider("Show sample rows", 1, max_rows, min(5, max_rows))
selected_index = st.sidebar.number_input(
	"Select sample index",
	min_value=0,
	max_value=max(0, len(sample_df) - 1),
	value=0,
)
show_saved_shap = st.sidebar.checkbox("Load saved SHAP values", value=saved_shap_values is not None)

st.header("Sample Input")
st.dataframe(sample_df.head(sample_count))

if "shap_values" not in st.session_state:
	st.session_state.shap_values = None

if show_saved_shap:
	if saved_shap_values is None:
		st.warning("No precomputed SHAP values found. Use the CLI to generate `models/shap_values.joblib` or disable this option.")	
	else:
		st.subheader("Precomputed SHAP — selected sample")
		try:
			fig1 = plt.figure(figsize=(8, 5))
			shap.plots.waterfall(saved_shap_values[selected_index], show=False)
			st.pyplot(fig1)

			st.subheader("Precomputed SHAP summary")
			fig2 = plt.figure(figsize=(8, 6))
			shap.plots.bar(saved_shap_values, max_display=20)
			st.pyplot(fig2)

			st.subheader("Saved feature impact table")
			feature_impacts = pd.DataFrame(
				{
					"feature": feature_cols,
					"mean_abs_shap": np.abs(saved_shap_values.values).mean(axis=0),
				}
			)
			st.dataframe(feature_impacts.sort_values(by="mean_abs_shap", ascending=False).head(20))
		except Exception as exc:
			st.error(f"Saved SHAP display failed: {exc}")

if st.button("Compute SHAP for selected sample"):
	if not SHAP_AVAILABLE:
		st.error("SHAP package is not available. Install shap in your environment.")
	else:
		with st.spinner("Computing SHAP values (may take a few seconds)..."):
			try:
				predict_fn = lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_cols))[:, 1]
				explainer = shap.Explainer(predict_fn, background, feature_names=feature_cols)
				shap_values = explainer(sample_df)

				st.subheader("Waterfall — selected sample")
				fig1 = plt.figure(figsize=(8, 5))
				shap.plots.waterfall(shap_values[selected_index], show=False)
				st.pyplot(fig1)

				st.subheader("Summary (bar) — sample set")
				fig2 = plt.figure(figsize=(8, 6))
				shap.plots.bar(shap_values, max_display=20)
				st.pyplot(fig2)

				st.subheader("Feature impact table")
				feature_impacts = pd.DataFrame(
					{
						"feature": feature_cols,
						"mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
					}
				)
				st.dataframe(feature_impacts.sort_values(by="mean_abs_shap", ascending=False).head(20))
				st.session_state.shap_values = shap_values

			except Exception as exc:
				st.error(f"SHAP computation failed: {exc}")
				st.session_state.shap_values = None

if st.button("Save computed SHAP values"):
	if st.session_state.shap_values is None:
		st.warning("Compute SHAP values first, then save them.")
	else:
		saved_path = save_shap_values_artifact(st.session_state.shap_values, SHAP_VALUES_ARTIFACT)
		st.success(f"Saved computed SHAP values to {saved_path}")

# quick model sanity check
try:
	preview = model.predict_proba(sample_df.head(3))[:, 1]
	st.sidebar.write("Model sample probabilities:", list(preview))
except Exception:
	st.sidebar.write("Model predict_proba preview failed (pipeline may require training columns).")

st.markdown("---")
st.caption("Run: `streamlit run dashboard/streamlit_app.py`")
