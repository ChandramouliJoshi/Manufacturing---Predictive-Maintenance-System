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


st.set_page_config(page_title="SHAP Explainability", layout="wide")

st.title("Model Explainability — SHAP")

MODEL_PATH = Path("models/production_failure_model.joblib")
SHAP_ARTIFACT = Path("models/shap_background.joblib")
SHAP_VALUES_ARTIFACT = Path("models/shap_values.joblib")


@st.cache_resource
def load_model(path: Path):
	if not path.exists():
		return None
	return joblib.load(path)


@st.cache_data
def load_shap_artifacts(path: Path):
	if not path.exists():
		return None
	return joblib.load(path)


@st.cache_data
def load_shap_values(path: Path):
	if not path.exists():
		return None
	return joblib.load(path)


model = load_model(MODEL_PATH)
artifacts = load_shap_artifacts(SHAP_ARTIFACT)

if model is None:
	st.error("Missing model: models/production_failure_model.joblib — run training first.")
	st.stop()

if artifacts is None:
	st.error("Missing SHAP artifacts: models/shap_background.joblib — run src/explainability.py")
	st.stop()

background = artifacts.get("background")
sample_df = artifacts.get("sample")
feature_cols = artifacts.get("feature_columns")

if background is None or sample_df is None or feature_cols is None:
	st.error("SHAP artifacts are incomplete. Expected keys: background, sample, feature_columns.")
	st.stop()

saved_shap_artifact = load_shap_values(SHAP_VALUES_ARTIFACT)
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

if show_saved_shap and saved_shap_values is not None:
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

			except Exception as exc:
				st.error(f"SHAP computation failed: {exc}")

# quick model sanity check
try:
	preview = model.predict_proba(sample_df.head(3))[:, 1]
	st.sidebar.write("Model sample probabilities:", list(preview))
except Exception:
	st.sidebar.write("Model predict_proba preview failed (pipeline may require training columns).")

st.markdown("---")
st.caption("Run: `streamlit run dashboard/streamlit_app.py`")

