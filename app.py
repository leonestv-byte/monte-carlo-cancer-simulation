import streamlit as st
import numpy as np
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="Cancer Patients Simulation", layout="centered")

st.title("Cancer Patients Simulation (Beta Monte-Carlo)")

st.write("### Shape parameters for Beta Distribution")

# Sliders
alpha_prior = st.slider("Alpha Prior", min_value=1, max_value=100, value=10)
beta_prior = st.slider("Beta Prior", min_value=1, max_value=100, value=10)

# Load data
data = load_breast_cancer(as_frame=True)
df = data.frame
y = df.target

successes = (y == 1).sum()
failures = (y == 0).sum()

# Posterior
alpha_post = alpha_prior + successes
beta_post = beta_prior + failures

# Monte Carlo sampling
samples = np.random.beta(alpha_post, beta_post, 1000)
mean_est = samples.mean()

st.write(f"### Monte Carlo (1000 samples) mean: **{mean_est:.5f}**")
st.write(f"### Total Cancer Patients (Dataset): **{successes}**")
st.write(f"### Estimated Cancer Patients (Applying θ): **{mean_est * (successes + failures):.5f}**")

