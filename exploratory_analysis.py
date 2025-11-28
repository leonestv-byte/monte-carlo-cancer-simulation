from sklearn.datasets import load_breast_cancer
import pandas as pd
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

# First, let's load our data
data = load_breast_cancer(as_frame=True)
df = data.frame

# Now, let's get our binary data
y = df.target

successes = (y == 1).sum()
failures = (y == 0).sum()

print(successes) # 357
print(failures) # 212

# We are predicting successes and failures, so, a beta distribution works.

# We define Theta as the proportion of cancer survivors, Theta ~ Beta(357, 212)


# Let us define our beta distribution parameters, based on approximations from above.
alpha_prior = 3
beta_prior = 2

alpha_posterior = alpha_prior + successes
beta_posterior = beta_prior + failures


x = np.linspace(0, 1, 1000)

prior_pdf = beta.pdf(x, alpha_prior, beta_prior)
posterior_pdf = beta.pdf(x, alpha_posterior, beta_posterior)

plt.figure(figsize=(10, 6))

# Plot the prior distribution
plt.plot(x, prior_pdf, label=f'Prior: Beta({alpha_prior}, {beta_prior})', color='blue', linewidth=2)

# Plot the posterior distribution
plt.plot(x, posterior_pdf, label=f'Posterior: Beta({alpha_posterior}, {beta_posterior})', color='red', linewidth=2)

plt.xlabel('Probability of Having Cancer (Success) (θ)')
plt.ylabel('Density')
plt.show()

theta = alpha_posterior / (alpha_posterior + beta_posterior)

print(theta)
# Therefore, the estimate for theta is 0.627177700348432.

# This is the estimate for our dataset.

# Now, let us take our posterior beta distribution, and conduct sampling.

# Monte Carlo Sampling is taking estimates of the theta parameter from
# the posterior distribution. We can then get our estimates on it.

num_samples = 100000  # Large number of simulations
monte_carlo_samples = np.random.beta(alpha_posterior, beta_posterior, num_samples)
monte_carlo_theta_mean = np.mean(monte_carlo_samples)
print(monte_carlo_theta_mean)
print(abs(monte_carlo_theta_mean - theta))
# Therefore, Monte Carlo Sampling delivers us a mean theta that is almost identical to our mean
# taken simply from our alpha and beta calculated from the posterior - it is only 7.610010607561613e-05.
