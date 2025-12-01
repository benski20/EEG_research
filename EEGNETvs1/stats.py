import statsmodels.stats.proportion as smp

# Example: 2375 correct out of 2500 trials
k, n = 1850, 2500
ci_low, ci_high = smp.proportion_confint(k, n, alpha=0.1, method='wilson')
print(f"95% CI: {ci_low*100:.2f} – {ci_high*100:.2f}")

ci = ci_high - ci_low
print(f"CI width: {(ci*100)/2:.2f}")