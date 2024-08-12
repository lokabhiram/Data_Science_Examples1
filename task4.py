import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# Create sample data for A/B testing 
np.random.seed(42)
group_a = np.random.normal(loc=100, scale=10, size=1000)  # Control group
group_b = np.random.normal(loc=105, scale=10, size=1000)  # Treatment group

# Create a DataFrame for easier analysis
df = pd.DataFrame({'group': ['A'] * len(group_a) + ['B'] * len(group_b),
                   'value': np.concatenate([group_a, group_b])})

# Perform t-test
t_statistic, p_value = ttest_ind(group_a, group_b)

# Print results
print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

# Interpret results and draw conclusions
alpha = 0.05  # Significance level
if p_value < alpha:
    print("Reject the null hypothesis. There is a statistically significant difference between the two groups.")
    # Further analysis to determine which group performed better
    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)
    if mean_b > mean_a:
        print("Group B (treatment) performed better than Group A (control).")
    else:
        print("Group A (control) performed better than Group B (treatment).")
else:
    print("Fail to reject the null hypothesis. There is no statistically significant difference between the two groups.")