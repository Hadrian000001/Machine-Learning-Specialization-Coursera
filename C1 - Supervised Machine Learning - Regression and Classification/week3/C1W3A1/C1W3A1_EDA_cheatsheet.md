Ultimate EDA Cheat Sheet** – a quick-reference guide for any regression or classification project. 

---

### 📋 General EDA Cheat Sheet

| Phase | Step | Action / Code Snippet (Short) | Why? (One Sentence) | Benchmark / Threshold |
| :--- | :--- | :--- | :--- | :--- |
| **1. Setup** | Load & Glimpse | `df.head()`, `df.info()`, `df.shape` | Get the lay of the land: know your size, column names, and dtypes upfront. | Ensure rows > columns. Dtypes must match reality (numeric vs object). |
| **2. Data Quality** | Missing Values | `df.isnull().sum()` | Identify gaps that will break models or bias predictions. | **>5% missing** per column → needs imputation/dropping. |
| **3. Data Quality** | Duplicates | `df.duplicated().sum()` | Duplicate rows artificially inflate training influence and leak info. | **>0** → remove them immediately. |
| **4. Target (y)** | Distribution Plot | `sns.histplot(df['y'], kde=True)` | Check for extreme skewness (affects regression residual normality & class balance). | **Skew > 0.5** (regression) → log-transform. **Classification** → ensure minority class > **20%**. |
| **5. Target (y)** | QQ Plot (Regression only) | `stats.probplot(df['y'], plot=plt)` | Visually assess if target follows a normal curve (linear regression assumption). | Points should closely follow the red diagonal line. |
| **6. Features (X)** | Numeric Summary | `df.describe()` | Spot scale mismatches (e.g., 1000s vs decimals) for gradient descent. | If max/min ratio > **100**, **StandardScaler** is mandatory. |
| **7. Features (X)** | Categorical Counts | `df['cat_col'].value_counts()` | Find rare categories that might cause overfitting. | Categories with < **5%** frequency should be grouped into "Other". |
| **8. Bivariate** | Feature vs Target (Numeric) | `sns.scatterplot(x='X', y='y', data=df)` | Check the **linearity assumption** for regression / separability for classification. | Regression: must see a straight-ish cloud. Classification: visible distinct clusters. |
| **9. Bivariate** | Feature vs Target (Categorical) | `sns.boxplot(x='cat', y='y')` (Reg) / `pd.crosstab()` (Clf) | Check if categories actually differentiate the target. | Boxplots must have different medians. Crosstab must show clear percentage differences. |
| **10. Feature-Feature** | Correlation Matrix | `df.corr()` | Detect **multicollinearity** (features giving the same information). | **Pearson r > 0.8** or **VIF > 5/10** → drop one or use Ridge/Lasso. |
| **11. Outliers** | Z-Score / IQR | `np.abs(stats.zscore(df['X'])) > 3` | Extreme values heavily skew coefficients (regression) and decision boundaries (logistic). | **Z > 3** or **IQR * 1.5** → investigate; cap (winsorize) or log-transform to mitigate. |
| **12. Outliers** | High Leverage (Regression) | Cook’s Distance *(after initial fit)* | Find specific rows that disproportionately pull the regression line. | **Cook's D > 1** or > `4/(n-k-1)` → remove or inspect deeply. |
| **13. Time/Index** | Trend / Seasonality | `df.set_index('date')['y'].plot()` (if time exists) | Ensure your model isn't predicting past trends that don't generalize. | Non-stationary data? Difference or use time-series specific models. |
| **14. Distributions** | Skewness & Kurtosis | `df['X'].skew()`, `df['X'].kurtosis()` | Heavy tails break the "constant variance" assumption (homoscedasticity). | Skew > **0.5** → apply `np.log1p()` or `np.sqrt()` to features. |
| **15. Final** | Summarize Insights | Print a bulleted text block | Force yourself to articulate findings before modeling; prevents blind spots. | Write down: "Top 3 predictors", "Biggest data quality issue", "Scaling needed". |

---

### 🔥 Pro-Tip Functions (imports to remember)

```python
# Imports
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# One-liner for VIF calculation
vif_data = pd.DataFrame({'Feature': X.columns, 'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})

# One-liner for Z-score outliers
outliers = (np.abs(stats.zscore(df[['col1','col2']])) > 3).any(axis=1)
```

Keep this beside you, and you will **never** miss a critical assumption before training! 🚀
