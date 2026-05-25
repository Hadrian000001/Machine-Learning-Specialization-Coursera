Here's your **Week 2 Summary** fully converted to GitHub‑friendly Markdown with proper LaTeX math delimiters (`$$` for display, `$` for inline). Copy and save this as `week2_linear_regression.md`.

# 📚 Week 2 Summary: Linear Regression, Gradient Descent, Feature Engineering, and Scikit‑Learn

This note consolidates everything you learned in Week 2 of the course. It includes **theory**, **formulas**, **code patterns**, and **key tips** for implementing linear regression from scratch and using scikit‑learn.

---

## 1. Linear Regression Model

### Theory
Predict a target value $y$ from features $x_1, x_2, \dots, x_n$ using a linear combination:

$$
f_{\mathbf{w},b}(\mathbf{x}) = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b = \mathbf{w} \cdot \mathbf{x} + b
$$

- $\mathbf{w}$ = weight vector (size $n$)
- $b$ = bias (intercept)

### Single feature (univariate)
$$
f_{w,b}(x) = w x + b
$$

### Multiple features (multivariate)
$$
f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b
$$

---

## 2. Cost Function (Mean Squared Error)

Measures how well the model fits the training data.

$$
J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \big( f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)} \big)^2
$$

- $m$ = number of training examples
- Division by $2$ simplifies gradient calculation.

### Vectorized implementation (NumPy)
```python
def compute_cost(X, y, w, b):
    m = X.shape[0]
    predictions = X @ w + b          # (m,)
    errors = predictions - y
    cost = (1 / (2 * m)) * (errors @ errors)   # dot product = sum of squares
    return cost

**Note:** `X` must be 2‑D `(m, n)`. For single feature, shape `(m, 1)`.  
`w` must be 1‑D `(n,)`. For single feature, `w` can be a scalar or `[w]`.
```
---

## 3. Gradient Descent

Iteratively update parameters to minimise cost.

### Update rules (simultaneous)
$$
w_j := w_j - \alpha \frac{\partial J}{\partial w_j}, \quad
b := b - \alpha \frac{\partial J}{\partial b}
$$

### Gradients (derivatives)
$$
\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \big( f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)} \big) x_j^{(i)}
$$
$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \big( f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)} \big)
$$

### Vectorized gradient computation
```python
def compute_gradient(X, y, w, b):
    m = X.shape[0]
    predictions = X @ w + b
    errors = predictions - y
    dj_dw = (1 / m) * (X.T @ errors)   # shape (n,)
    dj_db = (1 / m) * np.sum(errors)
    return dj_dw, dj_db
```

### Gradient descent loop
```python
def gradient_descent(X, y, w_init, b_init, alpha, num_iters):
    w = w_init.copy()
    b = b_init
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        # optionally track cost
    return w, b
```

**Important:** Always update `w` and `b` **simultaneously** using the old values.

---

## 4. Feature Scaling

Why needed? Features with very different scales cause gradient descent to converge slowly or oscillate.

### Z‑score normalization (standardisation)
$$
x_j^{(i)} := \frac{x_j^{(i)} - \mu_j}{\sigma_j}
$$
where $\mu_j$ = mean of feature $j$, $\sigma_j$ = standard deviation.

### Implementation
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
```

### Effect
- After scaling, each feature has mean 0 and std 1.
- Gradient descent becomes much faster and can use a larger learning rate.

**Key rule:** Always scale features **before** gradient descent (except when using the normal equation).

---

## 5. Feature Engineering & Polynomial Regression

When data is non‑linear, create new features by combining or transforming original ones.

### Polynomial features
```python
X_poly = np.c_[x, x**2, x**3]
```
Then use linear regression on these new features.

### Important notes
- Adding high‑degree polynomials can cause overfitting.
- **Always scale** polynomial features because their ranges differ drastically (e.g., $x$ vs $x^3$).
- Gradient descent will automatically give higher weights to more relevant features.

---

## 6. Normal Equation (Closed‑form solution)

Direct solution for linear regression (no iterations, no learning rate).

$$
\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}
$$

In scikit‑learn:
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
w = model.coef_
b = model.intercept_
```

**Pros:** Fast for small datasets, no scaling needed.  
**Cons:** Computationally expensive for large number of features ($O(n^3)$), only works for linear regression.

---

## 7. Using Scikit‑learn for Gradient Descent

`SGDRegressor` (Stochastic Gradient Descent) performs linear regression with gradient descent.

```python
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Train model
sgdr = SGDRegressor(max_iter=1000, alpha=0.01)   # alpha = learning rate
sgdr.fit(X_norm, y)

w = sgdr.coef_
b = sgdr.intercept_
```

**Note:** `SGDRegressor` uses a different cost function (ε‑insensitive) by default; set `loss='squared_error'` for standard MSE.

---

## 8. Common Pitfalls & Solutions

| Problem | Likely Cause | Fix |
|---------|--------------|-----|
| `ValueError: matmul: ...` | `X` is 1D or `w` has wrong shape | Reshape `X` to `(m,1)` or `(m,n)`; make `w` 1D |
| Cost increases instead of decreasing | Learning rate too high | Reduce `alpha` |
| Cost decreases very slowly | Features not scaled or `alpha` too small | Scale features, increase `alpha` |
| `setting an array element with a sequence` | `w` is an array used in scalar multiplication | Use `w[0]` or `float(w)`, or vectorised form `w * X` |
| Overfitting with polynomials | Degree too high | Reduce degree, add regularisation (later) |

---

## 9. Vectorisation Cheat Sheet

| Operation | Scalar loop | Vectorised (NumPy) |
|-----------|-------------|--------------------|
| Predictions | `for i: pred = w*x[i]+b` | `pred = X @ w + b` |
| Cost | `sum((pred-y)**2)` | `errors @ errors / (2*m)` |
| Gradients | `for i: dj_dw += error*x[i]` | `dj_dw = (X.T @ errors)/m` |
| Update | `w = w - alpha*dj_dw` (loop over features) | `w = w - alpha*dj_dw` (vector) |

**Rule:** If you can express an operation as a matrix/vector operation, do it – it’s faster and cleaner.

---

## 10. Quick Reference – Key Functions

```python
# Cost (vectorised)
def cost(X, y, w, b):
    m = X.shape[0]
    err = X @ w + b - y
    return (1/(2*m)) * (err @ err)

# Gradient (vectorised)
def grad(X, y, w, b):
    m = X.shape[0]
    err = X @ w + b - y
    dj_dw = (1/m) * (X.T @ err)
    dj_db = (1/m) * np.sum(err)
    return dj_dw, dj_db
```

```python
# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Closed‑form linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X, y)

# SGD regression
from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor(max_iter=1000, alpha=0.01).fit(X_scaled, y)
```

---

## 11. Final Tips for Week 2

1. **Always visualise your data first** (scatter plot for 1D, pair plots for many features).
2. **Normalise features** before using gradient descent – it’s not optional.
3. **Vectorise** – avoid `for` loops over training examples.
4. **Monitor cost** during gradient descent to detect convergence or divergence.
5. **Start with a small learning rate** and increase if cost decreases too slowly.
6. **Feature engineering** can turn a linear model into a powerful non‑linear model.
7. **Scikit‑learn** is your friend – use it for real projects, but understand the theory behind it.

---

**Congratulations on completing Week 2!** You now have a solid foundation in linear regression, optimisation, and feature engineering. Keep practising and experimenting with different datasets.
```

Just copy this into a file named `week2_linear_regression.md` and push it to GitHub. The equations will render perfectly.
