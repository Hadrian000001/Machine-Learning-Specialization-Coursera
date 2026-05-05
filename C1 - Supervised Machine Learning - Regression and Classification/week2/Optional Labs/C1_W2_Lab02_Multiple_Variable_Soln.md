# Teaching Lab: Multiple Variable Linear Regression

This lab extends the simple linear regression (one feature) to **multiple features** (e.g., house size, bedrooms, floors, age). All the key ideas – prediction, cost, gradient descent – are updated to work with vectors and matrices.  

We will walk through the notebook step by step, and I will highlight the **`error * error`** (or `np.dot(errors, errors)`) approach for computing the cost, as you requested.

---

## 1. Notation & Data

- **m** = number of training examples (here m = 3)
- **n** = number of features (here n = 4: size, bedrooms, floors, age)
- **X** = training example matrix of shape `(m, n)`. Each row is one house.
- **y** = target prices (in $1000s), shape `(m,)`.
- **w** = weight vector, shape `(n,)`
- **b** = bias scalar.

Example data:

| Size (sqft) | Bedrooms | Floors | Age | Price |
|-------------|----------|--------|-----|-------|
| 2104        | 5        | 1      | 45  | 460   |
| 1416        | 3        | 2      | 40  | 232   |
| 852         | 2        | 1      | 35  | 178   |

In code:
```python
X_train = np.array([[2104, 5, 1, 45],
                    [1416, 3, 2, 40],
                    [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
```

---

## 2. Model Prediction

For a single example `x` (shape `(n,)`), the prediction is:
\[
f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b = \sum_{j=0}^{n-1} w_j x_j + b
\]

### Loop version (slow, but educational)
```python
def predict_single_loop(x, w, b):
    p = 0
    for i in range(len(x)):
        p += x[i] * w[i]
    return p + b
```

### Vectorized version (using `np.dot`)
```python
def predict(x, w, b):
    return np.dot(x, w) + b
```
**Why vectorized?** `np.dot` calls highly optimised C code, avoids Python loops, and is much faster – especially when you later compute predictions for all examples at once.

---

## 3. Cost Function for Multiple Variables

The mean squared error cost is:
\[
J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \big( f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)} \big)^2
\]

### Loop version (given in the lab)
```python
def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b   # prediction for i-th example
        cost += (f_wb_i - y[i]) ** 2
    return cost / (2 * m)
```

### Your requested `error * error` (vectorised) version

Here is the **fully vectorised** cost – no loops over `m`:
```python
def compute_cost_vectorized(X, y, w, b):
    m = X.shape[0]
    predictions = X @ w + b          # shape (m,)
    errors = predictions - y         # shape (m,)
    cost = (1 / (2 * m)) * (errors @ errors)   # or np.dot(errors, errors)
    return cost
```
- `errors @ errors` is the dot product of the error vector with itself, which sums the squares of all errors.
- This is exactly `np.dot(errors, errors)`.
- It is **much faster** than the loop because it uses a single BLAS operation.

**Why `error * error`?**  
Because `errors` is a vector containing all `(prediction_i - y_i)`. The dot product `errors · errors` = sum of squares. The cost function then divides by `2m`.

The lab notebook uses the loop version for clarity, but later labs and real code prefer the vectorised form.

---

## 4. Gradient Descent for Multiple Variables

We update all parameters simultaneously:

\[
w_j := w_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \, x_j^{(i)} \quad (\text{for each } j)
\]
\[
b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})
\]

### Loop version (from the lab – two nested loops)
```python
def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i, j]
        dj_db += err

    dj_dw /= m
    dj_db /= m
    return dj_db, dj_dw
```

### Vectorised gradient (no loops over m or n)
```python
def compute_gradient_vectorized(X, y, w, b):
    m = X.shape[0]
    predictions = X @ w + b
    errors = predictions - y
    dj_dw = (1/m) * (X.T @ errors)   # shape (n,)
    dj_db = (1/m) * np.sum(errors)
    return dj_db, dj_dw
```
Here `X.T @ errors` is a matrix‑vector product: each component is the dot product of one column of `X` with the `errors` vector.

---

## 5. Putting It All Together: Gradient Descent Loop

The outer iteration loop (over `num_iters`) is **still required** because gradient descent is iterative. Inside each iteration we call the vectorised gradient function.

```python
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    w = w_in.copy()
    b = b_in
    J_history = []

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        J_history.append(cost_function(X, y, w, b))   # cost_function can be vectorised too
    return w, b, J_history
```

The lab then runs gradient descent on the housing data with a tiny learning rate (`alpha = 5.0e-7`) and 1000 iterations. The cost decreases slowly – that’s why the next lab introduces **feature scaling** to speed up convergence.

---

## 6. Key Takeaways

- **Use vectors and matrices** – represent all training data in `X` (shape `m×n`), parameters in `w` (shape `n`).
- **Vectorise predictions** with `X @ w + b`.
- **Compute cost** vectorised as `errors @ errors / (2*m)` where `errors = X@w + b - y`.
- **Compute gradients** vectorised as `dj_dw = (X.T @ errors)/m`, `dj_db = np.sum(errors)/m`.
- **Only the outer iteration loop** remains; inner loops over examples or features disappear.

The `error * error` (dot product of the error vector with itself) is the core of the vectorised cost. It’s clean, fast, and exactly matches the mathematical formula.

---

## 7. What’s Next?

The lab ends with the observation that gradient descent on raw features (size 2104 vs 852) does not converge quickly – the cost decreases slowly and predictions are not great. The next lab introduces **feature scaling** (normalisation) to fix this problem.

---

**You now have a complete understanding of multiple variable linear regression with vectorised operations. Practice by rewriting the cost and gradient functions using the `errors` vector and dot products – it will solidify your knowledge.**
