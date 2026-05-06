# Lesson: Feature Scaling and Learning Rate in Multivariate Linear Regression

This lab builds on your previous work with multiple features and focuses on two critical practical aspects of gradient descent:

1. **Choosing the learning rate α** – too small converges slowly, too large may diverge.
2. **Feature scaling (z‑score normalization)** – ensures all features contribute equally and speeds up convergence dramatically.

We’ll walk through the key ideas from the notebook.

---

## 1. Problem and Data

We continue with the housing price prediction. The training set now contains many examples, each with **4 features**:

- Size (sqft) – note: *not* divided by 1000 this time, so values range from ~300 to ~3000.
- Number of bedrooms (2–5)
- Number of floors (1–3)
- Age of home (1–100+ years)

The target is **price in $1000s**.

The data have **very different scales**: size is thousands, bedrooms are small integers, age is tens. This causes trouble for gradient descent.

---

## 2. Effect of Learning Rate α

You already know the gradient descent update:

\[
w_j := w_j - \alpha \frac{\partial J}{\partial w_j}, \quad b := b - \alpha \frac{\partial J}{\partial b}
\]

The lab tries three values of α on the **unscaled** data to show typical behaviours.

### α = 9.9e‑7 (too high)
- Cost **increases** with iterations.
- Parameters overshoot the minimum and diverge.
- This is visible in the plot of w₀ vs. iteration: it oscillates wildly.

### α = 9e‑7 (acceptable but slow)
- Cost decreases, but very slowly.
- Some parameters (like w₀ for size) still oscillate a little around the minimum because their gradient is much larger than that of other features.
- Convergence takes many iterations (thousands or more).

### α = 1e‑7 (too small)
- Cost decreases steadily, but **very** slowly.
- The parameters move toward the optimum without overshoot, but you would need many thousands of iterations to get close to the true minimum.

**Key lesson:** The right α depends on the scale of your features. Without scaling, a single α that works for one feature may be too large for another.

---

## 3. Why Uneven Scales Hurt Gradient Descent

Look at the gradients:

\[
\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^m (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \, x_j^{(i)}
\]

The gradient for a weight \(w_j\) is proportional to the **feature values** \(x_j^{(i)}\).  
If one feature (like size) is much larger in magnitude than another (like bedrooms), its gradient will be much larger.  
Thus, with a shared α:

- \(w_0\) (size) updates in big steps and may overshoot.
- \(w_1\) (bedrooms) updates in tiny steps.

This makes gradient descent inefficient and sensitive to α.

---

## 4. Feature Scaling – Z‑score Normalization

The solution is to **normalize each feature** so they have similar ranges.

Z‑score normalization transforms each feature column \(j\):

\[
x_j^{(i)} := \frac{x_j^{(i)} - \mu_j}{\sigma_j}
\]

where:
- \(\mu_j\) = mean of feature \(j\) over all training examples.
- \(\sigma_j\) = standard deviation of feature \(j\).

After this transformation:
- Each feature has mean 0 and standard deviation 1.
- All features are now on a similar scale (typically between -2 and +2).

The lab uses a helper function `zscore_normalize_features` (implemented inside) that returns the normalized features, along with the means and standard deviations (needed later for prediction).

**Effect on gradient descent:**
- Gradients for all weights become comparable in magnitude.
- A single α (e.g., 0.1) works well for all parameters.
- Convergence becomes **much faster** – often orders of magnitude fewer iterations.

---

## 5. Running Gradient Descent on Normalized Data

In the lab, after normalizing the features, they run gradient descent with α = 0.1 for 1000 iterations:

```python
w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 1000, 1.0e-1)
```

Notice:
- The cost drops from ~5.8e4 to ~2.19e2 in just 1000 steps.
- The gradients become tiny quickly, indicating convergence.
- The final weights are [110.56, –21.27, –32.71, –37.97] with b = 363.16.

These weights are **for the normalized features**. To interpret them, you would need to “un‑normalize” or use them with normalized inputs.

---

## 6. Making Predictions with Normalized Model

When you want to predict a new house (e.g., 1200 sqft, 3 bedrooms, 1 floor, 40 years), you must **normalize its features using the same μ and σ** computed from the training set:

```python
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
prediction = np.dot(x_house_norm, w_norm) + b_norm
```

This yields a price of about $318,709.

**Important:** Never re‑compute μ and σ on new data – always use the training set statistics.

---

## 7. Visualizing the Difference – Cost Contours

The lab includes a nice visualisation (via `plt_equal_scale`) showing the cost contours for two features:

- **Before scaling**: The contour plot is extremely elongated – changes in one feature affect cost much more than changes in the other. Gradient descent takes a zig‑zag path.
- **After scaling**: The contours become nearly circular. Gradient descent moves straight toward the minimum.

This is why scaled features allow a larger α and much faster convergence.

---

## 8. Summary of Key Takeaways

| Concept | Without scaling | With scaling |
|---------|----------------|--------------|
| Feature ranges | Very different (e.g., 300–3000 vs 2–5) | All about ±2 |
| Gradients | Large for big‑range features, small for small‑range | Balanced |
| Learning rate α | Must be tiny to avoid divergence, leading to slow convergence | Can be relatively large (e.g., 0.1) |
| Convergence speed | Slow (needs 10⁵+ iterations) | Fast (a few hundred iterations) |
| Robustness | Very sensitive to α | Works for a wide range of α |

**Practical rule of thumb:** Always scale your features before applying gradient descent when features have different units or magnitudes. Z‑score normalization (also called standardisation) is a safe and common choice.

---

## 9. Connection to Your Earlier Work

Recall the vectorised cost and gradient functions:

\[
\text{errors} = X\mathbf{w} + b - \mathbf{y}
\]
\[
J = \frac{1}{2m} (\text{errors}^T \text{errors})
\]
\[
\nabla_{\mathbf{w}} J = \frac{1}{m} X^T \text{errors}, \quad \frac{\partial J}{\partial b} = \frac{1}{m} \sum \text{errors}
\]

These formulas work exactly the same whether you use raw features or scaled features. The only difference is the numerical values inside \(X\). By scaling, you make the optimisation problem “better conditioned”, allowing gradient descent to find the solution quickly.

---

## 10. What to Do Next

After completing this lab, you understand:
- Why feature scaling is essential for gradient descent.
- How to implement z‑score normalisation.
- How to interpret and apply a model trained on scaled features.

The next lab (Feature Engineering and Polynomial Regression) will show you how to create new features (e.g., \(x^2, x^3\)) and why scaling becomes even more critical then.
