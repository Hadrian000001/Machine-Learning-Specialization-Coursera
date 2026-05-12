# Teaching Feature Engineering & Polynomial Regression (Super Simple!)

Alright, let's imagine you’re trying to draw a straight line through some dots on a graph, but the dots make a curve (like a U‑shape). A straight line will *never* fit a curve perfectly – that’s the problem linear regression has when your data is not a straight line.

But you can trick linear regression into fitting a curve by creating **new features** from your original data. This is called **feature engineering**.

Let me explain step by step, as if we’re learning together from scratch.

---

## 1. The problem: A straight line can’t fit a curve

Suppose you have data that follows \( y = 1 + x^2 \).  
If you plot it, it looks like a U‑shape (a parabola).  
If you try to fit a straight line \( y = w x + b \) to it, you will get a terrible fit. No matter how much you change `w` and `b`, the line will always miss the curve.

The notebook shows this: they generate `y = 1 + x^2` and try to fit it with just `x`. The cost is high and the predictions are bad.

---

## 2. The trick: Create a new feature `x²`

What if instead of giving the model just `x`, you give it `x²`?  
Then the model becomes:

\[
y = w \cdot (x^2) + b
\]

That’s still a **linear regression** because it’s linear in the parameter `w`. But now the feature is `x²`, so the prediction can follow a parabola. Perfect!

In the notebook, they do exactly this:

```python
X = x**2           # new feature = x squared
X = X.reshape(-1,1)   # make it a 2D array
```

Then they run gradient descent. The cost drops a lot, and the learned weights become `w ≈ 1`, `b ≈ 0.049`, which is almost the true `y = 1 * x² + 1`. The fit is nearly perfect.

---

## 3. What if we don’t know which power to use?

Sometimes you don’t know if the relation is `x²`, `x³`, or a combination.  
You can just add **many polynomial features** – for example, `x`, `x²`, `x³`, … and let gradient descent decide which ones matter.

In the notebook, they create:

```python
X = np.c_[x, x**2, x**3]   # three columns: x, x², x³
```

Then they run gradient descent. The final weights are:

- `w0 ≈ 0.08` (for `x`)
- `w1 ≈ 0.54` (for `x²`)
- `w2 ≈ 0.03` (for `x³`)

The `x²` weight is much larger, meaning the model “picked” the most useful feature. The other weights are tiny (nearly zero), so they barely matter.  
So gradient descent can do **automatic feature selection** – it makes the weights for useless features very small.

---

## 4. Wait – why do we need to scale the features?

Now look at the ranges of `x`, `x²`, and `x³` when `x` goes from 0 to 20:

- `x` goes from 0 to 20
- `x²` goes from 0 to 400
- `x³` goes from 0 to 8000

These numbers are **very different in size**. That’s a problem for gradient descent.  
The gradient for `w2` (the coefficient of `x³`) will be huge compared to the gradient for `w0` (coefficient of `x`). So with one learning rate `α`, the updates for different weights will behave very differently – some will jump around, others will move slowly.

The solution: **scale all features** so they have similar ranges (like between -2 and +2).  
The notebook uses **z‑score normalization**:

\[
x_{\text{norm}} = \frac{x - \text{mean}}{\text{std}}
\]

After scaling, all features have mean 0 and standard deviation 1. They become comparable in size.

Then you can use a much larger learning rate (like `α = 0.1`) and gradient descent converges **much faster**.

---

## 5. Even complex functions (like cosine) can be fit

The notebook ends with an example: they try to fit `y = cos(x/2)` using polynomials up to `x¹³`. With scaling, gradient descent finds a model that approximates the cosine curve very well.  
So with enough polynomial terms and scaling, linear regression can model **very non‑linear** shapes.

---

## 6. The cost function and the `error * error` part

Remember the cost function for linear regression:

\[
J = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2
\]

In vectorised code, we write:

```python
predictions = X @ w + b
errors = predictions - y
cost = (1/(2*m)) * (errors @ errors)   # errors @ errors = dot product of errors with itself
```

That `errors @ errors` is just the sum of squares of all the errors. It doesn’t matter whether we use raw features or engineered features – the cost is always computed the same way.

---

## Summary (in plain English)

- **Linear regression can only draw straight lines** – but we can **create new features** (like `x²`, `x³`) to make it draw curves.
- **Add many polynomial features** and let gradient descent figure out which ones are important (it will make useless weights tiny).
- **Different features can have very different sizes** – that slows down gradient descent. So **always scale your features** (make them all around the same size). Z‑score normalization is a good choice.
- **With scaling**, you can use a larger learning rate and the model learns much faster.
- **The cost function still uses `errors @ errors`** – nothing changes there.

Now you know how to make linear regression fit curves – by adding polynomial features and scaling them properly. This is super useful in real life, because most data is not a straight line!
