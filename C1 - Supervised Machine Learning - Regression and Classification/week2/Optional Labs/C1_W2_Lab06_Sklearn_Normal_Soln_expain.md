# 🧠 Linear Regression with Scikit-Learn – The Super Simple Version

Welcome! This lab shows you how to use **scikit‑learn** to do linear regression using a **closed‑form solution** (also called the **normal equation**). No gradient descent loops, no learning rate to pick – it just gives you the answer directly.

I’ll explain every tiny step as if you’ve never seen this before.

---

## 📦 What is scikit‑learn?

It’s a free Python library that has many ready‑to‑use machine learning tools. Think of it like a **cookbook** – instead of cooking from scratch, you just follow the recipe and get the dish.

Today we’re using the **`LinearRegression`** tool.

---

## 🧮 Closed‑form solution vs. Gradient Descent

- **Gradient Descent** = you take many small steps downhill to find the best line. You need to choose a learning rate and run many iterations.
- **Closed‑form solution** = you solve a math equation directly to get the best line in one go. No iterations, no learning rate. It’s like using a formula to solve a problem.

Scikit‑learn’s `LinearRegression` uses the closed‑form solution (normal equation). It’s very fast for small datasets.

> **Big plus:** It doesn’t need feature scaling! The numbers can be very different (like 1000 vs 2) and it still works fine.

---

## 📊 First Example: One Feature (House Size)

We start with a tiny dataset:

| Size (1000 sqft) | Price (1000s $) |
|------------------|-----------------|
| 1                | 300             |
| 2                | 500             |

We want to find the line that best fits these two points.

### Step 1: Load the data (easy)

```python
X_train = np.array([1.0, 2.0])   # features (size)
y_train = np.array([300, 500])   # target (price)
```

### Step 2: Create the model

```python
linear_model = LinearRegression()
```

This creates an empty model – like a blank calculator.

### Step 3: Fit the model (find the best line)

```python
linear_model.fit(X_train.reshape(-1, 1), y_train)
```

The `reshape(-1, 1)` is a technical requirement: scikit‑learn wants a **2‑D array** for the features, even if you have only one feature. So we change `[1, 2]` into `[[1],[2]]`.

`fit` means: “Here are my input numbers and the correct answers. Please learn the best `w` (slope) and `b` (intercept).”

### Step 4: See what it learned

```python
b = linear_model.intercept_
w = linear_model.coef_
print(f"w = {w}, b = {b:.2f}")
```

Output: `w = [200.], b = 100.00`

So the line is: **price = 200 × size + 100**  
Check: If size=1 → 200+100=300 ✅; size=2 → 400+100=500 ✅.

### Step 5: Make a prediction for a new house

Say we want the price for a **1200 sqft house**. But careful: our size is in **thousands of sqft**. So 1200 sqft = 1.2 (thousand sqft).

```python
X_test = np.array([[1.2]])   # must be 2D
prediction = linear_model.predict(X_test)
print(prediction)   # prints [340.0]
```

That means $340,000.

---

## 🏠 Second Example: Multiple Features (4 features)

Now we use the same housing data you’ve seen before: size (sqft), bedrooms, floors, age. This time we **don’t need to scale** because closed‑form solution is fine with different scales.

### Load the data

```python
X_train, y_train = load_house_data()   # many rows, 4 columns
```

### Create and fit the model

```python
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
```

This time we don’t reshape because `X_train` is already 2D (rows = houses, columns = features).

### Get the parameters

```python
b = linear_model.intercept_
w = linear_model.coef_
print(f"w = {w}, b = {b:.2f}")
```

Output:  
`w = [0.27  -32.62  -67.25  -1.47], b = 220.42`

These are the coefficients for (size, bedrooms, floors, age).

### Make predictions on the training data

```python
predictions = linear_model.predict(X_train)
```

You can compare these predictions with the real prices. They are close but not perfect – that’s normal.

### Predict a new house

A house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old:

```python
x_house = np.array([[1200, 3, 1, 40]])   # note double brackets!
price = linear_model.predict(x_house)[0]
print(f"${price*1000:.2f}")   # multiply by 1000 because original prices are in $1000s
```

Output: `$318709.09`

---

## ❓ Why no test set?

You’re right – in this lab they only show training predictions. That’s because the goal is just to demonstrate how to use scikit‑learn, not to evaluate generalization. In real work, you would split your data into training and test sets. But for learning the tool, it’s fine.

---

## 🎯 Summary – What you learned

- **Scikit‑learn** gives you a simple way to do linear regression without coding the math.
- **`LinearRegression`** uses the closed‑form solution (normal equation) – very fast for small data.
- **No need to scale features** – it works with raw numbers.
- **Steps:** create model → `fit` → get `coef_` and `intercept_` → `predict`.
- **Predictions** on new data must be in 2D array form.

Now you can do linear regression in three lines of code! 🚀
