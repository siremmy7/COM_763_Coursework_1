# ============================================
# Real Estate Price Prediction - ML Pipeline
# Models: Random Forest, Linear Regression, Decision Tree
# Includes: Full EDA, Evaluation, and Visualization
# ============================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import pickle

# ============================================
# 2. Load Dataset
# ============================================
data = pd.read_csv('Real estate.csv')

# ============================================
# 3. Data Cleaning
# ============================================

# Remove ID column if exists
if 'No' in data.columns:
    data.drop(columns=['No'], inplace=True)

# Remove missing values
data.dropna(inplace=True)

# ============================================
# 4. Exploratory Data Analysis (EDA)
# ============================================

# 4.1 Feature Distributions
data.hist(bins=20, figsize=(12, 8))
plt.suptitle("Feature Distributions")
plt.show()



# ============================================
# 5. Define Features and Target
# ============================================

target_name = data.columns[-1]
Y = data[target_name]
X = data.drop(columns=[target_name])

# ============================================
# 6. Handle Categorical Variables
# ============================================

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype('category').cat.codes

# ============================================
# 7. Train-Test Split
# ============================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

# ============================================
# 8. Train Random Forest
# ============================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    oob_score=True,
    bootstrap=True
)

rf_model.fit(X_train, Y_train)

print(f"\nRandom Forest OOB Score: {rf_model.oob_score_:.4f}")

# ============================================
# 9. Predictions
# ============================================

Y_pred_rf = rf_model.predict(X_test)

# ============================================
# 10. Evaluation Function
# ============================================

def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

rmse_rf, mae_rf, r2_rf = evaluate_model(Y_test, Y_pred_rf)

print(X.columns)

# ============================================
# 11. Linear Regression
# ============================================

lm_model = LinearRegression()
lm_model.fit(X_train, Y_train)

Y_pred_lm = lm_model.predict(X_test)
rmse_lm, mae_lm, r2_lm = evaluate_model(Y_test, Y_pred_lm)

# ============================================
# 12. Decision Tree
# ============================================

tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, Y_train)

Y_pred_tree = tree_model.predict(X_test)
rmse_tree, mae_tree, r2_tree = evaluate_model(Y_test, Y_pred_tree)

# ============================================
# 13. Cross Validation
# ============================================

cv_scores = cross_val_score(rf_model, X, Y, cv=5, scoring='r2')
print(f"\nCross-validation R2 Score: {cv_scores.mean():.4f}")

# ============================================
# 14. Results Display
# ============================================

print("\n=== MODEL COMPARISON ===")

print(f"\nRandom Forest:\nRMSE: {rmse_rf:.2f} | MAE: {mae_rf:.2f} | R2: {r2_rf:.2f}")
print(f"\nLinear Regression:\nRMSE: {rmse_lm:.2f} | MAE: {mae_lm:.2f} | R2: {r2_lm:.2f}")
print(f"\nDecision Tree:\nRMSE: {rmse_tree:.2f} | MAE: {mae_tree:.2f} | R2: {r2_tree:.2f}")

# ============================================
# 15. Model Comparison Plots
# ============================================

models = ['Linear Regression', 'Decision Tree', 'Random Forest']

# RMSE
plt.figure()
plt.bar(models, [rmse_lm, rmse_tree, rmse_rf])
plt.title("Model Comparison (RMSE)")
plt.ylabel("RMSE")
plt.grid()
plt.show()

# MAE
plt.figure()
plt.bar(models, [mae_lm, mae_tree, mae_rf])
plt.title("Model Comparison (MAE)")
plt.ylabel("MAE")
plt.grid()
plt.show()

# R2
plt.figure()
plt.bar(models, [r2_lm, r2_tree, r2_rf])
plt.title("Model Comparison (R2 Score)")
plt.ylabel("R2")
plt.grid()
plt.show()

# ============================================
# 16. Actual vs Predicted
# ============================================

plt.figure()
plt.scatter(Y_test, Y_pred_rf)

# Ideal line
plt.plot([Y_test.min(), Y_test.max()],
         [Y_test.min(), Y_test.max()])

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted (Random Forest)")
plt.grid()
plt.show()

# ============================================
# 17. Residual Analysis
# ============================================

residuals = Y_test - Y_pred_rf

# Residual scatter
plt.figure()
plt.scatter(Y_pred_rf, residuals)
plt.axhline(y=0)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid()
plt.show()

# Residual distribution
plt.figure()
plt.hist(residuals, bins=30)
plt.title("Residual Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.grid()
plt.show()

# ============================================
# 18. Feature Importance
# ============================================

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=True)

plt.figure()
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.grid()
plt.show()



plt.figure(figsize=(8,5))
plt.hist(data["Y house price of unit area"], bins=30)

plt.title("House Price Distribution")
plt.xlabel("House Price (unit area)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()




corr = data.corr()

plt.figure(figsize=(10,6))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.title("Feature Correlation Heatmap")
plt.show()





features = [
    "X1 transaction date",
    "X2 house age",
    "X3 distance to the nearest MRT station",
    "X4 number of convenience stores",
    "X5 latitude",
    "X6 longitude"
]

target = "Y house price of unit area"

plt.figure(figsize=(15,10))

for i, col in enumerate(features):
    plt.subplot(2, 3, i+1)
    plt.scatter(data[col], data[target], alpha=0.5)
    plt.title(f"{col} vs House Price")
    plt.xlabel(col)
    plt.ylabel("House Price")

plt.tight_layout()
plt.show()


# ============================================
# 19. Save Model
# ============================================

with open('model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print("\nModel saved successfully!")