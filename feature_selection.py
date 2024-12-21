""""this code is for hyperparameter tuning, written by Yuncong Mao in December 2024"""

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

data = pd.read_csv('/Users/tobymao/Desktop/neuro research/neuro-onc research/clinical_cov/encoded.csv')
temp = pd.read_csv('/Users/tobymao/Desktop/neuro research/neuro-onc research/clinical_cov/filtered_clean.csv')
y = temp['SurvFromDx(mo)']
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

# SHAP analysis for feature inclusion
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)
explainer = shap.Explainer(rf_model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, feature_names=data.columns)
shap_importances = np.abs(shap_values.values).mean(axis=0)
shap_feature_importance = pd.DataFrame({
    'Feature': data.columns,
    'Importance': shap_importances
}).sort_values(by='Importance', ascending=False)

# Get top features
top_features = shap_feature_importance['Feature'].head(20).tolist()
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

# Use IQR to remove outliers
Q1 = y_train.quantile(0.25)
Q3 = y_train.quantile(0.75)
IQR = Q3 - Q1
outlier_mask = (y_train >= (Q1 - 1.5 * IQR)) & (y_train <= (Q3 + 1.5 * IQR))
X_train_top, y_train = X_train_top[outlier_mask], y_train[outlier_mask]

# manage skewed right y dist using log-transformation
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

#Cross-Validation and Hyperparameter Tuning for Multiple Models
models = {
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "KNN": KNeighborsRegressor(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Neural Network": MLPRegressor(random_state=42, max_iter=500)
}

param_grids = {
    "AdaBoost": {
        'n_estimators': [50, 100, 150, 200, 250],
        'learning_rate': [0.01, 0.1, 1.0]
    },
    "Decision Tree": {
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10]
    },
    "KNN": {
        'n_neighbors': [3, 5, 10],
        'weights': ['uniform', 'distance']
    },
    "Random Forest": {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10]
    },
    "Neural Network": {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'learning_rate_init': [0.001, 0.01, 0.1]
    }
}

best_models = {}
for model_name, model in models.items():
    print(f"Tuning {model_name}...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[model_name],
        scoring='neg_root_mean_squared_error',
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train_top, y_train_log)
    best_models[model_name] = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")

# Evaluate the final models
evaluation_results = {}
for model_name, model in best_models.items():
    y_pred_log = model.predict(X_test_top)
    y_pred = np.expm1(y_pred_log)  
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    evaluation_results[model_name] = rmse
    print(f"{model_name} RMSE: {rmse}")

# Print and compare RMSE across models
print("\nComparison of RMSE for Best Models:")
for model_name, rmse in evaluation_results.items():
    print(f"{model_name}: RMSE = {rmse:.4f}")

eval_df = pd.DataFrame(list(evaluation_results.items()), columns=["Model", "RMSE"])
print("\nFinal Model Evaluation:")
print(eval_df)
