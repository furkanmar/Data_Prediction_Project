import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")

data = pd.read_csv('C:\Users\Furkan\Documents\Pratik kod\CE475\Proje\CE475_Proje.csv')

train_data = data.iloc[:100]
X = train_data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']]
y = train_data['Y']


loo = LeaveOneOut()


models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(random_state=42, n_estimators=100),
    'K-Nearest Neighbors Regressor': KNeighborsRegressor(),
    'Support Vector Regressor': SVR(),
    'Lasso Regressor': Lasso(),
    'Ridge Regressor': Ridge(),
    'Bagging Decision Tree Regressor': BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=50, random_state=42)
}


loocv_scores = {}
loocv_accuracy_scores = {}

for name, model in models.items():
    mse_scores = -cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')
    loocv_scores[name] = mse_scores.mean()
    model.fit(X, y)
    y_pred = model.predict(X)
    loocv_accuracy_scores[name] = r2_score(y, y_pred)




print("LOOCV MSE Scores:")
for name, score in loocv_scores.items():
    print(f"{name}: MSE = {score:.2f}")

print("\nLOOCV Accuracy Scores (R^2):")
for name, score in loocv_accuracy_scores.items():
    print(f"{name}: R^2 = {score:.2f}")



best_loocv_model_name = max(loocv_accuracy_scores, key=loocv_accuracy_scores.get)
best_loocv_model = models[best_loocv_model_name]



X_new = data.iloc[100:120][['x1', 'x2', 'x3', 'x4', 'x5', 'x6']]
best_loocv_model.fit(X, y)
y_pred_new_loocv = best_loocv_model.predict(X_new)


data.loc[100:119, 'Y_LOOCV'] = y_pred_new_loocv



output_file_path = 'CE475_Proje_with_predictions.csv'
data.to_csv(output_file_path, index=False, float_format='%.2f')

# Görselleştirme
model_names = list(models.keys())
loocv_accuracy = [loocv_accuracy_scores[name] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 7))
rects1 = ax.bar(x - width/2, loocv_accuracy, width, label='LOOCV')

ax.set_ylabel('Accuracy (R^2)')
ax.set_title('Model Accuracy Comparison: LOOCV vs 5-Fold CV')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend()

fig.tight_layout()
plt.show()
