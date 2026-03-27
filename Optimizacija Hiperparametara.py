import pandas as pd
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

# === Load data ===
#train_df = pd.read_excel("Cr3_dqb_training_set.xlsx")
train_df = pd.read_excel("Cr3_dgb_training_set_pročišćen_proširen.xlsx")

X, y = train_df.iloc[:, 2:10].values, train_df.iloc[:, 1].values

print(X)
print(y)


#train_data = np.random.randint(1, 100, size=(100, 10))
#train_labels = np.random.randint(2, size=(100))


#X,y= CatBoostRegressor(depth=4, iterations=500, learning_rate=0.092,
 #                          l2_leaf_reg=1.9, loss_function='MAE', border_count=64, verbose=0)

model=CatBoostRegressor()

param_grid = {"depth": [3, 4, 5, 6],
             "iterations": [250, 500, 700, 1000, 1500],
             "learning_rate": [0.08, 0.09, 0.0925, 0.0950, 0.1, 0.15],
             "l2_leaf_reg": [1.5, 1.75, 1.9, 2.0, 2.25, 2.5],
            "border_count": [32, 64, 128]}

grid_search_result = model.grid_search(param_grid,X=X,y=y,plot=False)

# Print best parameters and score
print("Best Parameters:", grid_search_result['params'])

results = grid_search_result['cv_results']
plt.plot(results['iterations'], results['test-RMSE-mean'])
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('Grid Search CV Results')
plt.show()

print(f"Best RMSE: {-grid_search_result.best_score_:.4f}")


#grid_search= GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error")

#grid_search_result.fit(X, y)

#grid_search_result.best_params_
