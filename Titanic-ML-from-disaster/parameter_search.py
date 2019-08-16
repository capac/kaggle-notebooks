from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

class BestParameterSearch():
    """Finds the best parameters for a specific model using the GridSearch class from scikit-learn."""

    def __init__(self, model, param_grid, scoring, cv, return_train_score):
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.return_train_score = return_train_score
    
    def best_fit_results(self, X_train, y_train):
        grid_search = GridSearchCV(self.model, self.param_grid, self.scoring, self.cv, self.return_train_score)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        cvres = grid_search.cv_results_
        results = []
        for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
            results.append({'RMSE': np.sqrt(-mean_score), 'Parameters': params})
            best_params_df = pd.DataFrame(results).sort_values(by='RMSE', ascending=True).reset_index(drop=True)
        return best_model, best_params_df
