from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, roc_auc_score, roc_curve
from time import time
import joblib
import numpy as np
import matplotlib.pyplot as plt

class ModelPipeline():
    """Class for creating model pipeline that uses the Simpleimputer class (with the 'medium' strategy) for numerical features and the OneHotEncoder class for categorical features."""

    def __init__(self, train_df, cat_attrib, num_attrib):
        self.train_df = train_df
        self.cat_attrib = cat_attrib
        self.num_attrib = num_attrib

    def full_pipeline(self):
        num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('std_scaler', StandardScaler())])

        full_pipeline = ColumnTransformer([('num', num_pipeline, self.num_attrib), ('cat', OneHotEncoder(), self.cat_attrib)])

        X_prepared = full_pipeline.fit_transform(self.train_df.drop('Survived', axis=1))
        y = self.train_df['Survived']

        X_train, X_test, y_train, y_test = train_test_split(X_prepared, y, test_size = 0.3, random_state=42)

        return X_train, X_test, y_train, y_test

class BinaryModelTraining():
    """Class for model fitting according to the dictionary of model parameters used as argument"""

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self, model_dict):
        t0 = time()
        y_pred_results = []
        y_pred_proba_results = []
        for name, model in model_dict.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            y_pred_results.append(y_pred)
            y_pred_proba = model.predict_proba(self.X_test)
            y_pred_proba_results.append(y_pred_proba)
            joblib.dump(model, '_'.join(name.lower().split(' '))+'.pkl')
            print(f'Accuracy of the {name.lower()} on test set: {model.score(self.X_test, self.y_test):.4f}')
        print(f'Time elapsed for training: {(time() - t0):.2f} seconds\n')
        return y_pred_results, y_pred_proba_results

class BinaryClassMetrics():
    """Class for the calculation of the mean squared error, confusion matrix, precision, recall, F-measure, support and ROC area-under-the-curve score"""

    def __init__(self, model_dict, y_pred_results, y_pred_proba_results):
        self.model_dict = model_dict
        self.y_pred_results = y_pred_results
        self.y_pred_proba_results = y_pred_proba_results

    def compute_metrics(self, y_test):
        """Requires the y_test array as a parameter"""

        for (name, _), y_pred in zip(self.model_dict.items(), self.y_pred_results):
            mse = mean_squared_error(y_test, y_pred)
            print(f'Root mean square error for the {name.lower():s} model: {np.sqrt(mse):.3f}')
        print('\n', end='')

        for (name, _), y_pred in zip(self.model_dict.items(), self.y_pred_results):
            cm = confusion_matrix(y_test, y_pred)
            print(f'Confusion matrix for the {name.lower()} model: \n{cm}\n')

        for (name, _), y_pred in zip(self.model_dict.items(), self.y_pred_results):
            class_report = classification_report(y_test, y_pred)
            print(f'Precision, recall, F-measure and support for the {name.lower()} model: \n{class_report}\n')

    def compute_plots(self, y_test):
        """Requires the y_test array as a parameter"""

        for (name, _), y_pred, y_pred_proba in zip(self.model_dict.items(), self.y_pred_results, self.y_pred_proba_results):
            model_roc_auc = roc_auc_score(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:,1])
            _, axes = plt.subplots(figsize=(10, 8))
            axes.plot(fpr, tpr, marker='.', ms=6, label='Model: {0:s}, Regression (area = {1:.3f})'.format(name.lower(), model_roc_auc))
            axes.plot([0, 1], [0, 1],'r--')
            axes.set_xlim([0.0, 1.0])
            axes.set_ylim([0.0, 1.05])
            axes.set_xlabel('False Positive Rate')
            axes.set_ylabel('True Positive Rate')
            axes.set_title('Receiver operating characteristic for the {0:s} model'.format(name.lower()))
            axes.legend(loc="lower right")
            plt.grid(True, linestyle='--')
