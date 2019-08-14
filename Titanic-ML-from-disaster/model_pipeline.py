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
from sklearn.externals import joblib
from time import time

class ModelPipeline():
    def __init__(self, train_df, test_df, cat_attrib, num_attrib):
        self.train_df = train_df
        self.test_df = test_df
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
        print(f'Time elapsed: {(time() - t0):.2f} seconds')
        return y_pred_results, y_pred_proba_results
