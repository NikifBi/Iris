import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader_iris import load_data
import time


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import preprocessing as prep
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


###Timer decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} took {end_time - start_time:.6f} seconds")
        return result
    return wrapper


df = load_data()

### Some basic info
#print(df.info())
#print(df.head())

### scatter plots of everything
#sns.pairplot(df, hue = 'Class', diag_kind = 'hist')
#plt.show()

### Characteristic selection and train-test split
X = df[['Petal Length','Petal Width']]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state = 123)

scalers = ['passthrough', prep.StandardScaler(), prep.MinMaxScaler(), prep.RobustScaler()]


###Implementation of the KNN classifier and hyperparameter tuning

@timer
def KNN_classifier():

    pipe_knn = Pipeline([
        ('scaler', 'passthrough'),
        ('model', KNeighborsClassifier())
    ])

    param_grid_knn = {
        'scaler':scalers,
        'model__n_neighbors':[3,5,7,9]
        }

    clf_knn = GridSearchCV(pipe_knn, param_grid_knn, n_jobs=-1, cv = 10,verbose = 1)
    clf_knn.fit(X_train, y_train)


    ### Results of GridSearchCV on a readable dataframe
    results_knn = pd.DataFrame(clf_knn.cv_results_)
    results_knn = results_knn[
        [
            'param_scaler',
            'param_model__n_neighbors',
            'mean_test_score',
            'std_test_score',
            'rank_test_score'
        ]
    ].sort_values('rank_test_score')

    ### Uncomment to print results:

    #print(results_knn)

    ### Evaluation metrics

    y_pred_knn = clf_knn.best_estimator_.predict(X_test)

    print("Test Accuracy:", accuracy_score(y_test, y_pred_knn))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))

    #fig, ax = plt.subplots(figsize=(6, 5))
    #display = DecisionBoundaryDisplay.from_estimator(knn, df2, response_method= 'predict', eps=0.1, ax= ax)
    #ax.scatter(df2.iloc[:, 0], df2.iloc[:, 1], c = y_train, edgecolor="k")
    #plt.show()



###Implementation of the SVM classifier and hyperparameter tuning

@timer
def SVM_classifier():

    pipe_svm = Pipeline([
        ('scaler', 'passthrough'),
        ('model', SVC())
    ])

    coarse_param_grid_svm= [
        {'scaler':scalers,
        'model__kernel':['linear'],                   #Linear kernel
        'model__C':np.logspace(-4,4,100)},

        {'scaler': scalers,
        'model__kernel':['rbf'],                      #RBF kernel
        'model__C':np.logspace(-4,4,100),
        'model__gamma':np.logspace(-5, 1, 50)},

        {'scaler':scalers,
        'model__kernel':['poly'],                     #Polynomial kernel
        'model__C':np.logspace(-4,4,100),
        'model__gamma':np.logspace(-5, 1, 50),
        'model__degree':[1,2,3,4,5]}  

    ]

    ###Coarse random search

    coarse_clf_svm = RandomizedSearchCV(pipe_svm, coarse_param_grid_svm, n_iter = 50, n_jobs = -1, cv = 10, random_state = 42, verbose = 1)
    coarse_clf_svm.fit(X_train, y_train)

    #print(coarse_clf_svm.best_params_)

    ###Precise Grid search

    param_grid_svm = {}

    best_C = coarse_clf_svm.best_params_['model__C']
    param_grid_svm['model__C'] = np.linspace(best_C/5,best_C*5,10)

    if not coarse_clf_svm.best_params_['model__kernel'] == 'linear':
        best_gamma = coarse_clf_svm.best_params_['model__gamma']
        param_grid_svm['model__gamma'] = np.linspace(best_gamma/5,best_gamma*5,10)


    pipe_svm = Pipeline([
        ('scaler',  coarse_clf_svm.best_params_['scaler']),
        ('model', SVC(kernel = coarse_clf_svm.best_params_['model__kernel']))
    ])


    clf_svm = GridSearchCV(pipe_svm, param_grid_svm, n_jobs=-1, cv = 10,verbose = 1)
    clf_svm.fit(X_train, y_train)

    ### Results of GridSearchCV on a readable dataframe
    results_svm = pd.DataFrame(clf_svm.cv_results_)
    results_svm = results_svm[
        [
            'param_model__C',
            'param_model__gamma',
            'mean_test_score',
            'std_test_score',
            'rank_test_score'
        ]
    ].sort_values('rank_test_score')

    ### Uncomment to print results:
    #print(results_svm)

    ### Evaluation metrics

    y_pred_svm = clf_svm.best_estimator_.predict(X_test)

    #print("Test Accuracy:", accuracy_score(y_test, y_pred_svm))
    #print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))
    #print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))



###Implementation of the LogisticRegression classifier and hyperparameter tuning

@timer
def LogReg_classifier():

    pipe_logreg = Pipeline([
        ('scaler', 'passthrough'),
        ('model', LogisticRegression())
    ])

    coarse_param_grid_logreg = {
        'scaler':scalers,
        'model__C':np.logspace(-4,4,100) 
        }

    ###Coarse random search

    coarse_clf_logreg = RandomizedSearchCV(pipe_logreg, coarse_param_grid_logreg, n_iter = 50, n_jobs=-1, cv = 10, random_state = 42, verbose = 1)
    coarse_clf_logreg.fit(X_train, y_train)

    #print(coarse_clf_logreg.best_params_)

    ###Precise Grid search

    pipe_logreg = Pipeline([
        ('scaler',  coarse_clf_logreg.best_params_['scaler']),
        ('model', LogisticRegression())
    ])


    param_grid_logreg = {'model__C':np.linspace(coarse_clf_logreg.best_params_['model__C']/5,coarse_clf_logreg.best_params_['model__C']*5,100)}

    clf_logreg = GridSearchCV(pipe_logreg, param_grid_logreg, n_jobs=-1, cv = 10,verbose = 1)
    clf_logreg.fit(X_train, y_train)

    ### Results of GridSearchCV on a readable dataframe
    results_logreg = pd.DataFrame(clf_logreg.cv_results_)
    results_logreg = results_logreg[
        [
            'param_model__C',
            'mean_test_score',
            'std_test_score',
            'rank_test_score'
        ]
    ].sort_values('rank_test_score')

    ### Uncomment to print results:
    #print(results_logreg)

    ### Evaluation metrics

    y_pred_logreg = clf_logreg.best_estimator_.predict(X_test)

    #print("Test Accuracy:", accuracy_score(y_test, y_pred_logreg))
    #print("\nClassification Report:\n", classification_report(y_test, y_pred_logreg))
    #print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))



###Implementation of the Random Forest classifier and hyperparameter tuning

@timer
def RandomForest_classifier():

    pipe_rf = Pipeline([
        ('model', RandomForestClassifier(random_state=42))
    ])

    param_grid_rf = {
        "model__n_estimators": np.arange(200, 1200, 100),
        "model__max_depth": [None] + list(np.arange(2, 50)),
        "model__min_samples_split": np.arange(2, 20),
        "model__min_samples_leaf": np.arange(1, 20),
        "model__max_features": ["sqrt", "log2", None]
    }

    clf_rf = RandomizedSearchCV(pipe_rf, param_grid_rf, n_iter = 100, n_jobs=-1, cv = 10, random_state = 42, verbose = 1)
    clf_rf.fit(X_train, y_train)

    print(clf_rf.best_params_) 

    ### Results of RandomizedSearchCV on a readable dataframe
    results_rf = pd.DataFrame(clf_rf.cv_results_)
    results_rf = results_rf[
        [
            'param_model__n_estimators',
            'param_model__min_samples_split',
            'param_model__min_samples_leaf',
            'param_model__max_features',
            'param_model__max_depth',
            'mean_test_score',
            'std_test_score',
            'rank_test_score'
        ]
    ].sort_values('rank_test_score')
    

    ### Uncomment to print results:
    #print(results_rf)

    ### Evaluation metrics

    y_pred_rf = clf_rf.best_estimator_.predict(X_test)

    #print("Test Accuracy:", accuracy_score(y_test, y_pred_rf))
    #print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
    #print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))



###Implementation of the MLP classifier and hyperparameter tuning

@timer
def MLP_classifier():

    pipe_mlp = Pipeline([
        ('scaler', prep.StandardScaler()),
        ('model', MLPClassifier(hidden_layer_sizes=(50,),
                                solver = 'lbfgs',
                                random_state = 123,
                                max_iter=1000))
    ])

    param_grid_mlp = {
        'scaler': [prep.StandardScaler(),prep.MinMaxScaler(),prep.RobustScaler()],
        'model__hidden_layer_sizes': [(50,), (100,), (50, 50)],
    }
    
    clf_mlp = GridSearchCV(pipe_mlp, param_grid_mlp, n_jobs=-1, cv = 10,verbose = 1)
    clf_mlp.fit(X_train, y_train)

    ### Results of GridSearchCV on a readable dataframe
    results_mlp = pd.DataFrame(clf_mlp.cv_results_)
    #print(results_mlp.columns)
    results_mlp = results_mlp[
        [
            'param_model__hidden_layer_sizes',
            'param_scaler',
            'mean_test_score',
            'std_test_score',
            'rank_test_score'
        ]
    ].sort_values('rank_test_score')
    

    ### Uncomment to print results:

    #print(results_mlp)

    ### Evaluation metrics

    y_pred_mlp = clf_mlp.best_estimator_.predict(X_test)

    #print("Test Accuracy:", accuracy_score(y_test, y_pred_mlp))
    #print("\nClassification Report:\n", classification_report(y_test, y_pred_mlp))
    #print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_mlp))



###Implementation of the XGBoost classifier and hyperparameter tuning

@timer
def XGB_classifier():

    pipe_xgb = Pipeline([
        ('model', XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=123))
    ])

    param_grid_xgb = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.1, 0.2],
    }

    clf_xgb = GridSearchCV(pipe_xgb, param_grid_xgb, n_jobs=-1, cv = 10,verbose = 1)
    clf_xgb.fit(X_train, y_train)

    ### Results of GridSearchCV on a readable dataframe
    results_xgb = pd.DataFrame(clf_xgb.cv_results_)
    #print(results_mlp.columns)
    results_xgb = results_xgb[
        [
            'param_model__n_estimators',
            'param_model__max_depth',
            'param_model__learning_rate',
            'mean_test_score',
            'std_test_score',
            'rank_test_score'
        ]
    ].sort_values('rank_test_score')
    

    ### Uncomment to print results:

    #print(results_xgb)

    ### Evaluation metrics

    y_pred_xgb = clf_xgb.best_estimator_.predict(X_test)

    #print("Test Accuracy:", accuracy_score(y_test, y_pred_xgb))
    #print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))
    #print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))




#XGB_classifier()

#KNN_classifier()

#MLP_classifier()


