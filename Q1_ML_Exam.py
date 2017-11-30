import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score as r2_s
from sklearn.metrics import mean_squared_error as m_s_e



data_path = '.\\Exam Datasets'
filename = 'regression_data.csv'
data_file = os.path.join(data_path, filename)

cols = ['bedrooms', 'bathrooms', 
        'floors', 'condition', 'grade', 
        'sqft_above', 'sqft_basement', 
        'sqft_living15', 'sqft_lot15', 'price']
dataset = pd.read_csv(data_file, usecols=cols)

# before norm
sns.pairplot(dataset)

# norm
def normalize(dataset, data_columns):
    for dc in data_columns:
        attribute = np.asarray([row[dc] for index, row in dataset.iterrows()])
        min_attr, max_attr = np.min(attribute), np.max(attribute)
        attr_norm = np.interp(attribute, [min_attr, max_attr], [0, 1])
        dataset[dc] = attr_norm
    return dataset

# norm dataset
dataset = normalize(dataset, cols)

# after norm
sns.pairplot(dataset)

# analyse dataset
data_head = dataset.head()
print ('\n Dataset Head :')
print (data_head)
data_tail = dataset.tail()
print ('\n Dataset Tail :')
print (data_tail)
data_desc = dataset.describe()
print ('\n Dataset Description :')
print (data_desc)
data_corr = dataset.corr()
print ('\n Dataset Correlation :')
print (data_corr)
plt.figure()
sns.heatmap(data_corr, annot=True)

# shuffle dataset once to desquence continuity
dataset = shuffle(dataset)

# split data
def train_test_split(dataset):
    percent = 0.80
    data_train = dataset[ : int(percent * dataset.shape[0])]
    data_test = dataset[int(percent * dataset.shape[0]) : ]
    return data_train, data_test

# Split the dataset into k folds
def split_kfold(dataset, folds):
    l = dataset.shape[0]
    s = 1.0 / folds
    splits = []
    for k in range(folds):
        split = dataset[int(k * s * l) : int((k + 1) * s * l)]
        splits.append(split)
    return splits

def get_X_Y(dataset):
    X = np.asarray([[row['bedrooms'], row['bathrooms'], 
                     row['floors'], row['condition'], 
                     row['grade'], row['sqft_above'], 
                     row['sqft_basement'], row['sqft_living15'], 
                     row['sqft_lot15']] for index, row in dataset.iterrows()])
    Y = np.asarray([row['price'] for index, row in dataset.iterrows()])
    return X, Y
    

data_cv, data_test = train_test_split(dataset)
cv_splits = split_kfold(data_cv, 5)
cv_train_X, cv_train_Y = get_X_Y(data_cv)
test_X, test_Y = get_X_Y(data_test)

ols = LinearRegression()

def run_ols():
    # CV
    print ()
    mses, rmses, rsqs = [], [], []
    for i in range(len(cv_splits)):
        print ('CV Epoch : ' + str(i + 1))
        cv_train, cv_test = train_test_split(cv_splits[i])
        cv_train_X, cv_train_Y = get_X_Y(cv_train)
        cv_test_X, cv_test_Y = get_X_Y(cv_test)
        ols_fit = ols.fit(cv_train_X, cv_train_Y)
        cv_pred = ols.predict(cv_test_X)
        # The Coefficients
        print('Coefficients : \n', ols.coef_)
        # Mean Squared Error
        mse = float(m_s_e(cv_test_Y, cv_pred))
        print('Mean Squared Error : %f' % mse)
        # Root Mean Squared Error
        rmse = float(np.sqrt(m_s_e(cv_test_Y, cv_pred)))
        print('Root Mean Squared Error : %f' % rmse)
        # R^2 (Coefficient of Determination) Regression Score
        rsq = float(r2_s(cv_test_Y, cv_pred))
        print('R^2 Regression Score : %f' % rsq)
        mses.append(mse)
        rmses.append(rmse)
        rsqs.append(rsq)
        print ()
    print('Average Mean Squared Error : %f' % np.mean(mses))
    print('Average Root Mean Squared Error : %f' % np.mean(rmses))
    print('Average R^2 Regression Score : %f' % np.mean(rsqs))
    print ()
    # Test
    test_X, test_Y = get_X_Y(data_test)
    test_pred = ols.predict(test_X)
    # The Coefficients
    print('Test Coefficients : \n', ols.coef_)
    # Mean Squared Error
    mse = float(m_s_e(test_Y, test_pred))
    print('Test Mean Squared Error : %f' % mse)
    # Root Mean Squared Error
    rmse = float(np.sqrt(m_s_e(test_Y, test_pred)))
    print('Test Root Mean Squared Error : %f' % rmse)
    # R^2 (Coefficient of Determination) Regression Score
    rsq = float(r2_s(test_Y, test_pred))
    print('Test R^2 Regression Score : %f' % rsq)
    print ()
    return None

#run_ols()

def run_lso():
    print ()
    alphas = np.array([0.001, 0.01, 0.1, 1.0, 2.0])
    lasso = Lasso(alpha=alphas)
    tuned_parameters = {'alpha': alphas}
    
    lso_clf = GridSearchCV(lasso, tuned_parameters, cv=5)
    
    lso_fit = lso_clf.fit(cv_train_X, cv_train_Y)
    lso_results = lso_clf.cv_results_
    lso_params = lso_clf.best_params_
    lso_score = lso_clf.best_score_
    
    
    test_pred = lso_clf.predict(test_X)
    # The Coefficients
    print('Test Estimator : \n', lso_clf.best_estimator_)
    # Mean Squared Error
    mse = float(m_s_e(test_Y, test_pred))
    print('Test Mean Squared Error : %f' % mse)
    # Root Mean Squared Error
    rmse = float(np.sqrt(m_s_e(test_Y, test_pred)))
    print('Test Root Mean Squared Error : %f' % rmse)
    # R^2 (Coefficient of Determination) Regression Score
    rsq = float(r2_s(test_Y, test_pred))
    print('Test R^2 Regression Score : %f' % rsq)
    return None

#run_lso()

def krr():
    # KRR
    # linear
    print ()
    tl = time.time()
    #alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    alphas = [1.0]
    lin_krr = KernelRidge(alpha=alphas, kernel='linear')
    hyperparams = {'alpha' : alphas}
    
    lin_krr_clf = GridSearchCV(lin_krr, hyperparams, cv=5)
    
    lin_krr_fit = lin_krr_clf.fit(cv_train_X, cv_train_Y)
    lin_krr_res = lin_krr_clf.cv_results_
    lin_krr_params = lin_krr_clf.best_params_
    lin_krr_score = lin_krr_clf.best_score_
    
    
    test_pred = lin_krr_clf.predict(test_X)
    # The Coefficients
    print('Test Estimator : \n', lin_krr_clf.best_estimator_)
    # Mean Squared Error
    mse = float(m_s_e(test_Y, test_pred))
    print('Test Mean Squared Error : %f' % mse)
    # Root Mean Squared Error
    rmse = float(np.sqrt(m_s_e(test_Y, test_pred)))
    print('Test Root Mean Squared Error : %f' % rmse)
    # R^2 (Coefficient of Determination) Regression Score
    rsq = float(r2_s(test_Y, test_pred))
    print('Test R^2 Regression Score : %f' % rsq)
    tl = time.time() - tl
    print ('Time Secs : %f' % tl)
    
    # polynomial
    print ()
    tp = time.time()
    alphas = [1.0]
    degs = [2, 4, 7] # M
    hyperparams = {'alpha' : alphas, 'degree' : degs}
    poly_krr = KernelRidge(kernel='poly', alpha=alphas, degree=degs, gamma=1, coef0=1)
    
    poly_krr_clf = GridSearchCV(poly_krr, hyperparams, cv=5)
    
    poly_krr_fit = poly_krr_clf.fit(cv_train_X, cv_train_Y)
    poly_krr_res = poly_krr_clf.cv_results_
    poly_krr_params = poly_krr_clf.best_params_
    poly_krr_score = poly_krr_clf.best_score_
    
    
    test_pred = poly_krr_clf.predict(test_X)
    # The Coefficients
    print('Test Estimator : \n', poly_krr_clf.best_estimator_)
    # Mean Squared Error
    mse = float(m_s_e(test_Y, test_pred))
    print('Test Mean Squared Error : %f' % mse)
    # Root Mean Squared Error
    rmse = float(np.sqrt(m_s_e(test_Y, test_pred)))
    print('Test Root Mean Squared Error : %f' % rmse)
    # R^2 (Coefficient of Determination) Regression Score
    rsq = float(r2_s(test_Y, test_pred))
    print('Test R^2 Regression Score : %f' % rsq)
    tp = time.time() - tp
    print ('Time Secs : %f' % tp)
    
    # gaussian/rbf
    print ()
    tg = time.time()
    alphas = [1.0]
    sigmas = [0.1, 0.5, 1.0, 2.0, 4.0]
    hyperparams = {'alpha' : alphas, 'gamma' : sigmas}
    rbf_krr = KernelRidge(kernel='rbf', alpha=alphas, gamma=sigmas)
    
    rbf_krr_clf = GridSearchCV(rbf_krr, hyperparams, cv=5)
    
    rbf_krr_fit = rbf_krr_clf.fit(cv_train_X, cv_train_Y)
    rbf_krr_res = rbf_krr_clf.cv_results_
    rbf_krr_params = rbf_krr_clf.best_params_
    rbf_krr_score = rbf_krr_clf.best_score_
    
    
    test_pred = rbf_krr_clf.predict(test_X)
    # The Coefficients
    print('Test Estimator : \n', rbf_krr_clf.best_estimator_)
    # Mean Squared Error
    mse = float(m_s_e(test_Y, test_pred))
    print('Test Mean Squared Error : %f' % mse)
    # Root Mean Squared Error
    rmse = float(np.sqrt(m_s_e(test_Y, test_pred)))
    print('Test Root Mean Squared Error : %f' % rmse)
    # R^2 (Coefficient of Determination) Regression Score
    rsq = float(r2_s(test_Y, test_pred))
    print('Test R^2 Regression Score : %f' % rsq)
    tg = time.time() - tg
    print ('Time Secs : %f' % tg)
    return None


# End of File