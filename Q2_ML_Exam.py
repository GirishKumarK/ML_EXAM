import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import recall_score as r_s
from sklearn.metrics import precision_score as p_s
from sklearn.metrics import accuracy_score as a_s
from sklearn.metrics import confusion_matrix as c_m

data_path = '.\\Exam Datasets'
filename = 'classification_data.tsv'
data_file = os.path.join(data_path, filename)

dataset = pd.read_csv(data_file, sep='\t')

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

# split dataset to batches for krr
def split_batches(dataset, num_batches):
    l = dataset.shape[0]
    s = 1.0 / num_batches
    batches = []
    for b in range(num_batches):
        batch = dataset[int(b * s * l) : int((b + 1) * s * l)]
        batches.append(batch)
    return batches

# split data
def train_test_split(dataset):
    percent = 0.80
    data_train = dataset[ : int(percent * dataset.shape[0])]
    data_test = dataset[int(percent * dataset.shape[0]) : ]
    return data_train, data_test

# Split the dataset into 5 folds
def split_kfold(dataset, folds):
    l = dataset.shape[0]
    s = 1.0 / folds
    splits = []
    for k in range(folds):
        split = dataset[int(k * s * l) : int((k + 1) * s * l)]
        splits.append(split)
    return splits

def get_X_Y(dataset):
    X = np.asarray([[row['Red'], row['Blue'], 
                     row['Green']] for index, row in dataset.iterrows()])
    Y = np.asarray([row['Class'] for index, row in dataset.iterrows()])
    return X, Y

data_cv, data_test = train_test_split(dataset)
cv_splits = split_kfold(data_cv, 5)

# krr batches
data_batches = split_batches(dataset, 15)

# KRR
def kl():
    # linear
    print ()
    tl = time.time()
    recs, precs, accs = [], [], []
    
    alphas = [1.0]
    lin_krr = KernelRidge(alpha=alphas, kernel='linear')
    hyperparams = {'alpha' : alphas}
    
    lin_krr_clf = GridSearchCV(lin_krr, hyperparams, cv=5)
    for batch in data_batches:
        batch_train, batch_test = train_test_split(batch)
        
        cv_train_X, cv_train_Y = get_X_Y(batch_train)
        lin_krr_fit = lin_krr_clf.fit(cv_train_X, cv_train_Y)
        lin_krr_res = lin_krr_clf.cv_results_
        lin_krr_params = lin_krr_clf.best_params_
        lin_krr_score = lin_krr_clf.best_score_
        
        test_X, test_Y = get_X_Y(batch_test)
        test_pred = lin_krr_clf.predict(test_X)
        # The Coefficients
        print('Test Estimator : \n', lin_krr_clf.best_estimator_)
        # Recall Score
        recall = r_s(test_Y, test_pred)
        print('Recall Score : \n', recall)
        # Precision Score
        precision = p_s(test_Y, test_pred)
        print('Precision Score : \n', precision)
        # Accuracy Score
        accuracy = a_s(test_Y, test_pred)
        print('Accuracy Score : \n', accuracy)
        # Conusion Matix
        print('Confusion Matrix : \n', c_m(test_Y, test_pred))
        recs.append(recall)
        precs.append(precision)
        accs.append(accuracy)
        print ()
    print('Average Test Recall Score : %f' % np.mean(recs))
    print('Average Test Precision Score : %f' % np.mean(precs))
    print('Average Test Accuracy Score : %f' % np.mean(accs))
    
    tl = time.time() - tl
    print ('Time Secs : %f' % tl)
    return None

def kp():
    # polynomial
    print ()
    tp = time.time()
    recs, precs, accs = [], [], []
    
    alphas = [1.0]
    degs = [2.0, 3.0] # M
    hyperparams = {'alpha' : alphas, 'degree' : degs}
    poly_krr = KernelRidge(kernel='poly', alpha=alphas, degree=degs, gamma=1, coef0=1)
    
    poly_krr_clf = GridSearchCV(poly_krr, hyperparams, cv=5)
    for batch in data_batches:
        batch_train, batch_test = train_test_split(batch)
    
        
        cv_train_X, cv_train_Y = get_X_Y(batch_train)
        poly_krr_fit = poly_krr_clf.fit(cv_train_X, cv_train_Y)
        poly_krr_res = poly_krr_clf.cv_results_
        poly_krr_params = poly_krr_clf.best_params_
        poly_krr_score = poly_krr_clf.best_score_
        
        test_X, test_Y = get_X_Y(batch_test)
        test_pred = poly_krr_clf.predict(test_X)
        # The Coefficients
        print('Test Estimator : \n', poly_krr_clf.best_estimator_)
        # Recall Score
        recall = r_s(test_Y, test_pred)
        print('Recall Score : \n', recall)
        # Precision Score
        precision = p_s(test_Y, test_pred)
        print('Precision Score : \n', precision)
        # Accuracy Score
        accuracy = a_s(test_Y, test_pred)
        print('Accuracy Score : \n', accuracy)
        # Conusion Matix
        print('Confusion Matrix : \n', c_m(test_Y, test_pred))
        recs.append(recall)
        precs.append(precision)
        accs.append(accuracy)
        print ()
    print('Average Test Recall Score : %f' % np.mean(recs))
    print('Average Test Precision Score : %f' % np.mean(precs))
    print('Average Test Accuracy Score : %f' % np.mean(accs))
    
    tp = time.time() - tp
    print ('Time Secs : %f' % tp)
    return None

def kg():
    # gaussian/rbf
    print ()
    tg = time.time()
    recs, precs, accs = [], [], []
    
    alphas = [1.0]
    sigmas = [0.1, 0.5, 1.0, 2.0, 4.0]
    hyperparams = {'alpha' : alphas, 'gamma' : sigmas}
    rbf_krr = KernelRidge(kernel='rbf', alpha=alphas, gamma=sigmas)
    
    rbf_krr_clf = GridSearchCV(rbf_krr, hyperparams, cv=5)
    for batch in data_batches:
        batch_train, batch_test = train_test_split(batch)
    
        cv_train_X, cv_train_Y = get_X_Y(batch_train)
        rbf_krr_fit = rbf_krr_clf.fit(cv_train_X, cv_train_Y)
        rbf_krr_res = rbf_krr_clf.cv_results_
        rbf_krr_params = rbf_krr_clf.best_params_
        rbf_krr_score = rbf_krr_clf.best_score_
        
        test_X, test_Y = get_X_Y(batch_test)
        test_pred = rbf_krr_clf.predict(test_X)
        # The Coefficients
        print('Test Estimator : \n', rbf_krr_clf.best_estimator_)
        # Recall Score
        recall = r_s(test_Y, test_pred)
        print('Recall Score : \n', recall)
        # Precision Score
        precision = p_s(test_Y, test_pred)
        print('Precision Score : \n', precision)
        # Accuracy Score
        accuracy = a_s(test_Y, test_pred)
        print('Accuracy Score : \n', accuracy)
        # Conusion Matix
        print('Confusion Matrix : \n', c_m(test_Y, test_pred))
        recs.append(recall)
        precs.append(precision)
        accs.append(accuracy)
        print ()
    print('Average Test Recall Score : %f' % np.mean(recs))
    print('Average Test Precision Score : %f' % np.mean(precs))
    print('Average Test Accuracy Score : %f' % np.mean(accs))
    
    tg = time.time() - tg
    print ('Time Secs : %f' % tg)
    return None

# ince large data - multinomal - sag(faster convergence)
mlr = LogisticRegression(multi_class='multinomial', solver='sag')
# log reg
def run_logreg():
    # CV
    print ()
    recs, precs, accs = [], [], []
    for i in range(len(cv_splits)):
        print ('CV Epoch : ' + str(i + 1))
        cv_train, cv_test = train_test_split(cv_splits[i])
        cv_train_X, cv_train_Y = get_X_Y(cv_train)
        cv_test_X, cv_test_Y = get_X_Y(cv_test)
        mlr_fit = mlr.fit(cv_train_X, cv_train_Y)
        cv_pred = mlr.predict(cv_test_X)
        # The Coefficients
        print('Coefficients : \n', mlr.coef_)
        # Recall Score
        recall = r_s(cv_test_Y, cv_pred)
        print('Recall Score : \n', recall)
        # Precision Score
        precision = p_s(cv_test_Y, cv_pred)
        print('Precision Score : \n', precision)
        # Accuracy Score
        accuracy = a_s(cv_test_Y, cv_pred)
        print('Accuracy Score : \n', accuracy)
        # Conusion Matix
        print('Confusion Matrix : \n', c_m(cv_test_Y, cv_pred))
        recs.append(recall)
        precs.append(precision)
        accs.append(accuracy)
        print ()
    print('Average Recall Score : %f' % np.mean(recs))
    print('Average Precision Score : %f' % np.mean(precs))
    print('Average Accuracy Score : %f' % np.mean(accs))
    print ()
    # Test
    test_X, test_Y = get_X_Y(data_test)
    test_pred = mlr.predict(test_X)
    # The Coefficients
    print('Test Coefficients : \n', mlr.coef_)
    # Recall Score
    recall = r_s(test_Y, test_pred)
    print('Recall Score : \n', recall)
    # Precision Score
    precision = p_s(test_Y, test_pred)
    print('Precision Score : \n', precision)
    # Accuracy Score
    accuracy = a_s(test_Y, test_pred)
    print('Accuracy Score : \n', accuracy)
    # Conusion Matix
    print('Confusion Matrix : \n', c_m(test_Y, test_pred))
    print ()
    return None

# SVC Hard C
def slh():
    # linear
    print ()
    tl = time.time()
    cs = [100000.0]
    lin_svc = SVC(C=cs, kernel='linear', cache_size=4096)
    hyperparams = {'C' : cs}
    
    lin_svc_clf = GridSearchCV(lin_svc, hyperparams, cv=5)
    cv_train_X, cv_train_Y = get_X_Y(data_cv)
    lin_svc_fit = lin_svc_clf.fit(cv_train_X, cv_train_Y)
    lin_svc_res = lin_svc_clf.cv_results_
    lin_svc_params = lin_svc_clf.best_params_
    lin_svc_score = lin_svc_clf.best_score_
    
    test_X, test_Y = get_X_Y(data_test)
    test_pred = lin_svc_clf.predict(test_X)
    # The Coefficients
    print('Test Estimator : \n', lin_svc_clf.best_estimator_)
    # Recall Score
    recall = r_s(test_Y, test_pred)
    print('Recall Score : \n', recall)
    # Precision Score
    precision = p_s(test_Y, test_pred)
    print('Precision Score : \n', precision)
    # Accuracy Score
    accuracy = a_s(test_Y, test_pred)
    print('Accuracy Score : \n', accuracy)
    # Conusion Matix
    print('Confusion Matrix : \n', c_m(test_Y, test_pred))
    tl = time.time() - tl
    print ('Time Secs : %f' % tl)
    return None

def sph():
    # polynomial
    print ()
    tp = time.time()
    cs = [100000.0]
    degs = [2.0, 3.0] # M
    hyperparams = {'C' : cs, 'degree' : degs}
    poly_svc = SVC(kernel='poly', C=cs, degree=degs, gamma=1, coef0=1, cache_size=4096)
    
    poly_svc_clf = GridSearchCV(poly_svc, hyperparams, cv=5)
    cv_train_X, cv_train_Y = get_X_Y(data_cv)
    poly_svc_fit = poly_svc_clf.fit(cv_train_X, cv_train_Y)
    poly_svc_res = poly_svc_clf.cv_results_
    poly_svc_params = poly_svc_clf.best_params_
    poly_svc_score = poly_svc_clf.best_score_
    
    test_X, test_Y = get_X_Y(data_test)
    test_pred = poly_svc_clf.predict(test_X)
    # The Coefficients
    print('Test Estimator : \n', poly_svc_clf.best_estimator_)
    # Recall Score
    recall = r_s(test_Y, test_pred)
    print('Recall Score : \n', recall)
    # Precision Score
    precision = p_s(test_Y, test_pred)
    print('Precision Score : \n', precision)
    # Accuracy Score
    accuracy = a_s(test_Y, test_pred)
    print('Accuracy Score : \n', accuracy)
    # Conusion Matix
    print('Confusion Matrix : \n', c_m(test_Y, test_pred))
    tp = time.time() - tp
    print ('Time Secs : %f' % tp)
    return None

def sgh():
    # gaussian/rbf
    print ()
    tg = time.time()
    cs = [100000.0]
    sigmas = [0.1, 0.5, 1.0, 2.0, 4.0]
    hyperparams = {'C' : cs, 'gamma' : sigmas}
    rbf_svc = SVC(kernel='rbf', C=cs, gamma=sigmas, cache_size=4096)
    
    rbf_svc_clf = GridSearchCV(rbf_svc, hyperparams, cv=5)
    cv_train_X, cv_train_Y = get_X_Y(data_cv)
    rbf_svc_fit = rbf_svc_clf.fit(cv_train_X, cv_train_Y)
    rbf_svc_res = rbf_svc_clf.cv_results_
    rbf_svc_params = rbf_svc_clf.best_params_
    rbf_svc_score = rbf_svc_clf.best_score_
    
    test_X, test_Y = get_X_Y(data_test)
    test_pred = rbf_svc_clf.predict(test_X)
    # The Coefficients
    print('Test Estimator : \n', rbf_svc_clf.best_estimator_)
    # Recall Score
    recall = r_s(test_Y, test_pred)
    print('Recall Score : \n', recall)
    # Precision Score
    precision = p_s(test_Y, test_pred)
    print('Precision Score : \n', precision)
    # Accuracy Score
    accuracy = a_s(test_Y, test_pred)
    print('Accuracy Score : \n', accuracy)
    # Conusion Matix
    print('Confusion Matrix : \n', c_m(test_Y, test_pred))
    tg = time.time() - tg
    print ('Time Secs : %f' % tg)
    return None

# SVC Soft C
def sls():
    # linear
    print ()
    tl = time.time()
    cs = [0.1, 0.5, 1.0, 2.0, 5.0]
    lin_svc = SVC(C=cs, kernel='linear', cache_size=4096)
    hyperparams = {'C' : cs}
    
    lin_svc_clf = GridSearchCV(lin_svc, hyperparams, cv=5)
    cv_train_X, cv_train_Y = get_X_Y(data_cv)
    lin_svc_fit = lin_svc_clf.fit(cv_train_X, cv_train_Y)
    lin_svc_res = lin_svc_clf.cv_results_
    lin_svc_params = lin_svc_clf.best_params_
    lin_svc_score = lin_svc_clf.best_score_
    
    test_X, test_Y = get_X_Y(data_test)
    test_pred = lin_svc_clf.predict(test_X)
    # The Coefficients
    print('Test Estimator : \n', lin_svc_clf.best_estimator_)
    # Recall Score
    recall = r_s(test_Y, test_pred)
    print('Recall Score : \n', recall)
    # Precision Score
    precision = p_s(test_Y, test_pred)
    print('Precision Score : \n', precision)
    # Accuracy Score
    accuracy = a_s(test_Y, test_pred)
    print('Accuracy Score : \n', accuracy)
    # Conusion Matix
    print('Confusion Matrix : \n', c_m(test_Y, test_pred))
    tl = time.time() - tl
    print ('Time Secs : %f' % tl)
    return None

def sps():
    # polynomial
    print ()
    tp = time.time()
    cs = [0.1, 0.5, 1.0, 2.0, 5.0]
    degs = [2.0, 3.0] # M
    hyperparams = {'C' : cs, 'degree' : degs}
    poly_svc = SVC(kernel='poly', C=cs, degree=degs, gamma=1, coef0=1, cache_size=4096)
    
    poly_svc_clf = GridSearchCV(poly_svc, hyperparams, cv=5)
    cv_train_X, cv_train_Y = get_X_Y(data_cv)
    poly_svc_fit = poly_svc_clf.fit(cv_train_X, cv_train_Y)
    poly_svc_res = poly_svc_clf.cv_results_
    poly_svc_params = poly_svc_clf.best_params_
    poly_svc_score = poly_svc_clf.best_score_
    
    test_X, test_Y = get_X_Y(data_test)
    test_pred = poly_svc_clf.predict(test_X)
    # The Coefficients
    print('Test Estimator : \n', poly_svc_clf.best_estimator_)
    # Recall Score
    recall = r_s(test_Y, test_pred)
    print('Recall Score : \n', recall)
    # Precision Score
    precision = p_s(test_Y, test_pred)
    print('Precision Score : \n', precision)
    # Accuracy Score
    accuracy = a_s(test_Y, test_pred)
    print('Accuracy Score : \n', accuracy)
    # Conusion Matix
    print('Confusion Matrix : \n', c_m(test_Y, test_pred))
    tp = time.time() - tp
    print ('Time Secs : %f' % tp)
    return None

def sgs():
    # gaussian/rbf
    print ()
    tg = time.time()
    cs = [0.1, 0.5, 1.0, 2.0, 5.0]
    sigmas = [0.1, 0.5, 1.0, 2.0, 4.0]
    hyperparams = {'C' : cs, 'gamma' : sigmas}
    rbf_svc = SVC(kernel='rbf', C=cs, gamma=sigmas, cache_size=4096)
    
    rbf_svc_clf = GridSearchCV(rbf_svc, hyperparams, cv=5)
    cv_train_X, cv_train_Y = get_X_Y(data_cv)
    rbf_svc_fit = rbf_svc_clf.fit(cv_train_X, cv_train_Y)
    rbf_svc_res = rbf_svc_clf.cv_results_
    rbf_svc_params = rbf_svc_clf.best_params_
    rbf_svc_score = rbf_svc_clf.best_score_
    
    test_X, test_Y = get_X_Y(data_test)
    test_pred = rbf_svc_clf.predict(test_X)
    # The Coefficients
    print('Test Estimator : \n', rbf_svc_clf.best_estimator_)
    # Recall Score
    recall = r_s(test_Y, test_pred)
    print('Recall Score : \n', recall)
    # Precision Score
    precision = p_s(test_Y, test_pred)
    print('Precision Score : \n', precision)
    # Accuracy Score
    accuracy = a_s(test_Y, test_pred)
    print('Accuracy Score : \n', accuracy)
    # Conusion Matix
    print('Confusion Matrix : \n', c_m(test_Y, test_pred))
    tg = time.time() - tg
    print ('Time Secs : %f' % tg)
    return None