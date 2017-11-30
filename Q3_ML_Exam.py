# Imports
#% matplotlib notebook
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

# Fetch Dataset Location
data_path = '.\\Exam Datasets'
filename = 'classification_data2.txt'
data_file = os.path.join(data_path, filename)

# Re-Aligning the Dataset
# Initial Dataset - Tab Spaced with Few (White)Spaces
file = open(data_file, 'r') # Read File in Read-only Mode - Not in Read and Write Mode
lines = file.read().split('\n')
new_lines = []
for line in lines:
    words = line.split('\t') # Split Line by Tab
    for word in words:
        if (word == ''):
            words.remove(word) # Remove White Spaces in the Line
    new_line = []
    for w in range(len(words)):
        new_line.append(words[w])
        new_line.append(',') # Concatenate Words with a Comma
    new_lines.append(new_line[0 : -1]) # Not till End as End-Character is a Comma - So Till -1
file.close()

# Save The New Lines as New File with CSV Extention
new_filename = filename.split('.')
new_filename = new_filename[0] + '.csv'
new_file = os.path.join(data_path, new_filename)

# Create the New File After Removing Tabs and Spaces
file = open(new_file, 'w') # Write File in Write-only mode - Not in Append Mode - File is Made Everytime
for nl in new_lines:
    nline = ''.join(nl)
    file.write(nline + '\n')
file.close()

# Selecting Columns for Dataset - Feature Selection
cols = ['area', 'perimeter', 
        'compactness', 'kernel_length', 'kernel_width', 
        'asym', 'groove_length', 
        'type']

# Get the New Dataset
dataset = pd.read_csv(new_file, usecols=cols)

# Plot Dataset
sns.pairplot(dataset) # Scatters and Histograms

# Dataset Visualization and Analysis
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

# HeatMap Figure of Correlation of Selected Features
plt.figure()
sns.heatmap(data_corr, annot=True)

# Shuffling Dataset to Desquence any Ordered Data
dataset = shuffle(dataset)

#  A Function to Split Dataset to Train and Test Sets
def train_test_split(dataset):
    percent = 0.80
    data_train = dataset[ : int(percent * dataset.shape[0])]
    data_test = dataset[int(percent * dataset.shape[0]) : ]
    return data_train, data_test

# A Function to Split Passed Dataset to K-Folds.
def split_kfold(dataset, folds):
    l = dataset.shape[0]
    s = (1.0 / folds)
    splits = []
    for k in range(folds):
        split = dataset[int(k * s * l) : int((k + 1) * s * l)]
        splits.append(split)
    return splits

# A Function to Obtain the Features and Target List
def get_X_Y(dataset):
    X = np.asarray([[row['area'], row['perimeter'], 
                     row['compactness'], row['kernel_length'], 
                     row['kernel_width'], row['asym'], 
                     row['groove_length']] for index, row in dataset.iterrows()])
    Y = np.asarray([row['type'] for index, row in dataset.iterrows()])
    return X, Y

# Spliting the Dataset
# Initial Split of Dataset to Train and Test
data_cv, data_test = train_test_split(dataset)
# Obtaining Cross Validation Splits for CV Dataset
cv_splits = split_kfold(data_cv, 5)
# Obtaining Train Features and Target
cv_train_X, cv_train_Y = get_X_Y(data_cv)
# Obtaining Test Features and Target
test_X, test_Y = get_X_Y(data_test)

# Linear Kernel Ridge Regression
print ()
# Functions and Parameters
alphas = [1.0]
hyperparams = {'alpha' : alphas}
lin_krr = KernelRidge(alpha=alphas, kernel='linear')
# Linear KRR Initializer
lin_krr_clf = GridSearchCV(lin_krr, hyperparams, cv=5)

# Train
tl = time.time()
lin_krr_fit = lin_krr_clf.fit(cv_train_X, cv_train_Y)
tl = time.time() - tl
print ('Time Taken To Train : %f Secs.' % tl)
lin_krr_res = lin_krr_clf.cv_results_
print ('Linear KRR CV Results : \n', lin_krr_res)
lin_krr_params = lin_krr_clf.best_params_
print ('Linear KRR Best Parameters : \n', lin_krr_params)
lin_krr_score = lin_krr_clf.best_score_
print ('Linear KRR Best Score : \n', lin_krr_score)
print ()

# Test and Results
tl = time.time()
test_pred = lin_krr_clf.predict(test_X)
tl = time.time() - tl
print ('Time Taken To Test : %f Secs.' % tl)
# Round Test Predictions to Avoid Multiclass Continuous Targets Error
test_pred = np.round(test_pred)
# The Best Estimator
print ('Test Estimator : \n', lin_krr_clf.best_estimator_)
# Recall Score
recall = r_s(test_Y, test_pred, average='micro')
print ('Recall Score : \n', recall)
# Precision Score
precision = p_s(test_Y, test_pred, average='micro')
print ('Precision Score : \n', precision)
# Accuracy Score
accuracy = a_s(test_Y, test_pred)
print ('Accuracy Score : \n', accuracy)
# Confusion Matix
print('Confusion Matrix : \n', c_m(test_Y, test_pred))
print ()

# Polynomial Kernel Ridge Regression
print ()
# Functions and Parameters
alphas = [1.0]
degs = [2.0, 3.0, 4.0] # M
hyperparams = {'alpha' : alphas, 'degree' : degs}
poly_krr = KernelRidge(kernel='poly', alpha=alphas, degree=degs, gamma=1, coef0=1)
# Polynomial KRR Initializer
poly_krr_clf = GridSearchCV(poly_krr, hyperparams, cv=5)

# Train
tp = time.time()
poly_krr_fit = poly_krr_clf.fit(cv_train_X, cv_train_Y)
tp = time.time() - tp
print ('Time Taken To Train : %f Secs.' % tp)
poly_krr_res = poly_krr_clf.cv_results_
print ('Polynomial KRR CV Results : \n', poly_krr_res)
poly_krr_params = poly_krr_clf.best_params_
print ('Polynomial KRR Best Parameters : \n', poly_krr_params)
poly_krr_score = poly_krr_clf.best_score_
print ('Polynomial KRR Best Score : \n', poly_krr_score)
print ()

# Test and Results
tp = time.time()
test_pred = poly_krr_clf.predict(test_X)
tp = time.time() - tp
print ('Time Taken To Test : %f Secs.' % tp)
# The Best Estimator
print ('Test Estimator : \n', poly_krr_clf.best_estimator_)
# Recall Score
recall = r_s(test_Y, test_pred)
print ('Recall Score : \n', recall)
# Precision Score
precision = p_s(test_Y, test_pred)
print ('Precision Score : \n', precision)
# Accuracy Score
accuracy = a_s(test_Y, test_pred)
print ('Accuracy Score : \n', accuracy)
# Confusion Matix
print ('Confusion Matrix : \n', c_m(test_Y, test_pred))
print ()

# Gaussian/RBF Kernel Ridge Regression
print ()
# Functions and Parameters
alphas = [1.0]
sigmas = [0.1, 0.5, 1.0, 2.0, 4.0]
hyperparams = {'alpha' : alphas, 'gamma' : sigmas}
rbf_krr = KernelRidge(kernel='rbf', alpha=alphas, gamma=sigmas)
# Gaussian/RBF KRR Initializer
rbf_krr_clf = GridSearchCV(rbf_krr, hyperparams, cv=5)

# Train
tg = time.time()
rbf_krr_fit = rbf_krr_clf.fit(cv_train_X, cv_train_Y)
tg = time.time() - tg
print ('Time Taken To Train : %f Secs.' % tg)
rbf_krr_res = rbf_krr_clf.cv_results_
print ('Gaussian KRR CV Results : \n', rbf_krr_res)
rbf_krr_params = rbf_krr_clf.best_params_
print ('Gaussian KRR Best Parameters : \n', rbf_krr_params)
rbf_krr_score = rbf_krr_clf.best_score_
print ('Gaussian KRR Best Score : \n', rbf_krr_score)
print ()

# Test and Results
tg = time.time()
test_pred = rbf_krr_clf.predict(test_X)
tg = time.time() - tg
print ('Time Taken To Test : %f Secs.' % tg)
# The Best Estimator
print ('Test Estimator : \n', rbf_krr_clf.best_estimator_)
# Recall Score
recall = r_s(test_Y, test_pred)
print ('Recall Score : \n', recall)
# Precision Score
precision = p_s(test_Y, test_pred)
print ('Precision Score : \n', precision)
# Accuracy Score
accuracy = a_s(test_Y, test_pred)
print ('Accuracy Score : \n', accuracy)
# Confusion Matix
print ('Confusion Matrix : \n', c_m(test_Y, test_pred))
print ()

# Logistic Regression
# However Dataset is Not Large in this Case, 'sag' is used as better solver.
lr = LogisticRegression(multi_class='multinomial', solver='sag') # LogR Initializer

# Train
tlr = time.time()
print ()
recs, precs, accs = [], [], []
for i in range(len(cv_splits)):
    print ('CV Epoch : ' + str(i + 1))
    cv_train, cv_test = train_test_split(cv_splits[i])
    cv_train_X, cv_train_Y = get_X_Y(cv_train)
    cv_test_X, cv_test_Y = get_X_Y(cv_test)
    lr_fit = lr.fit(cv_train_X, cv_train_Y)
    cv_pred = lr.predict(cv_test_X)
    # The Coefficients
    print ('Coefficients : \n', lr.coef_)
    # Recall Score
    recall = r_s(cv_test_Y, cv_pred)
    print ('Recall Score : \n', recall)
    # Precision Score
    precision = p_s(cv_test_Y, cv_pred)
    print ('Precision Score : \n', precision)
    # Accuracy Score
    accuracy = a_s(cv_test_Y, cv_pred)
    print ('Accuracy Score : \n', accuracy)
    # Conusion Matix
    print ('Confusion Matrix : \n', c_m(cv_test_Y, cv_pred))
    recs.append(recall)
    precs.append(precision)
    accs.append(accuracy)
    print ()
tlr = time.time() - tlr
print ('Time Taken To Train : %f Secs.' % tlr)

# Calculation of Average Metrics after CV
print('Average Recall Score : %f' % np.mean(recs))
print('Average Precision Score : %f' % np.mean(precs))
print('Average Accuracy Score : %f' % np.mean(accs))
print ()

# Test and Results
tlr = time.time()
test_pred = lr.predict(test_X)
tlr = time.time() - tlr
print ('Time Taken To Test : %f Secs.' % tlr)
# The Coefficients
print('Test Coefficients : \n', lr.coef_)
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

# Linear Support Vector Classification
print ()
# Functions and Parameters
cs = [100000.0]
lin_svc = SVC(C=cs, kernel='linear', cache_size=4096)
hyperparams = {'C' : cs}
# Linear SVC Initializer
lin_svc_clf = GridSearchCV(lin_svc, hyperparams, cv=5)

# Train
tl = time.time()
lin_svc_fit = lin_svc_clf.fit(cv_train_X, cv_train_Y)
tl = time.time() - tl
print ('Time Taken To Train : %f Secs.' % tl)
lin_svc_res = lin_svc_clf.cv_results_
print ('Linear SVC CV Results : \n', lin_svc_res)
lin_svc_params = lin_svc_clf.best_params_
print ('Linear SVC Best Parameters : \n', lin_svc_params)
lin_svc_score = lin_svc_clf.best_score_
print ('Linear SVC Best Score : \n', lin_svc_score)
print ()

# Test and Results
tl = time.time()
test_pred = lin_svc_clf.predict(test_X)
tl = time.time() - tl
print ('Time Taken To Test : %f Secs.' % tl)
# The Best Estimator
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
# Confusion Matix
print('Confusion Matrix : \n', c_m(test_Y, test_pred))
print ()

# Polynomial Support Vector Classification
print ()
# Functions and Parameters
cs = [100000.0]
degs = [2.0, 3.0, 4.0] # M
poly_svc = SVC(kernel='poly', C=cs, degree=degs, gamma=1, coef0=1, cache_size=4096)
hyperparams = {'C' : cs, 'degree' : degs}
# Polynomial SVC Initializer
poly_svc_clf = GridSearchCV(poly_svc, hyperparams, cv=5)

# Train
tp = time.time()
poly_svc_fit = poly_svc_clf.fit(cv_train_X, cv_train_Y)
tp = time.time() - tp
print ('Time Taken To Train : %f Secs.' % tp)
poly_svc_res = poly_svc_clf.cv_results_
print ('Polynomial SVC CV Results : \n', poly_svc_res)
poly_svc_params = poly_svc_clf.best_params_
print ('Polynomial SVC Best Parameters : \n', poly_svc_params)
poly_svc_score = poly_svc_clf.best_score_
print ('Polynomial SVC Best Score : \n', poly_svc_score)
print ()

# Test and Results
tp = time.time()
test_pred = poly_svc_clf.predict(test_X)
tp = time.time() - tp
print ('Time Taken To Test : %f Secs.' % tp)
# The Best Estimator
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
# Confusion Matix
print('Confusion Matrix : \n', c_m(test_Y, test_pred))
print ()

# Gaussian/RBF Support Vector Classification
print ()
# Functions and Parameters
cs = [100000.0]
sigmas = [0.1, 0.5, 1.0, 2.0, 4.0]
hyperparams = {'C' : cs, 'gamma' : sigmas}
rbf_svc = SVC(kernel='rbf', C=cs, gamma=sigmas, cache_size=4096)
# Gaussian/RBF SVC Initializer
rbf_svc_clf = GridSearchCV(rbf_svc, hyperparams, cv=5)

# Train
tg = time.time()
rbf_svc_fit = rbf_svc_clf.fit(cv_train_X, cv_train_Y)
tg = time.time() - tg
print ('Time Taken To Train : %f Secs.' % tg)
rbf_svc_res = rbf_svc_clf.cv_results_
print ('Gaussian/RBF SVC CV Results : \n', rbf_svc_res)
rbf_svc_params = rbf_svc_clf.best_params_
print ('Gaussian/RBF SVC Best Parameters : \n', rbf_svc_params)
rbf_svc_score = rbf_svc_clf.best_score_
print ('Gaussian/RBF SVC Best Score : \n', rbf_svc_score)
print ()

# Test and Results
tg = time.time()
test_pred = rbf_svc_clf.predict(test_X)
tg = time.time() - tg
print ('Time Taken To Test : %f Secs.' % tg)
# The Best Estimator
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
# Confusion Matix
print('Confusion Matrix : \n', c_m(test_Y, test_pred))
print ()

# Linear Support Vector Classification
print ()
# Functions and Parameters
cs = [0.1, 0.5, 1.0, 2.0, 5.0]
lin_svc = SVC(C=cs, kernel='linear', cache_size=4096)
hyperparams = {'C' : cs}
# Linear SVC Initializer
lin_svc_clf = GridSearchCV(lin_svc, hyperparams, cv=5)

# Train
tl = time.time()
lin_svc_fit = lin_svc_clf.fit(cv_train_X, cv_train_Y)
tl = time.time() - tl
print ('Time Taken To Train : %f Secs.' % tl)
lin_svc_res = lin_svc_clf.cv_results_
print ('Linear SVC CV Results : \n', lin_svc_res)
lin_svc_params = lin_svc_clf.best_params_
print ('Linear SVC Best Parameters : \n', lin_svc_params)
lin_svc_score = lin_svc_clf.best_score_
print ('Linear SVC Best Score : \n', lin_svc_score)
print ()

# Test and Results
tl = time.time()
test_pred = lin_svc_clf.predict(test_X)
tl = time.time() - tl
print ('Time Taken To Test : %f Secs.' % tl)
# The Best Estimator
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
# Confusion Matix
print('Confusion Matrix : \n', c_m(test_Y, test_pred))
print ()

# Polynomial Support Vector Classification
print ()
# Functions and Parameters
cs = [0.1, 0.5, 1.0, 2.0, 5.0]
degs = [2.0, 3.0, 4.0] # M
poly_svc = SVC(kernel='poly', C=cs, degree=degs, gamma=1, coef0=1, cache_size=4096)
hyperparams = {'C' : cs, 'degree' : degs}
# Polynomial SVC Initializer
poly_svc_clf = GridSearchCV(poly_svc, hyperparams, cv=5)

# Train
tp = time.time()
poly_svc_fit = poly_svc_clf.fit(cv_train_X, cv_train_Y)
tp = time.time() - tp
print ('Time Taken To Train : %f Secs.' % tp)
poly_svc_res = poly_svc_clf.cv_results_
print ('Polynomial SVC CV Results : \n', poly_svc_res)
poly_svc_params = poly_svc_clf.best_params_
print ('Polynomial SVC Best Parameters : \n', poly_svc_params)
poly_svc_score = poly_svc_clf.best_score_
print ('Polynomial SVC Best Score : \n', poly_svc_score)
print ()

# Test and Results
tp = time.time()
test_pred = poly_svc_clf.predict(test_X)
tp = time.time() - tp
print ('Time Taken To Test : %f Secs.' % tp)
# The Best Estimator
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
# Confusion Matix
print('Confusion Matrix : \n', c_m(test_Y, test_pred))
print ()

# Gaussian/RBF Support Vector Classification
print ()
# Functions and Parameters
cs = [0.1, 0.5, 1.0, 2.0, 5.0]
sigmas = [0.1, 0.5, 1.0, 2.0, 4.0]
hyperparams = {'C' : cs, 'gamma' : sigmas}
rbf_svc = SVC(kernel='rbf', C=cs, gamma=sigmas, cache_size=4096)
# Gaussian/RBF SVC Initializer
rbf_svc_clf = GridSearchCV(rbf_svc, hyperparams, cv=5)

# Train
tg = time.time()
rbf_svc_fit = rbf_svc_clf.fit(cv_train_X, cv_train_Y)
tg = time.time() - tg
print ('Time Taken To Train : %f Secs.' % tg)
rbf_svc_res = rbf_svc_clf.cv_results_
print ('Gaussian/RBF SVC CV Results : \n', rbf_svc_res)
rbf_svc_params = rbf_svc_clf.best_params_
print ('Gaussian/RBF SVC Best Parameters : \n', rbf_svc_params)
rbf_svc_score = rbf_svc_clf.best_score_
print ('Gaussian/RBF SVC Best Score : \n', rbf_svc_score)
print ()

# Test and Results
tg = time.time()
test_pred = rbf_svc_clf.predict(test_X)
tg = time.time() - tg
print ('Time Taken To Test : %f Secs.' % tg)
# The Best Estimator
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
# Confusion Matix
print('Confusion Matrix : \n', c_m(test_Y, test_pred))
print ()

