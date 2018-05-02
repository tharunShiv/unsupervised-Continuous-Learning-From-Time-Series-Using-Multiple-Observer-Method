from __future__ import division, print_function

from os.path import expanduser

from sklearn.metrics import confusion_matrix, classification_report

from shapelets_lts.classification import LtsShapeletClassifier
from shapelets_lts.util import ucr_dataset_loader
import pandas as pd

"""
This example uses dataset from the UCR archive "UCR Time Series Classification
Archive" format.  

- Follow the instruction on the UCR page 
(http://www.cs.ucr.edu/~eamonn/time_series_data/) to download the dataset. You 
need to be patient! :) 
- Update the vars below to point to the correct dataset location in your  
machine.

Otherwise update _load_train_test_datasets() below to return your own dataset.
"""

'''
ucr_dataset_base_folder = expanduser(r'D:\MyLife\HandsOn\onlinetransactions')
ucr_dataset_name = 'OnlineRetail'
'''

'''
ucr_dataset_base_folder = expanduser(r'D:\MyLife\HandsOn\Shapelets\shaplets-python\UCR_TS_Archive_2015')
ucr_dataset_name = 'Gun_Point'
'''

'''
def _load_train_test_datasets():
    """
    :return: numpy arrays, train_data, train_labels, test_data, test_labels
        train_data and test_data shape is: (n_samples, n_features)
        train_labels and test_labels shape is: (n_samples)
    """
    return ucr_dataset_loader.load_dataset(
        dataset_name=ucr_dataset_name,
        dataset_folder=ucr_dataset_base_folder
    )
    
    dataset = pd.read_csv('creditcard.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    return X_train, y_train, X_test, y_test
    
    return 0
'''

################################################################
    

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv(r'D:\MyLife\HandsOn\Shapelets\shaplets-python\MartProb\Train.csv')
test = pd.read_csv(r'D:\MyLife\HandsOn\Shapelets\shaplets-python\MartProb\Test.csv')

# preprocessing

### mean imputations 
train['Item_Weight'].fillna((train['Item_Weight'].mean()), inplace=True)
test['Item_Weight'].fillna((test['Item_Weight'].mean()), inplace=True)

### reducing fat content to only two categories 
train['Item_Fat_Content'] = train['Item_Fat_Content'].replace(['low fat','LF'], ['Low Fat','Low Fat']) 
train['Item_Fat_Content'] = train['Item_Fat_Content'].replace(['reg'], ['Regular']) 
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace(['low fat','LF'], ['Low Fat','Low Fat']) 
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace(['reg'], ['Regular'])

## for calculating establishment year
train['Outlet_Establishment_Year'] = 2018 - train['Outlet_Establishment_Year'] 
test['Outlet_Establishment_Year'] = 2018 - test['Outlet_Establishment_Year'] 

### missing values for size
train['Outlet_Size'].fillna('Small',inplace=True)
test['Outlet_Size'].fillna('Small',inplace=True)

### label encoding cate. var.
col = ['Outlet_Size','Outlet_Location_Type','Outlet_Type','Item_Fat_Content']
test['Item_Outlet_Sales'] = 0
combi = train.append(test)
number = LabelEncoder()
for i in col:
    combi[i] = number.fit_transform(combi[i].astype('str'))
    combi[i] = combi[i].astype('int')
train = combi[:train.shape[0]]
test = combi[train.shape[0]:]
test.drop('Item_Outlet_Sales',axis=1,inplace=True)

## removing id variables 
training = train.drop(['Outlet_Identifier','Item_Type','Item_Identifier'],axis=1)
testing = test.drop(['Outlet_Identifier','Item_Type','Item_Identifier'],axis=1)
y_train = training['Item_Outlet_Sales']
training.drop('Item_Outlet_Sales',axis=1,inplace=True)

features = training.columns
target = 'Item_Outlet_Sales'

X_train, X_test = training, testing

import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']



#Model exploration
from xgboost import XGBRegressor
from sklearn.linear_model import BayesianRidge, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
#from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


model_factory = [
    RandomForestRegressor(),
    XGBRegressor(nthread=1),
    #MLPRegressor(),
    Ridge(),
    BayesianRidge(),
    ExtraTreesRegressor(),
    ElasticNet(),
    KNeighborsRegressor(),
    GradientBoostingRegressor()
]


for model in model_factory:
    model.seed = 42
    num_folds = 3

    scores = cross_val_score(model, X_train, y_train, cv=num_folds, scoring='neg_mean_squared_error')
    score_description = " %0.2f (+/- %0.2f)" % (np.sqrt(scores.mean()*-1), scores.std() * 2)

    print('{model:25} CV-5 RMSE: {score}'.format(
        model=model.__class__.__name__,
        score=score_description
    ))


# XGBRegressor has teh best results, as it has the lowest RMSE



## normal submission using xgb
model = XGBRegressor()
model.fit(X_train,y_train)
pred = model.predict(X_test)

## saving file
sub = pd.DataFrame(data = pred, columns=['Item_Outlet_Sales'])
sub['Item_Identifier'] = test['Item_Identifier']
sub['Outlet_Identifier'] = test['Outlet_Identifier']
sub.to_csv('bigmart-xgb.csv', index='False')

cross_val_score(model, X_train, y_train, cv=num_folds, scoring='neg_mean_squared_error') # ,n_jobs = 8

#pseudo-labelling
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, RegressorMixin

class PseudoLabeler(BaseEstimator, RegressorMixin):
    '''
    Sci-kit learn wrapper for creating pseudo-lebeled estimators.
    '''
    
    def __init__(self, model, unlabled_data, features, target, sample_rate=0.2, seed=42):
        '''
        @sample_rate - percent of samples used as pseudo-labelled data
                       from the unlabled dataset
        '''
        assert sample_rate <= 1.0, 'Sample_rate should be between 0.0 and 1.0.'
        
        self.sample_rate = sample_rate
        self.seed = seed
        self.model = model
        self.model.seed = seed
        
        self.unlabled_data = unlabled_data
        self.features = features
        self.target = target
        
    def get_params(self, deep=True):
        return {
            "sample_rate": self.sample_rate,
            "seed": self.seed,
            "model": self.model,
            "unlabled_data": self.unlabled_data,
            "features": self.features,
            "target": self.target
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

        
    def fit(self, X, y):
        '''
        Fit the data using pseudo labeling.
        '''

        augemented_train = self.__create_augmented_train(X, y)
        self.model.fit(
            augemented_train[self.features],
            augemented_train[self.target]
        )
        
        return self


    def __create_augmented_train(self, X, y):
        '''
        Create and return the augmented_train set that consists
        of pseudo-labeled and labeled data.
        '''        
        num_of_samples = int(len(self.unlabled_data) * self.sample_rate)
        
        # Train the model and creat the pseudo-labels
        self.model.fit(X, y)
        pseudo_labels = self.model.predict(self.unlabled_data[self.features])
        
        # Add the pseudo-labels to the test set
        pseudo_data = self.unlabled_data.copy(deep=True)
        pseudo_data[self.target] = pseudo_labels
        
        # Take a subset of the test set with pseudo-labels and append in onto
        # the training set
        sampled_pseudo_data = pseudo_data.sample(n=num_of_samples)
        temp_train = pd.concat([X, y], axis=1)
        augemented_train = pd.concat([sampled_pseudo_data, temp_train])

        return shuffle(augemented_train)
        
    def predict(self, X):
        '''
        Returns the predicted values.
        '''
        return self.model.predict(X)
    
    def get_model_name(self):
        return self.model.__class__.__name__




#only with the pseudoLabeller    
model = PseudoLabeler(
    XGBRegressor(nthread=1),
    test,
    features,
    target,
    sample_rate = 0.2 #0.3
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

cross_val_score(model, X_train, y_train, cv=num_folds, scoring='neg_mean_squared_error') # , n_jobs=8    
# The results should be the best
# o/p: 
#array([-1209513.47901035, -1164221.21852411, -1161820.31905521,
#       -1155255.72783089, -1168333.90383921])


sub = pd.DataFrame(data = pred, columns=['Item_Outlet_Sales'])
sub['Item_Identifier'] = test['Item_Identifier']
sub['Outlet_Identifier'] = test['Outlet_Identifier']
sub.to_csv('pseudo-labelling.csv', index='False')

    
    
model_factory = [
    XGBRegressor(nthread=1),
    
    PseudoLabeler(
        XGBRegressor(nthread=1),
        test,
        features,
        target,
        sample_rate=0.2 #0.3
    ),
]

for model in model_factory:
    model.seed = 42
    num_folds = 2 #8
    
    scores = cross_val_score(model, X_train, y_train, cv=num_folds, scoring='neg_mean_squared_error') #n_jobs=8
    score_description = "MSE: %0.4f (+/- %0.4f)" % (np.sqrt(scores.mean()*-1), scores.std() * 2)

    print('{model:25} CV-{num_folds} {score_cv}'.format(
        model=model.__class__.__name__,
        num_folds=num_folds,
        score_cv=score_description
    ))

'''
O/P:
    XGBRegressor              CV-8 MSE: 1083.2088 (+/- 122498.1181)
    PseudoLabeler             CV-8 MSE: 1081.2880 (+/- 127579.5128)
'''



# Performance of the pseudo labeller at various other sample rates
sample_rates = np.linspace(0, 1, 10)

def pseudo_label_wrapper(model):
    return PseudoLabeler(model, test, features, target)

# List of all models to test
model_factory = [
    RandomForestRegressor(n_jobs=1),
    XGBRegressor(),
]

# Apply the PseudoLabeler class to each model
model_factory = map(pseudo_label_wrapper, model_factory)

# Train each model with different sample rates
results = {}
num_folds = 5

for model in model_factory:
    model_name = model.get_model_name()
    print('%s' % model_name)

    results[model_name] = list()
    for sample_rate in sample_rates:
        model.sample_rate = sample_rate
        
        # Calculate the CV-3 R2 score and store it
        scores = cross_val_score(model, X_train, y_train, cv=num_folds, scoring='neg_mean_squared_error') # n_jobs = 4
        results[model_name].append(np.sqrt(scores.mean()*-1))
        
        
#Plot the diff between the XGB and Pseudo at sampleRate vs RSME        
plt.figure(figsize=(16, 18))

i = 1
for model_name, performance in results.items():    
    plt.subplot(3, 3, i)
    i += 1
    
    plt.plot(sample_rates, performance)
    plt.title(model_name)
    plt.xlabel('sample_rate')
    plt.ylabel('RMSE')
    

plt.show()





################################################################






def _evaluate_LtcShapeletClassifier():
    # load the data
    '''
    train_data, train_labels, test_data, test_labels = (
        _load_train_test_datasets()
    )
    '''
    '''
    dataset = pd.read_csv('finanDis.csv')
    X = dataset.iloc[:, 3:].values
    y = dataset.iloc[:, 2].values
    from sklearn.cross_validation import train_test_split
    train_data, test_data, train_labels, test_labels  = train_test_split(X, y, test_size = 0.3, random_state = 0)
    train_labels = train_labels.reshape(2570,1)
    test_labels = test_labels.reshape(1102, 1)
    y = y.reshape(3672, 1)
    '''
    # Applying PCA
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components = 2)   # first run with the n_components=None , then after seeing the explained variance, choose the number
    # train_data = pca.fit_transform(train_data)
    # explained_variance = pca.explained_variance_ratio_
    
    
    #from sklearn.decomposition import PCA
    #pca = PCA(n_components = 2)   # first run with the n_components=None , then after seeing the explained variance, choose the number
    #test_data = pca.fit_transform(test_data)
    #explained_variance = pca.explained_variance_ratio_
    # create a classifier (the parameter values as per Table1 for the GunPoint
    # dataset ). 200 epochs yielded 0.99 accuracy
    
    
    
    
    
    
    train_data, test_data = X_train.iloc[:, :].values, X_test.iloc[:, :].values
    train_labels, test_labels = y_train, pred
    
    sample_rates = sample_rates.reshape(10, 1)
    scores = scores.reshape(5, 1)
    pred = pred.reshape(5681, 1)
    train_labels = train_labels.reshape(8523, 1)
    y_train = y_train.reshape(8523, 1)
    test_labels = test_labels.reshape(5681, 1)
    
    
    #Standardization - of X_train 3rd col and y_train
    from pandas import Series
    from sklearn.preprocessing import StandardScaler
    from math import sqrt
    
    # load the dataset and print the first 5 rows
    series = X_train.iloc[:, 3].values
    #series1 = X_test.iloc[:, 3].values
    
    #print(series.head())
    
    # prepare data for standardization
    #values = series.values
    series = series.reshape((len(series), 1))
    #series1 = series1.reshape((len(series1), 1))
    
    # prepare data for standardization
    # values = series.values
    #values = values.reshape((len(values), 1))
    # train the standardization
    scaler = StandardScaler()
    #scaler1 = StandardScaler()
    #scaler2 = StandardScaler()
    
    scaler = scaler.fit(series)
    #scaler1 = scaler1.fit(series1)
    #scaler2 = scaler2.fit(y_train)
    print('Train Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))
    
    # standardization the dataset and print the first 5 rows
    normalized = scaler.transform(series)
    for i in range(len(series)):
        train_data[i, 3] = normalized[i]
    # inverse transform and print the first 5 rows
    # inversed = scaler.inverse_transform(normalized)
    # for i in range(5):
    #	  print(inversed[i]) 
    
    '''FOr test data'''
    series1 = X_test.iloc[:, 3].values    
    series1 = series1.reshape((len(series1), 1))   
    scaler1 = StandardScaler()   
    scaler1 = scaler1.fit(series1)
    print('Train Mean: %f, StandardDeviation: %f' % (scaler1.mean_, sqrt(scaler1.var_)))
    normalized = scaler1.transform(series1)
    for i in range(len(series1)):
        test_data[i, 3] = normalized[i]
        
    '''FOr y_train data'''
    series2 = y_train  
    series2 = series2.reshape((len(series2), 1))   
    scaler2 = StandardScaler()   
    scaler2 = scaler2.fit(series2)
    print('Train Mean: %f, StandardDeviation: %f' % (scaler2.mean_, sqrt(scaler2.var_)))
    normalized = scaler2.transform(series2)
    for i in range(len(series2)):
        y_train[i, 0] = normalized[i]
        
    
    #y_train = y_train.reshape((len(y_train), 1))

    '''For pred  data'''
    series3 = pred  
    series3 = series3.reshape((len(series3), 1))   
    scaler3 = StandardScaler()   
    scaler3 = scaler3.fit(series3)
    print('Train Mean: %f, StandardDeviation: %f' % (scaler3.mean_, sqrt(scaler3.var_)))
    normalized = scaler3.transform(series3)
    for i in range(len(series3)):
        pred[i, 0] = normalized[i]
        

    test_labels = pred
    
    
    
    Q = train_data.shape[1]
    K = int(0.15 * Q)
    L_min = int(0.2 * Q)
    classifier = LtsShapeletClassifier(
        K=K,
        R=3,
        L_min=L_min,
        epocs=40,
        lamda=0.1,
        eta=0.1,
        shapelet_initialization='segments_centroids',
        plot_loss=True
    ) 
    ''' eta = 0.1 '''
    ''' epocs = 200'''
    ''' lambda = 0.01'''
    
    # train the classifier
    classifier.fit(train_data, train_labels)

    # evaluate on test data
    prediction = classifier.predict(test_data)
    print(classification_report(test_labels, prediction))
    print(confusion_matrix(test_labels, prediction))


if __name__ == '__main__':
    _evaluate_LtcShapeletClassifier()

'''
#Autoencoders
from numpy import genfromtxt
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score

def estimateGaussian(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.cov(dataset.T)
    return mu, sigma
    
def multivariateGaussian(dataset,mu,sigma):
    p = multivariate_normal(mean=mu, cov=sigma)
    return p.pdf(dataset)

def selectThresholdByCV(probs,gt):
    best_epsilon = 0
    best_f1 = 0
    f = 0
    stepsize = (max(probs) - min(probs)) / 1000;
    epsilons = np.arange(min(probs),max(probs),stepsize)
    for epsilon in np.nditer(epsilons):
        predictions = (probs < epsilon)
        f = f1_score(gt, predictions, average = "binary")
        if f > best_f1:
            best_f1 = f
            best_epsilon = epsilon
    return best_f1, best_epsilon

'''
'''
a_train = X_train
a_train[8] = y_train

a_test = X_test
a_test[8] = pred

data_a = a_train.append(a_test)

plt.figure()
plt.plot(range(len(data_a)), data_a.iloc[:, 8].values, 'bx')
plt.show()

tr_data = data_a.iloc[:, 8].values 
mu, sigma = estimateGaussian(tr_data)
p = multivariateGaussian(tr_data,mu,sigma)

p_cv = multivariateGaussian(tr_data,mu,sigma)
fscore, ep = selectThresholdByCV(p_cv,tr_data)
outliers = np.asarray(np.where(p < ep))

plt.figure() 
plt.xlabel("Latency (ms)") 
plt.ylabel("Throughput (mb/s)") 
plt.plot(range(len(re_data)),tr_data[:,0],"bx") 
plt.plot(tr_data[outliers,0],tr_data[outliers,1],"ro") 
plt.show()
'''


'''
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42
LABELS = ["Normal", "Anomaly"]
'''
