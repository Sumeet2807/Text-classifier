from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time
import pickle
import yaml
import os
from models.base import Model




class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.toarray()


class BOW_ensemble():
    def __init__(self,classifier,vectorizer,scale=True,estimator_grps=5,cv_folds=5):

        self.vectorizer = vectorizer
        self.clf = classifier
        pipes = [('vectorizer', self.vectorizer)]
        if scale:
            pipes.extend([('sparse_to_dense', DenseTransformer()),('scaler',StandardScaler())])
        pipes.append(('estimator', self.clf))
        
        self.pipeline = Pipeline(pipes)
        self.cv_folds = cv_folds
        self.estimator_grps = estimator_grps
        self.estimators = None

    def fit(self,X,y):

        precision = []
        recall = []
        pipelines = []
        for i in tqdm(range(self.estimator_grps)):
            indices = np.arange(0,len(X))
            np.random.shuffle(indices)
            scores = cross_validate(self.pipeline, X[indices], y[indices], cv = self.cv_folds,scoring=['precision_macro','recall_macro'],return_estimator=True)
            pipelines.extend(scores['estimator'])
            precision.append(np.mean(scores['test_precision_macro']))
            recall.append(np.mean(scores['test_recall_macro']))
        
        self.estimators = pipelines
        return np.mean(precision), np.mean(recall)
        # return pipelines, precision, recall



    def predict(self,X):
        y_ensemble = []
        for est in self.estimators:
            y_ensemble.append(est.predict_proba(X)[:,1])

        y_ensemble = np.mean(np.array(y_ensemble),axis=0)
        predictions = (y_ensemble > 0.5).astype(int)
        return predictions
    
    def predict_proba(self,X):
        y_ensemble = []
        for est in self.estimators:
            y_ensemble.append(est.predict_proba(X)[:,1])

        y_ensemble = np.mean(np.array(y_ensemble),axis=0)
        return y_ensemble



class Linear_ensemble(Model):
    def __init__(self,params):

        if params['vectorizer-type'] == 'count':
            vectorizer = CountVectorizer(max_features=params['vectorizer-max-features'],ngram_range=params['vectorizer-ngrams'])
        elif params['vectorizer-type'] == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=params['vectorizer-max-features'],ngram_range=params['vectorizer-ngrams'])
        else:
            raise Exception('Unknown vectorizer type - %s' % params['vectorizer-type'])

        clf = LR(penalty=params['clf-penalty'], max_iter=params['clf-max-iter'], solver=params['clf-solver'])
        self.ensemble = BOW_ensemble(clf,vectorizer,params['scale-inputs'],params['ensemble-groups'],params['ensemble-folds'])
        self.label_encoder = LabelEncoder()
        self.precision = None
        self.recall = None


    def fit(self,X,y):
        y = self.label_encoder.fit_transform(y)
        self.precision, self.recall =  self.ensemble.fit(X,y)

    def predict(self,X):
        y_pred = self.ensemble.predict(X)
        return self.label_encoder.inverse_transform(y_pred)
    
    def predict_proba(self,X):
        return self.ensemble.predict_proba(X)

    def report_metrics(self):
        print('training precision - %s\ntraining recall - %s' % (self.precision,self.recall))







class Linear_ensemble_sgd(Model):
    def __init__(self,params):

        if params['vectorizer-type'] == 'count':
            vectorizer = CountVectorizer(max_features=params['vectorizer-max-features'],ngram_range=params['vectorizer-ngrams'])
        elif params['vectorizer-type'] == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=params['vectorizer-max-features'],ngram_range=params['vectorizer-ngrams'])
        else:
            raise Exception('Unknown vectorizer type - %s' % params['vectorizer-type'])

        clf = SGD(penalty=params['clf-penalty'], max_iter=params['clf-max-iter'], 
                early_stopping=params['clf-early-stop'], loss=params['clf-loss'])
        self.ensemble = BOW_ensemble(clf,vectorizer,params['scale-inputs'],params['ensemble-groups'],params['ensemble-folds'])
        self.label_encoder = LabelEncoder()
        self.OneHotEncoder = OneHotEncoder()


    def fit(self,X,y):   
        y = self.label_encoder.fit_transform(y)
        self.precision, self.recall =  self.ensemble.fit(X,y)

    def predict(self,X):
        y_pred = self.ensemble.predict(X)
        return self.OneHotEncoder.inverse_transform(y_pred)
    
    def predict_proba(self,X):
        return self.ensemble.predict_proba(X)

    def report_metrics(self):
        print('training precision - %s\ntraining recall - %s' % (self.precision,self.recall))



    


