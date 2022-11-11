from sklearn.feature_extraction.text import CountVectorizer as CountVectorizer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
import numpy as np


def get_model_instance(model_dict):
    name = model_dict['class']
    name = 'models.' + name
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)

    return mod(model_dict['args'])


# class DenseTransformer(TransformerMixin):

#     def fit(self, X, y=None, **fit_params):
#         return self

#     def transform(self, X, y=None, **fit_params):
#         return X.todense()


# class BOW_ensemble():
#     def __init__(self,classifier,vectorizer,scale=True,estimator_grps=5,cv_folds=5):

#         self.vectorizer = vectorizer
#         self.clf = classifier
#         pipes = [('vectorizer', self.vectorizer)]
#         if scale:
#             pipes.extend([('sparse_to_dense', DenseTransformer()),('scaler',StandardScaler())])
#         pipes.append(('estimator', self.clf))
        
#         self.pipeline = Pipeline(pipes)
#         self.cv_folds = cv_folds
#         self.estimator_grps = estimator_grps
#         self.estimators = None

#     def fit(self,X,y):

#         precision = []
#         recall = []
#         pipelines = []
#         for i in tqdm(range(self.estimator_grps)):
#             np.random.shuffle(X)
#             scores = cross_validate(self.pipeline, X, y, cv = self.cv_folds,scoring=['precision_macro','recall_macro'],return_estimator=True)
#             pipelines.extend(scores['estimator'])
#             precision.append(np.mean(scores['test_precision_macro']))
#             recall.append(np.mean(scores['test_recall_macro']))
        
#         self.estimators = pipelines
#         return np.mean(precision), np.mean(recall)
#         # return pipelines, precision, recall



#     def predict(self,X):
#         y_ensemble = []
#         for est in self.estimators:
#             y_ensemble.append(est.predict_proba(X)[:,1])

#         y_ensemble = np.mean(np.array(y_ensemble),axis=0)
#         predictions = (y_ensemble > 0.5).astype(int)
#         return predictions
    
#     def predict_proba(self,X):
#         y_ensemble = []
#         for est in self.estimators:
#             y_ensemble.append(est.predict_proba(X)[:,1])

#         y_ensemble = np.mean(np.array(y_ensemble),axis=0)
#         return y_ensemble
