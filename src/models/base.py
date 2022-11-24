import time, yaml, pickle, os

class Model():

    def __init__(self, params):
        pass

    def fit(self,X,y):
        raise NotImplementedError

    def predict(self,X):
        raise NotImplementedError
    
    def predict_proba(self,X):
        raise NotImplementedError

    def load(self,object_name):
        with open(object_name, 'rb') as pkl_file:
            model = pickle.load(pkl_file)
        return model

    def save(self,object_name):        
        with open(object_name, 'wb') as pkl_file:
            pickle.dump(self,pkl_file)
        return object_name

    def report_metrics(self):
        pass

