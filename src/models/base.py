import time, yaml, pickle, os

class Model():


    def fit(self,X,y):
        raise NotImplementedError

    def predict(self,X):
        raise NotImplementedError
    
    def predict_proba(self,X):
        raise NotImplementedError

    def load(self,config_filename):
        model_filename = config_filename[:-4] + '.pkl'
        with open(model_filename, 'rb') as pkl_file:
            model = pickle.load(pkl_file)
        return model

    def save(self,dir,config):
        tstamp = str(int(time.time()))
        pkl_filename = os.path.join(dir,tstamp + '.pkl')
        yaml_filename = os.path.join(dir,tstamp + '.yml')
        with open(pkl_filename, 'wb') as pkl_file:
            pickle.dump(self,pkl_file)
        with open(yaml_filename, 'w+') as yaml_file:
            yaml.dump(config, yaml_file, allow_unicode=True, default_flow_style=False)