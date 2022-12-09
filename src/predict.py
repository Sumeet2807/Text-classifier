from preprocessing.utils import get_preprocessor_class
from models.utils import get_model_class
from data_io.utils import get_datahandler_class
from utils import get_data_from_config
import yaml
from sklearn.metrics import classification_report as cr
import sys
import os

if len(sys.argv) < 2:
    raise Exception('A configuration yaml needs to supplied as argument')

with open(str(sys.argv[1]), "r") as f:
    config = yaml.safe_load(f)

data_config = config['data']

yaml_filename = os.path.join(config['model']['path'],'specs' + '.yml')
with open(yaml_filename, "r") as f:
    model_yml = yaml.safe_load(f)



preprocess_config = model_yml['preprocessing']
model_config = model_yml['model']
preprocessor = get_preprocessor_class(preprocess_config['class'])(preprocess_config['args'])
print("\n\n****** Loading model - %s ******" % model_config['class'])

model = get_model_class(model_config['class'])(model_config['args']).load(model_config['saved-object'])
print("\n\n****** Loading data from source******")

df,X,_ = get_data_from_config(data_config['read'],preprocessor)
print('Loaded %s datapoints' %len(X))

if data_config['write']['null-text-dedicated-class'] is not None:
    X_complete = X
    y_complete = np.array([data_config['write']['null-text-dedicated-class']]*len(X))
    non_null_indices = ~(X_complete=='')
    X = X_complete[non_null_indices]


print("\n\n****** Predicting******")
if not len(X):
    raise Exception('No datapoint supplied to predict. Check preprocessing and dedicated class for null text in the config')
y_pred = model.predict(X)

if data_config['write']['null-text-dedicated-class'] is not None:
    X = X_complete
    y_complete[non_null_indices] = y_pred
    y_pred = y_complete

if 'write' in config['data']:
    print("\n\n****** Writing predictions to destination******")

    df[data_config['write']['label-column']] = y_pred
    get_datahandler_class(data_config['write']['class'])().write(df,data_config['write']['args'])
