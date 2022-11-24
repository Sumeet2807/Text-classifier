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
print("\n\n****** Loading model ******")

model = get_model_class(model_config['class'])(model_config['args']).load(model_config['saved-object'])
print("\n\n****** Loading data from source******")

df,X,_ = get_data_from_config(data_config['read'])

print("\n\n****** Predicting ******")
y_pred = model.predict(X)

if 'write' in config['data']:
    print("\n\n****** Writing predictions to destination******")

    df[data_config['write']['label-column']] = y_pred
    get_datahandler_class(data_config['write']['class'])().write(df,data_config['write']['args'])