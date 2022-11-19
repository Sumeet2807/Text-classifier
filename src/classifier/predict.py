import pandas as pd
from sklearn.preprocessing import LabelEncoder
from preprocessing.preprocessor import Text_preprocessor_from_dict
from models.model_utils import get_model_class
from data_io.utils import get_datahandler_class
from classifier.utils import get_data_from_config
import yaml
from sklearn.metrics import classification_report as cr


with open("config_predict.yml", "r") as f:
    config = yaml.safe_load(f)
config

read_data_config = config['data']['read']

with open(config['model']['yml-path'], "r") as f:
    model_yml = yaml.safe_load(f)



preprocess_config = model_yml['preprocessing']
model_config = model_yml['model']
preprocessor = Text_preprocessor_from_dict(preprocess_config)
print("\n\n****** Loading model ******")

model = get_model_class(model_config['class'])(model_config['args']).load(config['model']['yml-path'])
print("\n\n****** Loading data from source******")

df,X,_ = get_data_from_config(read_data_config)

print("\n\n****** Predicting ******")
y_pred = model.predict(X)

if 'write' in config['data']:
    print("\n\n****** Writing predictions to destination******")

    write_data_config = config['data']['write']
    df[write_data_config['label-column']] = y_pred
    get_datahandler_class(write_data_config['source-class'])().write(df,write_data_config['args'])
