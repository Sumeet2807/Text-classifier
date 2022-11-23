from preprocessing.utils import get_preprocessor_class
from models.utils import get_model_class
import yaml
from sklearn.metrics import classification_report as cr
from utils import get_data_from_config
import os
import sys

if len(sys.argv) < 2:
    raise Exception('A configuration yaml needs to supplied as argument')

with open(str(sys.argv[1]), "r") as f:
    config = yaml.safe_load(f)


preprocess_config = config['preprocessing']
model_config = config['model']
data_config = config['data']

preprocessor = get_preprocessor_class(preprocess_config['class'])(preprocess_config['args'])
model = get_model_class(model_config['class'])(model_config['args'])


print("\n\n****** Loading training data ******")

_,X_train,y_train = get_data_from_config(data_config['train'],preprocessor)


print('\n\n****** Training model - %s ******' % model_config['class'])
model.fit(X_train,y_train)
model.report_metrics()

if 'test' in data_config:

    test_data_config = data_config['test']
    _,X_test,y_test = get_data_from_config(test_data_config,preprocessor)

    print('\n\n****** Testing model ******')
    y_pred = model.predict(X_test)
    print(cr(y_test,y_pred, zero_division=0))

print('\n\n****** Saving model ******')

if not os.path.exists(model_config['save-dir']):
    os.makedirs(model_config['save-dir'])

model.save(model_config['save-dir'],config)
