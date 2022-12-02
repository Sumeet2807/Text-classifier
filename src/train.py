from preprocessing.utils import get_preprocessor_class
from models.utils import get_model_class
import yaml
from sklearn.metrics import classification_report as cr
from utils import get_data_from_config
import os
import time
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

print('Loaded %s datapoints' %len(X_train))
print('\n\n****** Training model - %s ******' % model_config['class'])
model.fit(X_train,y_train)
model.report_metrics()

if 'test' in data_config:

    test_data_config = data_config['test']
    _,X_test,y_test = get_data_from_config(test_data_config,preprocessor)

    print('\n\n****** Testing model ******')
    print('Loaded %s datapoints to test' %len(X_test))
    y_pred = model.predict(X_test)
    print(cr(y_test,y_pred, zero_division=0))

if model_config['save-dir'] is not None:
    print('\n\n****** Saving model ******')
    save_folder = os.path.join(model_config['save-dir'],str(int(time.time())))
    os.makedirs(save_folder,exist_ok=True)
    save_object_name = os.path.join(save_folder,'model')
    save_object_name = model.save(save_object_name)
    model_config['saved-object'] = save_object_name
    yaml_filename = os.path.join(save_folder,'specs' + '.yml')
    with open(yaml_filename, 'w+') as yaml_file:
        yaml.dump(config, yaml_file, allow_unicode=True, default_flow_style=False)
    print('model saved at - %s' % save_folder)
