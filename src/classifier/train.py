import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.io import get_datahandler_instance
from preprocessing.preprocessor import Text_preprocessor_from_dict
from models.model_utils import get_model_instance
import yaml
from sklearn.metrics import classification_report as cr


with open("config.yml", "r") as f:
    config = yaml.safe_load(f)
config


preprocess_config = config['preprocessing']
model_config = config['model']
data_config = config['data']
preprocessor = Text_preprocessor_from_dict(preprocess_config)

model = get_model_instance(model_config)
train_data_handler = get_datahandler_instance(data_config['train'])
print("\n\n****** Loading training and test data ******")
X_train,y_train = train_data_handler.read(data_config['train']['text-column'],data_config['train']['args'],
                                        data_config['train']['label-column'],text_preprocessing=preprocessor)




print('\n\n****** Training model - %s ******' % model_config['class'])
model.fit(X_train,y_train)
model.report_metrics()

if 'test' in data_config:
    test_data_handler = get_datahandler_instance(data_config['train'])

    X_test,y_test = test_data_handler.read(data_config['test']['text-column'],data_config['test']['args'],
                                            data_config['test']['label-column'],text_preprocessing=preprocessor)
    print('\n\n****** Testing model ******')
    y_pred = model.predict(X_test)
    print(cr(y_test,y_pred, zero_division=0))

print('\n\n****** Saving model ******')
model.save(model_config['save-dir'],config)
