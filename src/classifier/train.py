import pandas as pd
from sklearn.preprocessing import LabelEncoder
from preprocessing.preprocessor import Text_preprocessor_from_dict
from models.model_utils import get_model_class
import yaml
from sklearn.metrics import classification_report as cr
from classifier.utils import get_data_from_config


with open("config.yml", "r") as f:
    config = yaml.safe_load(f)



preprocess_config = config['preprocessing']
model_config = config['model']
train_data_config = config['data']['train']
preprocessor = Text_preprocessor_from_dict(preprocess_config)

model = get_model_class(model_config['class'])(model_config['args'])


print("\n\n****** Loading training and test data ******")

# train_data_handler = get_datahandler_class(train_data_config['source-class'])()
# X_train,y_train = train_data_handler.read(train_data_config['text-column'],train_data_config['args'],
#                                         train_data_config['label-column'],text_preprocessing=preprocessor)

_,X_train,y_train = get_data_from_config(train_data_config,preprocessor)


print('\n\n****** Training model - %s ******' % model_config['class'])
model.fit(X_train,y_train)
model.report_metrics()

if 'test' in config['data']:
    test_data_config = config['data']['test']
    # test_data_handler = get_datahandler_class(test_data_config['source-class'])()
    # X_test,y_test = test_data_handler.read(test_data_config['text-column'],test_data_config['args'],
    #                                         test_data_config['label-column'],text_preprocessing=preprocessor)
    _,X_test,y_test = get_data_from_config(test_data_config,preprocessor)

    print('\n\n****** Testing model ******')
    y_pred = model.predict(X_test)
    print(cr(y_test,y_pred, zero_division=0))

print('\n\n****** Saving model ******')
model.save(model_config['save-dir'],config)
