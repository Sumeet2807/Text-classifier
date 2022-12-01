# Text-classifier
A modular yml based utility to help you classify text

## How to install?
Simply clone this repo or download the zip. Make sure you have the required python packages listed in the requirements.txt file installed.

## What can you use this utility for?
You may use this utility to train and save text classification models on a variety of text data.

## How to use?

There are two entrypoints that define the complete functionality of this utility.

### 1.) train.py

Use this python script to train a text classification with an appropriate preprocessing and model. All the information about the corpus sources, the model and the preprocessing required will be supplied to this script through a Yaml file.

example run command - python train.py <your_training_config_yml_file>



### 2.) predict.py

Use this python script to run a pretrained model on a corpus. All the information about corpus source and the saved model to use will supplied through a Yaml file.

example run command - python predict.py <your_evaluation_config_yml_file>


## Yaml file structure:-

1.) Training Yaml - 

Training Yaml file consists of 3 mandatory sections - Data, Preprocessing, Model. It contains all the information about the data source to use, the preprocessing steps to follow and finally the type of model to use with its relevant hyperparameters.

Please refer to examples/sample_train_config.yml for detailed information on how to create a yml.





