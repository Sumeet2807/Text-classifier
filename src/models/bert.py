import numpy as np
import tensorflow as tf
from models.base import Model
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import pickle
from transformers import AutoTokenizer, TFAutoModelForMaskedLM
from datasets import Dataset
from transformers import DefaultDataCollator



def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer(examples['text'], padding="max_length", truncation=True)

def distilbert_classifier(hidden_size, output_classes, model, finetune_bert=False):
    distilbert_base_layer = model.layers[0]
    distilbert_base_layer.trainable = finetune_bert
    input_ids = tf.keras.Input(shape=(1), dtype=tf.int32)
    attention_mask = tf.keras.Input(shape=(1), dtype=tf.int32)
    input = {'input_ids':input_ids,
    'attention_mask':attention_mask        
    }
    x = distilbert_base_layer(input).last_hidden_state[:,0,:]
    x = tf.keras.layers.Dense(hidden_size,activation='gelu')(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(output_classes)(x)
    return tf.keras.Model(inputs=input, outputs=x, name="distilbert")


class Dense_top(Model):
    def __init__(self,args):

        self.epochs = args['max-epochs']
        self.patience = args['early-stop-patience']
        self.learning_rate  = args['learning-rate']
        self.batch_size = args['batch-size']
        self.validation_train_ratio = args['validation-to-train-ratio']
        self.label_encoder = LabelEncoder()
        self.dense_size = args['dense-top-size']
        self.finetune_bert = args['finetune-whole-bert']

        self.net = None


    def fit(self,X,y):

        
        y = self.label_encoder.fit_transform(y)
        num_classes = len(np.unique(y))

        
        df = pd.DataFrame(list(zip(X,y)),columns=['text', 'class'])
        df_val = df.iloc[:int(self.validation_train_ratio*len(df))]
        df_train = df.iloc[int(self.validation_train_ratio*len(df)):]

        dataset_train = Dataset.from_pandas(df_train.dropna()).map(tokenize_function, batched=True)
        dataset_val = Dataset.from_pandas(df_val.dropna()).map(tokenize_function, batched=True)
        

        data_collator = DefaultDataCollator(return_tensors="tf")

        tf_train_dataset = dataset_train.to_tf_dataset(
            columns=["attention_mask", "input_ids"],
            label_cols=["class"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=32,
        )

        tf_val_dataset = dataset_val.to_tf_dataset(
            columns=["attention_mask", "input_ids"],
            label_cols=["class"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=32,
        )

        distilbert_pretrained = TFAutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")

        self.net = distilbert_classifier(self.dense_size,num_classes,distilbert_pretrained,self.finetune_bert)

        earlystop = tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True)
        self.net.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                    metrics=['accuracy'])
        history = self.net.fit(tf_train_dataset, epochs=self.epochs, 
                        callbacks=[earlystop],
                        # class_weight=class_weight,
                        validation_data=tf_val_dataset)
        self.history = dict(history.history)



    def predict(self,X):

        df = pd.DataFrame(X,columns=['text'])

        dataset_eval = Dataset.from_pandas(df).map(tokenize_function, batched=True)

        data_collator = DefaultDataCollator(return_tensors="tf")
        tf_dataset_eval = dataset_eval.to_tf_dataset(
            columns=["attention_mask", "input_ids"],
            # label_cols=["class"],
            # shuffle=True,
            collate_fn=data_collator,
            batch_size=32,
        )
        y_pred = tf.math.argmax(self.net.predict(tf_dataset_eval),axis=1)
        return self.label_encoder.inverse_transform(y_pred)
    
    def predict_proba(self,X):
        

        df = pd.DataFrame(X,columns=['text'])

        dataset_eval = Dataset.from_pandas(df).map(tokenize_function, batched=True)

        data_collator = DefaultDataCollator(return_tensors="tf")
        tf_dataset_eval = dataset_eval.to_tf_dataset(
            columns=["attention_mask", "input_ids"],
            # label_cols=["class"],
            # shuffle=True,
            collate_fn=data_collator,
            batch_size=32,
        )
        y_pred = self.net.predict(tf_dataset_eval)
        return y_pred

    def load(self,object_name):
        net_path = os.path.join(object_name,'net')
        wrapper_path = os.path.join(object_name,'wrapper')
        with open(wrapper_path, 'rb') as pkl_file:
            model = pickle.load(pkl_file)
        model.net = tf.keras.models.load_model(net_path)
        return model

    def save(self,object_name): 
        if not os.path.exists(object_name):
            os.makedirs(object_name)
        net_path = os.path.join(object_name,'net')
        wrapper_path = os.path.join(object_name,'wrapper')
        tf.keras.models.save_model(self.net,net_path)
        temp_net = self.net
        self.net = None           
        with open(wrapper_path, 'wb') as pkl_file:
            pickle.dump(self,pkl_file)
        self.net = temp_net

        return object_name

    def report_metrics(self):
        for key in list(self.history.keys()):
            print(str(key) + ' - ' + str(self.history[key][-1]))

