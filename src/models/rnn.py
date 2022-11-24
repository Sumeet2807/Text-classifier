import numpy as np
import tensorflow as tf
from models.base import Model
from sklearn.preprocessing import LabelEncoder
import os
import pickle

def bilstm_classifier(encoder, embedding_size=32,hidden_architecture=[32],dense_architecture=[64],num_classes=2,dropout_rate=0.2):
#input is a tensor of strings
#hidden_architecture - tuple of number of LSTM layers in sequence
#dense_architecture - tuple of number of dense layers in sequence
    model = tf.keras.Sequential([
            encoder,
            tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=embedding_size,
            # Use masking to handle the variable sequence lengths
            mask_zero=True)])
        

    for layer_size in hidden_architecture[:-1]:
        
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(layer_size,
                                                                 return_sequences=True)))
        tf.keras.layers.Dropout(dropout_rate)
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_architecture[-1])))
    tf.keras.layers.Dropout(dropout_rate)

    for layer_size in dense_architecture:        
        model.add(tf.keras.layers.Dense(layer_size,activation='relu'))
        tf.keras.layers.Dropout(dropout_rate)
        
    model.add(tf.keras.layers.Dense(num_classes))

    return(model)


class BLSTM(Model):
    def __init__(self, args):

        self.max_vocab_size = args['vectorizer-max-vocab-size']
        self.epochs = args['max-epochs']
        self.patience = args['early-stop-patience']
        self.harch = args['recurrent-architecture']
        self.darch = args['dense-architecture']
        self.emb_size = args['embedding-size']
        self.learning_rate  = args['learning-rate']
        self.dropout  = args['dropout-rate']
        self.batch_size = args['batch-size']
        self.validation_train_ratio = args['validation-to-train-ratio']
        self.label_encoder = LabelEncoder()

        self.net = None




    def fit(self,X,y):

        y = self.label_encoder.fit_transform(y)
        num_classes = len(np.unique(y))

        indices = np.arange(0,len(X))
        np.random.shuffle(indices)

        X_train = X[indices[int(self.validation_train_ratio * X.shape[0]):]]
        y_train = y[indices[int(self.validation_train_ratio * y.shape[0]):]]
        X_val = X[indices[:int(self.validation_train_ratio * X.shape[0])]]
        y_val = y[indices[:int(self.validation_train_ratio * y.shape[0])]]

        
        dset1 = tf.data.Dataset.from_tensor_slices(X_train)
        dset2 = tf.data.Dataset.from_tensor_slices(y_train.astype(int))
        dset_train = tf.data.Dataset.zip((dset1, dset2)).batch(self.batch_size)

        dset1 = tf.data.Dataset.from_tensor_slices(X_val)
        dset2 = tf.data.Dataset.from_tensor_slices(y_val.astype(int))
        dset_val = tf.data.Dataset.zip((dset1, dset2)).batch(self.batch_size)

        vectorizer = tf.keras.layers.TextVectorization(
                            max_tokens = self.max_vocab_size)
        vectorizer.adapt(dset_train.map(lambda text, label: text))

        self.net = bilstm_classifier(encoder=vectorizer,
                    embedding_size=self.emb_size,hidden_architecture=self.harch,
                        dense_architecture=self.darch,num_classes=num_classes,dropout_rate=self.dropout)
        # class_weight = None
        # if balan
        # class_weight = {0: 1.,
        #                 1: (len(y_train) - np.sum(y_train))/np.sum(y_train)}
        earlystop = tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True)
        self.net.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                    metrics=['accuracy'])
        history = self.net.fit(dset_train, epochs=self.epochs, 
                        callbacks=[earlystop],
                        # class_weight=class_weight,
                        validation_data=dset_val)
        self.history = dict(history.history)


    def predict(self,X):
        dset = tf.data.Dataset.from_tensor_slices(X).batch(self.batch_size)
        y_pred = tf.math.argmax(self.net.predict(dset),axis=1)
        return self.label_encoder.inverse_transform(y_pred)

    
    def predict_proba(self,X):
        dset = tf.data.Dataset.from_tensor_slices(X).batch(self.batch_size)
        y_pred = self.net.predict(dset)
        return y_pred

    def report_metrics(self):
        for key in list(self.history.keys()):
            print(str(key) + ' - ' + str(self.history[key][-1]))


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

