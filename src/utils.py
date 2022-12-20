from data_io.utils import get_datahandler_class
import numpy as np
import pandas as pd



def get_data_from_config(data_config, preprocessor=None):
    df = get_datahandler_class(data_config['class'])().read(data_config['args'])
    if data_config['shuffle'] or (data_config['fraction']<1):
        df = df.sample(frac=data_config['fraction'])
    if data_config['remove-null-text']:
        df = df[~df[data_config['text-column']].isnull()] 
    
    y = None
    if 'label-column' in data_config:
        df = df[~df[data_config['label-column']].isnull()]
        y = df[data_config['label-column']].to_numpy()
        
    X = df[data_config['text-column']]
    if preprocessor is not None:
        X = X.apply(preprocessor)
        
    X = X.to_numpy()

    if 'random-oversample' in data_config and data_config['random-oversample']:
        ros = RandomOverSampler(random_state=42)
        X,y = ros.fit_resample(X[...,np.newaxis], y)
        X = X[:,0]


    return df, X, y



def augment_dataset(df, text_col, class_col, samples_to_add, max_samples_to_combine):
    df_pos = df[df[class_col] == 1]
    df_neg = df[df[class_col] == 0]

    new_samples = []
    for i in np.random.randint(0,2,size=samples_to_add):
        text = ''       
        sents = np.random.randint(2,max_samples_to_combine+1)
        if i:
            pos_index = np.random.randint(len(df_pos))
            text = df_pos.iloc[pos_index][text_col]
            rest_indices = np.random.randint(len(df),size=sents-1)
            for index in rest_indices:
                text = text + '\n' + df.iloc[index][text_col]
        else: 
            indices = np.random.randint(len(df_neg),size=sents)
            for index in indices:
                text = text + '\n' + df_neg.iloc[index][text_col]

        new_samples.append([text,i])
    df_new = pd.DataFrame(new_samples,columns=[text_col,class_col])
    df_aug = pd.concat([df[[text_col,class_col]],df_new],axis=0)
    return df_aug
