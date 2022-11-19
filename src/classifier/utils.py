from data_io.utils import get_datahandler_class



def get_data_from_config(data_config, preprocessor=None):
    df = get_datahandler_class(data_config['source-class'])().read(data_config['args'])
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
        X_train = X_train.apply(preprocessor)
     

    return df, X.to_numpy(), y