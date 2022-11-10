import pandas as pd


def csv_reader(file, text_column, category_column=None, sep=',',text_preprocessing=None, shuffle=False,header=None):
    df = pd.read_csv(file, sep=sep,header=header)
    if shuffle:
        df = df.sample(frac=1)
    X = df[text_column]
    if text_preprocessing:
        X = X.apply(text_preprocessing).to_numpy()
    y = None
    if category_column:
        y = df[category_column].to_numpy()

    return X, y

def csv_writer(file, df, sep=','):
    df.to_csv(file,sep=sep)