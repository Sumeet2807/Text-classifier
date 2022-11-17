import pandas as pd



def csv_reader(file, text_column, category_column=None, sep=',',frac=1,text_preprocessing=None, shuffle=False,header=None):
    df = pd.read_csv(file, sep=sep,header=header)
    if shuffle or (frac<1):
        df = df.sample(frac=frac)
    X = df[text_column]
    if text_preprocessing:
        X = X.apply(text_preprocessing).to_numpy()
    y = None
    if category_column:
        y = df[category_column].to_numpy()

    return X, y

def csv_writer(file, df, sep=','):
    df.to_csv(file,sep=sep)



class Delimited_file():
    def read(self,text_column, args, category_column=None,text_preprocessing=None,shuffle=True):
        sep = args['seperator']
        if sep == 'tab':
            sep = '\t'

        df = pd.read_csv(args['filepath'], sep=sep,encoding='latin')
        if shuffle or (args['fraction']<1):
            df = df.sample(frac=args['fraction'])
        if args['remove-null-text']:
            df = df[~df[text_column].isnull()]        
        y = None
        if category_column:
            df = df[~df[category_column].isnull()]
            y = df[category_column].to_numpy()

        X = df[text_column]
        if text_preprocessing is not None:
            X = X.apply(text_preprocessing)  

        return X.to_numpy(), y

    def write(self,X, y ,text_column, category_column,args):

        df = pd.DataFrame([X,y],columns=[text_column,category_column])
        df.to_csv(args['filepath'],sep=args['sep'])



def get_datahandler_class(name):
    name = 'utils.' + name
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)

    return mod
