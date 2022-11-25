from data_io.base import Base_file
import pandas as pd



class Delimited_file(Base_file):

    def read(self, args):
        sep = args['seperator']
        if sep == 'tab':
            sep = '\t'

        self.dataframe = pd.read_csv(args['filepath'], sep=sep,encoding='latin')
        return self.dataframe

    def write(self,dataframe,args):
        
        sep = args['seperator']
        if sep == 'tab':
            sep = '\t'

        dataframe.to_csv(args['filepath'],sep=sep,index=False)






# class Delimited_file(Base_file):

#     def __init__(self, dataframe):
#         self.dataframe = dataframe


#     def read(self,text_column, args, category_column=None,text_preprocessing=None,shuffle=True):
#         sep = args['seperator']
#         if sep == 'tab':
#             sep = '\t'

#         df = pd.read_csv(args['filepath'], sep=sep,encoding='latin')
#         if shuffle or (args['fraction']<1):
#             df = df.sample(frac=args['fraction'])
#         if args['remove-null-text']:
#             df = df[~df[text_column].isnull()]        
#         y = None
#         if category_column:
#             df = df[~df[category_column].isnull()]
#             y = df[category_column].to_numpy()

#         X = df[text_column]
#         if text_preprocessing is not None:
#             X = X.apply(text_preprocessing)  

#         return X.to_numpy(), y

#     def write(self,X, y ,text_column, category_column,args):

#         df = pd.DataFrame([X,y],columns=[text_column,category_column])
#         df.to_csv(args['filepath'],sep=args['sep'])
