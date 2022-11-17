from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer as CountVectorizer
from nltk.stem import WordNetLemmatizer
import re

class Text_reducer():
    def __init__(self,core_words=[]):
        self.core_words = core_words
    def __call__(self,s): 
        red_text = ''   
        if len(self.core_words):
            match_pattern = r'(' + re.escape(self.core_words[0])
            for i in range(1,len(self.core_words)):
                match_pattern += (r'|' + re.escape(self.core_words[i]))
            match_pattern += r')'
            sents = re.split(';|\.', s)
            for sent in sents:
                if re.findall(match_pattern, sent):            
                    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
                    sent = re.sub('\d*\.?\d+', ' num ', sent)
                    red_text = '\n'.join([red_text, sent])
        else:
            red_text = re.sub('[^A-Za-z0-9]+', ' ', s)
            red_text = re.sub('\d*\.?\d+', ' num ', s)
            
        return red_text

class Text_preprocessor():
    def __init__(self,force_lower_case=False,reduce_text=None,remove_stop_words=True, lemmatize=True):
        self.reduce_text = reduce_text
        self.remove_stop_words = remove_stop_words
        self.lemmatize = lemmatize
        self.force_lower_case = force_lower_case
    def __call__(self,x):
        x = str(x)
        if self.force_lower_case:
            x = x.lower()
        if self.reduce_text:
            x = self.reduce_text(x)
        if self.remove_stop_words:        
            stop_words = set(stopwords.words('english'))
        if self.lemmatize:
            lemmatizer = WordNetLemmatizer()
        
        if self.lemmatize or self.remove_stop_words:
            word_tokens = word_tokenize(str(x))

            if self.lemmatize and self.remove_stop_words:
                filtered_sentence = [lemmatizer.lemmatize(w) for w in word_tokens if not w.lower() in stop_words]
            elif self.lemmatize:
                filtered_sentence = [lemmatizer.lemmatize(w) for w in word_tokens]
            elif self.remove_stop_words:
                filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]


            x = ' '.join(filtered_sentence).lower()

        return x


reducer = Text_reducer(['rebate'])
preprocessor = Text_preprocessor(True,reducer,False,False)
preprocessor('ddf REBATE d df. fadf af saf and')



import pandas as pd

def csv_loader(file, text_column, category_column, sep=',',text_preprocessing=None, shuffle=False):
    df = pd.read_csv(file, sep=sep)
    if shuffle:
        df = df.sample(frac=1)
    X = df[text_column]
    if text_preprocessing:
        X = X.apply(text_preprocessing)
    y = df[category_column]

    return X.to_numpy(), y.to_numpy()

file = '../data/predictions_greater_than_2021-1-1_v2.csv'
text_column = 'Actual text'
category_column = 'Predictions'
X,y = csv_loader(file, text_column, category_column,text_preprocessing=preprocessor)
