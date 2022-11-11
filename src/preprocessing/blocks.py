from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer as CountVectorizer
from nltk.stem import WordNetLemmatizer
import nltk
import re

nltk.download('wordnet')

class Text_reducer():
    def __init__(self,core_words=None,remove_garbage=True,remove_numbers=True):

        if core_words is not None:
            self.core_words = core_words
        else:
            self.core_words = []
        self.remove_garbage = remove_garbage
        self.remove_numbers = remove_numbers
    def __call__(self,x):   
        if len(self.core_words):
            match_pattern = r'(\b' + re.escape(self.core_words[0])
            for i in range(1,len(self.core_words)):
                match_pattern += (r'\b|\b' + re.escape(self.core_words[i]))
            match_pattern += r'\b)'
            sents = re.split(';|\.|\n', x)
            x=''
            for sent in sents:
                if re.findall(match_pattern, sent): 
                    if self.remove_garbage:           
                        sent = re.sub('[^A-Za-z0-9]+', ' ', sent) #remove garbage
                    if self.remove_numbers:
                        sent = re.sub('\d*\.?\d+', ' num ', sent) #remove numbers
                    x = '\n'.join([x, sent])
        else:
            if self.remove_garbage:           
                x = re.sub('[^A-Za-z0-9]+', ' ', x) #remove garbage
            if self.remove_numbers:
                x = re.sub('\d*\.?\d+', ' num ', x) #remove numbers
            
        return x

class Remove_stop_words_lemmatize():
    def __init__(self,force_lower_case=False,remove_stop_words=True, lemmatize=True):
        self.remove_stop_words = remove_stop_words
        self.lemmatize = lemmatize
        self.force_lower_case = force_lower_case
    def __call__(self,x):
        # x = str(x)
        # if self.force_lower_case:
        #     x = x.lower()
        # if self.reduce_text:
        #     x = self.reduce_text(x)
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


            x = ' '.join(filtered_sentence)

        return x


def convert_lowercase(x):
    return x.lower()


