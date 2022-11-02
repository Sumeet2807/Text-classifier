from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer as CountVectorizer
from nltk.stem import WordNetLemmatizer
import re

class Text_reducer():
    def __init__(self,core_words=[]):
        self.core_words = core_words
    def reduce(self,s): 
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

def text_preprocessing(x, reduce_text=None,remove_stop_words=True, lemmatize=True):
    if reduce_text:
        x = reduce_text(x)
    if remove_stop_words:        
        stop_words = set(stopwords.words('english'))
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
    
    if lemmatize or remove_stop_words:
        word_tokens = word_tokenize(str(x))

        if lemmatize and remove_stop_words:
            filtered_sentence = [lemmatizer.lemmatize(w) for w in word_tokens if not w.lower() in stop_words]
        elif lemmatize:
            filtered_sentence = [lemmatizer.lemmatize(w) for w in word_tokens]
        elif remove_stop_words:
            filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]


        x = ' '.join(filtered_sentence).lower()

    return x
