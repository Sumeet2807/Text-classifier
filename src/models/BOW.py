import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer as CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import nltk
from sklearn.linear_model import LogisticRegression as LR
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
import numpy as np
import re
from models.model_utils import BOW_ensemble



class Linear_BOW_ensemble():
    def __init__(self,vectorizer_type='count', vectorizer_max_features=1000, vectorizer_ngrams=(1,1),
                scale=True,estimator_grps=10,cv_folds=10,
                lr_penalty = 'none',lr_solver = 'lbfgs', lr_max_iter=1000):
        if vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(max_features=vectorizer_max_features,ngram_range=vectorizer_ngrams)
        else:
            self.vectorizer = TfidfVectorizer(max_features=vectorizer_max_features,ngram_range=vectorizer_ngrams)

        self.clf = LR(penalty=lr_penalty, max_iter=lr_max_iter, solver=lr_solver)
        self.ensemble = BOW_ensemble(self.clf,self.vectorizer)

    def fit(self,X,y):
        return self.ensemble.fit(X,y)

    def predict(self,X):
        return self.ensemble.predict(X)
    
    def predict_proba(self,X):
        return self.ensemble.predict_proba(X)




# #Seeds and hyperparameters
# # np.random.seed(42)
# kfold = 10
# LR_MAX_ITER = 10000
# NGRAM_TUPLE = (1,1)
# MAX_FEATURES = 100000
# TEST_TRAIN_RATIO = 0.1
# LR_PENALTY = 'none'
# LR_SOLVER = 'lbfgs'
# scale=True

# # train_file = 
# precision = []
# recall = []
# predictions = []

# def reduce_text(s): 
#     red_text = ''   
#     s = re.sub('\d*\.?\d+', ' num ', s)
#     sents = re.split(';|\.', s.lower())
#     for sent in sents:
#         sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
#         if re.findall(r'(rebate)+', sent):
#             # sent = sent.replace("rebate", "*****REBATE*****")
#             red_text = '\n'.join([red_text, sent])
#     return red_text

# def text_preprocessing(x):
#     x = reduce_text(x)
#     x = x.replace('#num#', 'num') # tokenizer seperates # from words, so need to clean
#     lemmatizer = WordNetLemmatizer()
#     stop_words = set(stopwords.words('english'))
#     word_tokens = word_tokenize(str(x))
#     filtered_sentence = [lemmatizer.lemmatize(w) for w in word_tokens if not w.lower() in stop_words]
#     # filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
#     # filtered_sentence = [lemmatizer.lemmatize(w) for w in word_tokens]

#     return ' '.join(filtered_sentence).lower()

# for i in range(20):

#     df = pd.read_csv(train_file, sep=sep, encoding='latin').sample(frac=1) #shuffle
#     # df = pd.read_csv(train_file, sep=sep, encoding='latin').iloc[indices].sample(frac=1) #shufflea 
#     # print(len(df))
#     df = df[~df[text_col].isnull()]
#     # sns.countplot(df[class_col])

#     #preprocessing and bag of words vectorizing 
#     df[text_col] = df[text_col].apply(text_preprocessing)
#     vectorizer = TfidfVectorizer(max_features=MAX_FEATURES,ngram_range=NGRAM_TUPLE)
#     # vectorizer = CountVectorizer(max_features=MAX_FEATURES,ngram_range=NGRAM_TUPLE)
#     X_train = vectorizer.fit_transform(df[text_col].iloc[int(TEST_TRAIN_RATIO*len(df)):])
#     X_test = vectorizer.transform(df[text_col].iloc[:int(TEST_TRAIN_RATIO*len(df))])
#     y_test = df[class_col].iloc[:int(TEST_TRAIN_RATIO*len(df))].to_numpy()
#     y_train = df[class_col].iloc[int(TEST_TRAIN_RATIO*len(df)):].to_numpy()
#     # print(X_train.shape)

#     #Scaler
#     if scale:
#         scaler = StandardScaler()
#         X_train = scaler.fit_transform(X_train.toarray())
#         X_test = scaler.transform(X_test.toarray())

#     clf = LR(penalty=LR_PENALTY, max_iter=LR_MAX_ITER, solver=LR_SOLVER)


#     pipeline = Pipeline([('estimator', clf)])
#     #('scaler', scaler),
#     scores = cross_validate(pipeline, X_train, y_train, cv = kfold,scoring=['precision','recall'],return_estimator=True)
#     precision.append(np.mean(scores['test_precision']))
#     recall.append(np.mean(scores['test_recall']))




#     #holdout test
#     df_hold = pd.read_csv(hold_file, sep=hold_sep, encoding='latin')#.sample(frac=1) #shuffle 
#     # print(df_hold.columns)
#     df_hold = df_hold[~df_hold[hold_text_col].isnull()]
#     # df_hold['Small text'] = df_hold['sentence']
#     processed_text = df_hold[hold_text_col].apply(text_preprocessing)


#     X_hold = vectorizer.transform(processed_text).toarray()
#     if scale:
#         X_hold = scaler.transform(X_hold)

#     y_ensemble = []
#     for est in scores['estimator']:
# ########
#         # indices = np.argsort(np.abs(est[0].coef_))[0][-1:-10:-1]
#         # for i in indices:
#         #     for k, v in vectorizer.vocabulary_.items():
#         #         if v == i:
#         #             print(k, est[0].coef_[0][i])
#         # print('********\n\n')
# ########
#         # y_ensemble.append(est.predict(X_hold))
#         y_ensemble.append(est.predict_proba(X_hold)[:,1])
#     y_ensemble = (np.mean(np.array(y_ensemble),axis=0) > 0.5).astype(int)
#     # print(y_ensemble)
#     # print(np.sum(y_ensemble)/y_ensemble.shape[0])
#     # for i in np.argwhere(y_ensemble == 1).flatten():
#     #     print(df_hold.iloc[i]['Small text'])
#     #     print('\n ******************** \n')

#     predictions.append(y_ensemble)

# print('mean precision - ', np.mean(precision))
# print('mean recall - ', np.mean(recall))
# print(np.mean(predictions,axis=0))
# print(np.std(predictions,axis=0))
