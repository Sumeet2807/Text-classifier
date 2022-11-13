from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time
import pickle
import yaml
import os
from models.base import Model




class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.toarray()


class BOW_ensemble():
    def __init__(self,classifier,vectorizer,scale=True,estimator_grps=5,cv_folds=5):

        self.vectorizer = vectorizer
        self.clf = classifier
        pipes = [('vectorizer', self.vectorizer)]
        if scale:
            pipes.extend([('sparse_to_dense', DenseTransformer()),('scaler',StandardScaler())])
        pipes.append(('estimator', self.clf))
        
        self.pipeline = Pipeline(pipes)
        self.cv_folds = cv_folds
        self.estimator_grps = estimator_grps
        self.estimators = None

    def fit(self,X,y):

        precision = []
        recall = []
        pipelines = []
        for i in tqdm(range(self.estimator_grps)):
            indices = np.arange(0,len(X))
            np.random.shuffle(indices)
            scores = cross_validate(self.pipeline, X[indices], y[indices], cv = self.cv_folds,scoring=['precision_macro','recall_macro'],return_estimator=True)
            pipelines.extend(scores['estimator'])
            precision.append(np.mean(scores['test_precision_macro']))
            recall.append(np.mean(scores['test_recall_macro']))
        
        self.estimators = pipelines
        return np.mean(precision), np.mean(recall)
        # return pipelines, precision, recall



    def predict(self,X):
        y_ensemble = []
        for est in self.estimators:
            y_ensemble.append(est.predict_proba(X)[:,1])

        y_ensemble = np.mean(np.array(y_ensemble),axis=0)
        predictions = (y_ensemble > 0.5).astype(int)
        return predictions
    
    def predict_proba(self,X):
        y_ensemble = []
        for est in self.estimators:
            y_ensemble.append(est.predict_proba(X)[:,1])

        y_ensemble = np.mean(np.array(y_ensemble),axis=0)
        return y_ensemble



class Linear_ensemble(Model):
    def __init__(self,params):

        if params['vectorizer-type'] == 'count':
            vectorizer = CountVectorizer(max_features=params['vectorizer-max-features'],ngram_range=params['vectorizer-ngrams'])
        elif params['vectorizer-type'] == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=params['vectorizer-max-features'],ngram_range=params['vectorizer-ngrams'])
        else:
            raise Exception('Unknown vectorizer type - %s' % params['vectorizer-type'])

        clf = LR(penalty=params['clf-penalty'], max_iter=params['clf-max-iter'], solver=params['clf-solver'])
        self.ensemble = BOW_ensemble(clf,vectorizer,params['scale-inputs'],params['ensemble-groups'],params['ensemble-folds'])
        self.label_encoder = LabelEncoder()


    def fit(self,X,y):
        y = self.label_encoder.fit_transform(y)
        return self.ensemble.fit(X,y)

    def predict(self,X):
        y_pred = self.ensemble.predict(X)
        return self.label_encoder.inverse_transform(y_pred)
    
    def predict_proba(self,X):
        return self.ensemble.predict_proba(X)





class Linear_ensemble_sgd():
    def __init__(self,params):

        if params['vectorizer-type'] == 'count':
            vectorizer = CountVectorizer(max_features=params['vectorizer-max-features'],ngram_range=params['vectorizer-ngrams'])
        elif params['vectorizer-type'] == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=params['vectorizer-max-features'],ngram_range=params['vectorizer-ngrams'])
        else:
            raise Exception('Unknown vectorizer type - %s' % params['vectorizer-type'])

        clf = SGD(penalty=params['clf-penalty'], max_iter=params['clf-max-iter'], 
                early_stopping=params['clf-early-stop'], loss=params['clf-loss'])
        self.ensemble = BOW_ensemble(clf,vectorizer,params['scale-inputs'],params['ensemble-groups'],params['ensemble-folds'])
        self.label_encoder = LabelEncoder()
        self.OneHotEncoder = OneHotEncoder()


    def fit(self,X,y):        
        y = self.OneHotEncoder.fit_transform(y[...,np.newaxis])
        print(y.shape)
        return self.ensemble.fit(X,y)

    def predict(self,X):
        y_pred = self.ensemble.predict(X)
        return self.OneHotEncoder.inverse_transform(y_pred)
    
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
