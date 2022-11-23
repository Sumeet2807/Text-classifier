
train_file = '../data/data_only_rebate_terms_2021-1-1.csv'
text_col = 'sentence'
class_col = 'class'
sep = '\t'

# train_file = '../data/dataset_only_rebate_text_2021-1-1_v3.csv'
# text_col = 'sentence'
# class_col = 'class'
# sep = ','

hold_file = '../data/data_only_rebate_terms_greater_than_2021-1-1.csv'
hold_text_col = 'sentence'
hold_class_col = 'class'
hold_sep = '\t'

# hold_file = '../data/predictions_greater_than_2021-1-1_v2.csv'
# hold_text_col = 'Rebate text'
# hold_class_col = 'Predictions'
# hold_sep = ','

# hold_file = '../data/data_complete_rebate_after_2021-1-1.csv'
# hold_text_col = 'text'
# hold_class_col = 'Predictions'
# hold_sep = ','

# def reduce_text(s): 
#     red_text = ''   
#     s = re.sub('\d*\.?\d+', ' num ', s)
#     sents = re.split('\. ', s.lower())
#     for sent in sents:
#         if re.findall(r'(rebate)+', sent):
#             # sent = sent.replace("rebate", "*****REBATE*****")
#             red_text = '\n'.join([red_text, sent])
#     return red_text

def reduce_text(s): 
    red_text = ''   
    s = re.sub('\d*\.?\d+', ' num ', s)
    sents = re.split(';|\.|\n', s.lower())
    for sent in sents:
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        if re.findall(r'(rebate)+', sent):
            red_text = '\n'.join([red_text, sent])
    return red_text

def text_preprocessing(x):
    x = reduce_text(x)
    x = x.replace('#num#', 'num') # tokenizer seperates # from words, so need to clean
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(str(x))
    # filtered_sentence = [lemmatizer.lemmatize(w) for w in word_tokens if not w.lower() in stop_words]
    # filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    # filtered_sentence = [lemmatizer.lemmatize(w) for w in word_tokens]
    filtered_sentence = [w for w in word_tokens]

    return ' '.join(filtered_sentence).lower()


df = pd.read_csv(train_file, sep=sep, encoding='latin').sample(frac=1) #shuffle
# df = pd.read_csv(train_file, sep=sep, encoding='latin').iloc[indices].sample(frac=1) #shufflea 
# df = augment_dataset(df,text_col,class_col,10000,3)
print(len(df))
df = df[~df[text_col].isnull()]
# print(len(df))
# sns.countplot(df[class_col])

#preprocessing and bag of words vectorizing 
df[text_col] = df[text_col].apply(text_preprocessing)
print(df[text_col])

TEST_TRAIN_SPLIT = 0.05
VAL_TRAIN_SPLIT = 0.05
BATCH_SIZE = 16
df_test = df.iloc[:int(TEST_TRAIN_SPLIT*len(df))]
df_train = df.iloc[int(TEST_TRAIN_SPLIT*len(df)):]

df_val = df_train.iloc[:int(VAL_TRAIN_SPLIT*len(df_train))]
df_train = df_train.iloc[int(VAL_TRAIN_SPLIT*len(df_train)):]
df_train = augment_dataset(df_train,text_col,class_col,10000,3)
y_train = df_train[class_col].to_numpy()
y_test = df_test[class_col].to_numpy()
y_val = df_val[class_col].to_numpy()


print(len(df_train),len(df_val),len(df_test))

dset1 = tf.data.Dataset.from_tensor_slices(df_train[text_col].to_numpy())
dset2 = tf.data.Dataset.from_tensor_slices(df_train[class_col].to_numpy().astype(int))
dset_train = tf.data.Dataset.zip((dset1, dset2)).batch(BATCH_SIZE)

dset1 = tf.data.Dataset.from_tensor_slices(df_val[text_col].to_numpy())
dset2 = tf.data.Dataset.from_tensor_slices(df_val[class_col].to_numpy().astype(int))
dset_val = tf.data.Dataset.zip((dset1, dset2)).batch(BATCH_SIZE)

dset1 = tf.data.Dataset.from_tensor_slices(df_test[text_col].to_numpy())
dset2 = tf.data.Dataset.from_tensor_slices(df_test[class_col].to_numpy().astype(int))
dset_test = tf.data.Dataset.zip((dset1, dset2)).batch(BATCH_SIZE)

#hyperparameters
VOCAB_SIZE = None
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(dset_train.map(lambda text, label: text))

net = bilstm_classifier(inputs_as_embeddings=False, encoder=encoder,
                    embedding_size=8,hidden_architecture=[8],
                        dense_architecture=[8],num_classes=2,dropout_rate=0.3)
class_weight = {0: 1.,
                1: (len(y_train) - np.sum(y_train))/np.sum(y_train)}
earlystop = tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
net.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(5e-4),
              metrics=['accuracy'])
history = net.fit(dset_train, epochs=200, 
    callbacks=[earlystop],
    validation_data=dset_val)

y_pred = tf.math.argmax(net.predict(dset_test),axis=1)
print(cr(y_test,y_pred)) 

#holdout test
df_hold = pd.read_csv(hold_file, sep=hold_sep, encoding='latin')#.sample(frac=1) #shuffle 
# print(df_hold.columns)
df_hold = df_hold[~df_hold[hold_text_col].isnull()]
y_hold = df_hold[hold_class_col].to_numpy()
# df_hold['Small text'] = df_hold['sentence']
processed_text = df_hold[hold_text_col].apply(text_preprocessing).to_list()


y_pred_hold = tf.math.argmax(net(tf.constant(processed_text)),axis=1).numpy().astype(int)
print(cr(y_hold,y_pred_hold))
