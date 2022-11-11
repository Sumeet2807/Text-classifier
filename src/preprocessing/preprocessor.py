import preprocessing.blocks as blocks


class Text_preprocessor_from_dict():
    def __init__(self,preprocess_dict):

        core_words = preprocess_dict['core-words']
        remove_garbage = preprocess_dict['remove-garbage-words']
        remove_numbers = preprocess_dict['remove-numbers']
        remove_stopwords = preprocess_dict['remove-stopwords']
        lemmatize = preprocess_dict['lemmatize']
        force_lower_case = preprocess_dict['force-lower-case']

        pipeline = []

        if force_lower_case:
            pipeline.append(blocks.convert_lowercase)

        pipeline.append(blocks.Text_reducer(core_words,remove_garbage,remove_numbers))
        pipeline.append(blocks.Remove_stop_words_lemmatize(remove_stopwords, lemmatize))

        self.pipeline = pipeline

    def __call__(self,x):
        for block in self.pipeline:
            x = block(str(x))

        return x

    
