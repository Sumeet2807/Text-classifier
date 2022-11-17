from sklearn.feature_extraction.text import CountVectorizer as CountVectorizer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
import numpy as np


def get_model_class(name):
    name = 'models.' + name
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
