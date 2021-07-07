
"""
imdb dataset parser.
"""
import os
import random
from itertools import chain
import json

from mindspore import Tensor

import numpy as np
import gensim

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk.stem
remove = str.maketrans('','',string.punctuation) 
wnl = WordNetLemmatizer()

class DatasetParser():
    """
    parse aclImdb data to features and labels.
    sentence->tokenized->encoded->padding->features
    """
    def __init__(self, data_path, glove_path):
        self.__segs = ['train', 'test']
        self.__data_path = data_path



    def parse(self):

        for seg in self.__segs:
            print('preprocessing',seg)

            sentence_file = os.path.join(self.__data_path, seg+'.txt')
            with open(sentence_file, mode='r', encoding='utf8') as f:
                data_lists = f.readlines()

            features = []
            labels = []
            for line in data_lists:
                labels.append(int(line[0]))

                lower = line[2:].lower() #小写 # strip()
                without_punctuation = lower.translate(remove)
                tokens = nltk.word_tokenize(without_punctuation) # 去标点
                without_stopwords = [w for w in tokens if not w in stopwords.words('english')] # 去停用词
                tokenized_sentence = [wnl.lemmatize(ws) for ws in without_stopwords] # 词形还原
                features.append(tokenized_sentence)

            datadict = {
                'lines': features,
                'labels': labels
            }

            os.makedirs(os.path.join(self.__data_path, 'processed'), exist_ok=True)
            with open(os.path.join(self.__data_path, 'processed',seg+'.json'),'w') as f:
                json.dump(datadict,f)








