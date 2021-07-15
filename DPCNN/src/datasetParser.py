
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
# from tqdm import tqdm

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk.stem
remove = str.maketrans('','',string.punctuation) 
wnl = WordNetLemmatizer()
from nltk.corpus import wordnet as wn

class DatasetParser():
    """
    parse aclImdb data to features and labels.
    sentence->tokenized->encoded->padding->features
    """
    def __init__(self, data_path, glove_path, embed_size):
        self.__segs = ['train', 'test']
        self.__data_path = data_path

        glove_file = os.path.join(glove_path, 'glove.6B.'+str(embed_size)+'d.txt')
        self.__wvmodel = gensim.models.KeyedVectors.load_word2vec_format(glove_file)
        self.embed_size = embed_size

        os.makedirs(os.path.join(self.__data_path, 'processed'), exist_ok=True)


    def parse(self):
        
        # vocabs = set()
        for seg in self.__segs:
            print('preprocessing',seg)

            sentence_file = os.path.join(self.__data_path, seg+'.txt')
            with open(sentence_file, mode='r', encoding='utf8') as f:
                data_lists = f.readlines()

            features = []
            labels = []
            vocab = []
            # pbar = tqdm(f)
            # for line in pbar:
            for line in data_lists:
                lower = line[2:].replace('\n',' ').lower() #小写 # strip()
                without_punctuation = lower.translate(remove)
                tokens = nltk.word_tokenize(without_punctuation) # 去标点
                without_stopwords = [w for w in tokens if not w in stopwords.words('english')] # 去停用词
                tokenized_sentence = [wnl.lemmatize(ws) for ws in without_stopwords] # 词形还原

                features.append(tokenized_sentence)
                labels.append(int(line[0]))



                if seg == 'train':
                    for word in tokenized_sentence:
                        syn  = wn.synsets(word)
                        if len(syn)!=0:
                            vocab.append(syn[0].lemma_names())


                        # try:
                        #     # word = random.choice(self.twittermodel.most_similar(word, topn=5))[0]
                        #     word = random.choice(self.__wvmodel.most_similar(word, topn=5))[0]           
                        # except:
                        #     pass

            if seg == 'train':
                vocab = set(chain(*features)) #| set(chain(*vocab))
                word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}  # 得到word的id才能 onehot、bow、embedding
                word_to_idx['<unk>'] = 0

                weight_np = np.zeros((len(word_to_idx), self.embed_size), dtype=np.float32)
                for word, idx in word_to_idx.items():
                    if word in self.__wvmodel:
                        word_vector = self.__wvmodel.get_vector(word)
                        weight_np[idx, :] = word_vector
                
                
                with open(os.path.join(self.__data_path, 'processed', 'vocab.json'),'w') as f:
                    json.dump(word_to_idx,f)
                np.savetxt(os.path.join(self.__data_path, 'processed', 'weight_'+str(self.embed_size)+'d.txt'), weight_np)
                # embedding_table = np.loadtxt(os.path.join(self.__data_path, 'processed', 'weight.txt')).astype(np.float32)

            

            datadict = {
                'lines': features,
                'labels': labels
            }


            with open(os.path.join(self.__data_path, 'processed',seg+'.json'),'w') as f:
                json.dump(datadict,f)








