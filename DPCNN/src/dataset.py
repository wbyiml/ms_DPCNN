
import os
import json
import random

import numpy as np

import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter

from .datasetParser import DatasetParser


import gensim
from nltk.corpus import wordnet as wn
# import gensim.downloader as gensimapi
# print(list(gensim.downloader.info()['models'].keys()))      # 在gensim-data中显示所有可用的模型
# https://github.com/RaRe-Technologies/gensim-data



class RTDataset:
    def __init__(self, data_path, glove_path,seq_len,embed_size, is_train=True):
        self.is_train = is_train
        self.seq_len = seq_len
        self.embed_size = embed_size

        if os.path.isfile(os.path.join(data_path, 'processed/train.json')) and \
           os.path.isfile(os.path.join(data_path, 'processed/test.json')) and \
           os.path.isfile(os.path.join(data_path, 'processed/vocab.json')) and \
           os.path.isfile(os.path.join(data_path, 'processed', 'weight_'+str(embed_size)+'d.txt')):
            print('datasets already processed.')
        else:
            parser = DatasetParser(data_path, glove_path, embed_size)
            parser.parse()


        # print('loading gensim wvmodel')
        # glove_file = os.path.join(glove_path, 'glove.6B.'+str(self.embed_size)+'d.txt')
        # self.__wvmodel = gensim.models.KeyedVectors.load_word2vec_format(glove_file)
        
        # print('loading glove twitter')
        # glove_twitter_file = os.path.join(glove_path,'glove-twitter-25')
        # if os.path.isfile(glove_twitter_file):
        #     print('a')
        #     self.twittermodel = gensim.models.KeyedVectors.load_word2vec_format(glove_twitter_file) 
        #     print('aa')
        # else:
        #     self.twittermodel = gensimapi.load('glove-twitter-25') 


        
        if self.is_train:
            with open(os.path.join(data_path, 'processed','train.json'),'r') as f:
                datadict = json.load(f)
        else:
            with open(os.path.join(data_path, 'processed','test.json'),'r') as f:
                datadict = json.load(f)
        self.datas = datadict['lines']
        self.labels = datadict['labels']

        with open(os.path.join(data_path, 'processed','vocab.json'),'r') as f:
            self.vocab = json.load(f)
   

    def __getitem__(self, index):
        """
        Args:
            index, int: Index.

        Returns:
            image, PIL.Image: Image of the given index.
            target, str: target of the given index.
        """
        label = self.labels[index]

        sentence = self.datas[index]


        if len(sentence) > self.seq_len:
            ids = sorted(random.sample( range(len(sentence)), self.seq_len ))
            cut_sentence = [sentence[i] for i in ids]
        else:
            cut_sentence = sentence


        sentence_ids = np.zeros(self.seq_len, dtype=np.int32 )
        for i,word in enumerate(cut_sentence):
            # if self.is_train:
            #     if random.random() < 0.4:
            #         syn  = wn.synsets(word)
            #         if len(syn)!=0:
            #             word = random.choice( syn[0].lemma_names() )   # pre

            #     if random.random() < 0.4:
            #         try:
            #             # word = random.choice(self.twittermodel.most_similar(word, topn=5))[0]
            #             word = random.choice(self.__wvmodel.most_similar(word, topn=5))[0]           
            #         except:
            #             pass
            
            # word 2 vec
            sentence_ids[i] = self.vocab.get(word, 0)

        # if self.is_train:
        #     if random.random() < 0.4: # 乱序
        #         length = len(sentence) if len(sentence) < self.seq_len else self.seq_len
        #         swapids = random.sample( range(length), random.choice(range(length+1)) )
        #         sentence_ids[sorted(swapids)] = sentence_ids[swapids]
            
        #     if random.random() < 0.4: # cutout
        #         length = len(sentence) if len(sentence) < self.seq_len else self.seq_len
        #         cutids = random.sample( range(length), random.choice(range(int(length/4)+1)) )
        #         sentence_ids[cutids] = 0
        


        
        return sentence_ids, label

    def __len__(self):
        """Length of the dataset.

        Returns:
            length, int: Length of the dataset.
        """
        return len(self.datas)



def create_dataset(batch_size, data_path, glove_path,seq_len,embed_size, is_train=True):
    ds.config.set_seed(1)

    dataset_generator = RTDataset(data_path, glove_path,seq_len,embed_size, is_train)

    dataset = ds.GeneratorDataset(dataset_generator, ["sentence", "label"], shuffle=True)

    dataset = dataset.shuffle(buffer_size=dataset.get_dataset_size())
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.repeat(count=1)

    return dataset, len(dataset_generator.vocab)
