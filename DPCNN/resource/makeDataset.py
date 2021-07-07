import os
import random


# TODO：先文本预处理（大小写的处理，标点符号的处理，文本的分词，去除停用词，以及词干的提取），再划分数据集





train_percent = 0.9

posfile = open('rt-polaritydata/rt-polarity.pos','r', errors='ignore')
poslines = posfile.readlines()
numpos = len(poslines)
listpos = range(numpos)
numpostr = int(numpos * train_percent)
listpostr = random.sample(listpos, numpostr)

poslines_tr = ['1 '+poslines[i] for i in listpostr]
poslines_te = ['1 '+poslines[i] for i in (set(listpos)-set(listpostr))]


negfile = open('rt-polaritydata/rt-polarity.neg','r', errors='ignore')
neglines = negfile.readlines()
numneg = len(neglines)
listneg = range(numneg)
numnegtr = int(numneg * train_percent)
listnegtr = random.sample(listneg, numnegtr)

neglines_tr = ['0 '+neglines[i] for i in listnegtr]
neglines_te = ['0 '+neglines[i] for i in (set(listneg)-set(listnegtr))]


with open('rt-polaritydata/train.txt','w') as f:
    f.writelines(poslines_tr)
    f.writelines(neglines_tr)
    print(len(poslines_tr)+len(neglines_tr))
    
with open('rt-polaritydata/test.txt','w') as f:
    f.writelines(poslines_te)
    f.writelines(neglines_te)
    print(len(poslines_te)+len(neglines_te))
 

