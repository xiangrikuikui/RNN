# -*- coding: utf-8 -*-

"""first, building training data matrices,the RNN"""






import csv
import itertools
import operator
import numpy as np
import nltk
import theano
import sys
import os
import time
from datetime import datetime
import i_utils
from i_run_theano import RNNTheano

GPU = True
if GPU:
    print "Trying to run under a GPU. If this is not desired, then modify " +\
          "network3.py to set the GPU flag to False."
    try:
        theano.config.device = 'gpu'
    except:
        pass  # it's already set
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU. If this is not desired, then the modify " +\
          "network3.py to set the GPU flag to True."
         
            
_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))         
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '80'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '5'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '100'))
_MODEL_FILE = os.environ.get('MODEL_FILE')



"""Now that we are able to calculate the gradients for our parameters we can implement SGD. I like to do 
this in two steps: 1. A function sdg_step that calculates the gradients and performs the updates for one 
batch. 2. An outer loop that iterates through the training set and adjusts the learning rate."""
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    """现在我们能够计算参数的梯度，因此可以实现SGD。分两个步骤：1.函数sdg_step计算一个批次的渐变并执行更新。 
    2.通过训练集迭代并调整学习速率的外循环。"""
    #We keep track of the losses so we can plot them later
    #model: the RNN model instance
    #X_train: the training dataset
    #y_train: the training data labels
    #evaluate_loss_after: evaluate the loss after this many epochs
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        #Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            #Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush() #python的stdout是有缓冲区的
            #Added! Saving model parameters
            i_utils.save_model_parameters_theano("./data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
        #For each training example...
        for i in range(len(y_train)):
            #One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
                  

vocabulary_size = _VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

#Read the data and append SENTENCE_START and SENTENCE_END tokens
print "Reading csv file..."
with open('data/reddit-comments-2015-08.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    #Split full comments into sentences
    #chain()可以把一组迭代对象串联起来，形成一个更大的迭代器
    #nltk:natural language toolkit, sent_tokenize:分句
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    #Append SENTENCE_START and SENTENCE_END
    """准备特殊的开始和结束标签：我们还想知道哪些词倾向在一个句子的开始和结束。为此，我们在前面加上一个
    特殊的SENTENCE_START标签，并为每个句子附加一个特殊的SENTENCE_END标签。这样我们可以提问：鉴于第一个
    标签SENTENCE_START，下一个字（实际上句子中的第一个字）最可能的什么？"""
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print "Parsed %d sentences." % (len(sentences))

#Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]


#Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())


#Get the most common words and build index_to word and word_to_index vectors
"""RNN的输入是向量，而不是字符串。因此，我们在单词和其索引之间创建一个映射，index_to_word和word_to_index。"""
"""most_common：显示最频繁的vocabulary_size-1个标识符, vocab[word, times]"""
vocab = word_freq.most_common(vocabulary_size-1)
#print "vocab:", vocab

index_to_word = [x[0] for x in vocab] #x[0]表示vocab中的word
#print "index_to_word:", index_to_word
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
#print "word_to_index:", word_to_index


print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])


#Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):  #sent代表分句
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
    
#Create the training data
"""训练示例x可以看起来像[0,179,341,416]，其中0对应于SENTENCE_START。相应的标签y将是[179,341,416,1]。
请记住，我们的目标是预测下一个字，所以y只是x向量移动一个位置，最后一个元素是SENTENCE_END标记。换句话说，
上面对字179的正确预测将是341，即实际的下一个字"""
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])


model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)
t1 = time.time()
model.sgd_step(X_train[10], y_train[10], _LEARNING_RATE)
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

if _MODEL_FILE != None:
    i_utils.load_model_parameters_theano(_MODEL_FILE, model)

train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)
