# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
BATCH_SIZE = 5

#traindata = np.load("/data/x10126.wang/SV/wxning/small/newdata.npy")
#onehotlabel = np.load("/data/x10126.wang/SV/wxning/small/newonehotlabel.npy")
#label = np.load("/data/x10126.wang/SV/wxning/small/newlabel.npy")
#print("len(traindata): ", len(traindata)) #15808076
#print("len(onehotlabel): ", len(onehotlabel))
#print("len(label): ", len(label))
#print(len(np.array(list(set([tuple(t) for t in onehotlabel])))))
#print(len(np.array(list(set([tuple(t) for t in label[0]])))))

##get max_frame_number###
#dic={}
#for item in [x[0] for x in label]:
#  #print("item: ", item)
#  dic[item] = dic.get(item, 0) + 1
#for key in dic:
#  print(key, ' vaule', dic[key])
#print("max frames: ", max(dic.values()))
#print("min_frames: ", min(dic.values()))

#from collections import defaultdict
#b = [x[0] for x in label]
#dic = defaultdict(list)
#framedata = []
#framelabel = []
#for k,va in [(v,i) for i,v in enumerate(b)]:
#  dic[k].append(va)
#for key in dic: 
#  #print(key, dic[key][0])
#  a = traindata[dic[key][0],:]
#  framelabel.append(onehotlabel[dic[key][0],:])
#  for i in dic[key][1:]:
#    a = np.vstack((a,traindata[i,:]))
#  framedata.append(a)
#  #print("key, len: ", key, len(a))
#print("len(framedata): ", len(framedata)) #76032
#print("len(framelabel): ", len(framelabel))
#
#np.save("/data/x10126.wang/SV/wxning/small/framedata.npy", framedata)
#np.save("/data/x10126.wang/SV/wxning/small/framelabel.npy", framelabel)  
#
#framedata = np.load("/data/x10126.wang/SV/wxning/small/framedata.npy")
#framelabel = np.load("/data/x10126.wang/SV/wxning/small/framelabel.npy") 
#datatrain, datatest, labeltrain, labeltest = train_test_split(framedata, framelabel, test_size=0.25,random_state=42) 
#print("len(datatrain): ", len(datatrain))
#print("len(datatest): ", len(datatest))
#np.save("/data/x10126.wang/SV/wxning/small/datatrain.npy", datatrain)
#np.save("/data/x10126.wang/SV/wxning/small/labeltrain.npy", labeltrain)
#np.save("/data/x10126.wang/SV/wxning/small/datatest.npy", datatest)
#np.save("/data/x10126.wang/SV/wxning/small/labeltest.npy", labeltest)

#print(len(np.unique(framelabel.view(framelabel.dtype.descr * framelabel.shape[1]))))
#print(len(np.array(list(set([tuple(t) for t in framelabel])))))
#
#datatrain = np.load("/data/x10126.wang/SV/wxning/preparedata/mfcc/datatrain.npy")
#labeltrain = np.load("/data/x10126.wang/SV/wxning/preparedata/mfcc/labeltrain.npy") 
#datatest = np.load("/data/x10126.wang/SV/wxning/preparedata/mfcc/datatest.npy")
#labeltest = np.load("/data/x10126.wang/SV/wxning/preparedata/mfcc/labeltest.npy") 
#print("len(datatrain): ", len(datatrain)) 
#print("len(labeltrain): ", len(labeltrain))

def TotalBatch(framedata):
  if len(framedata) % BATCH_SIZE == 0:
    return len(framedata) / BATCH_SIZE
  else:
    return len(framedata) / BATCH_SIZE + 1  

#def ShuffleData(framedata, framelabel):
#  import random
#  randnum = random.randint(0, len(framedata))
#  random.seed(randnum)
#  random.shuffle(framedata)
#  random.seed(randnum)
#  random.shuffle(framelabel)
#  return framedata, framelabel

def ShuffleData(framedata, framelabel):
  index = [i for i in range(len(framedata))]
  random.shuffle(index)
  framedata = framedata[index]
  framelabel = framelabel[index]
  return framedata, framelabel

index1 = 0
def GetTrainBatch(traindata_shuffle, onehotlabel_shuffle):
  global index1
  #print("train before: ", index1)
  if index1 < (len(traindata_shuffle) - BATCH_SIZE):   
    trainx = traindata_shuffle[index1:index1 + BATCH_SIZE][:]      
    trainy = onehotlabel_shuffle[index1:index1 + BATCH_SIZE][:]    
    index1 = index1 + BATCH_SIZE 
  else:
    trainx = traindata_shuffle[index1:][:]
    trainy = onehotlabel_shuffle[index1:][:]
    index1 = 0 
  #print("train after: ", index1)
  #print("loadframedatatrainy: ", np.argmax(trainy,1))
  return trainx, trainy
  
index2 = 0
def GetTestBatch(traindata_shuffle, onehotlabel_shuffle):
  global index2
  #print("test before: ", index2)
  if index2 < (len(traindata_shuffle) - BATCH_SIZE):   
    trainx = traindata_shuffle[index2:index2 + BATCH_SIZE][:]      
    trainy = onehotlabel_shuffle[index2:index2 + BATCH_SIZE][:]    
    index2 = index2 + BATCH_SIZE 
  else:
    trainx = traindata_shuffle[index2:][:]
    trainy = onehotlabel_shuffle[index2:][:]
    index2 = 0 
  #print("test after: ", index2)
  return trainx, trainy

def SampleTrainFrame(trainx, trainy):
  traininput = []
  trainlabel = []
  numframes = []
  cnt = 0
  for i in range(len(trainx)):
    if len(trainx[i]) > 80:
      cnt = cnt + 1
      index = random.randint(0, len(trainx[i])-80)
      a = trainx[i][index: index+80]
      traininput.append(a)
      trainlabel.append(trainy[i])
      numframes.append(80)
#      for j in range(30):
#        index = random.randint(0, len(trainx[i])-80)
#        a = trainx[i][index: index+80]
#        traininput.append(a)
#        trainlabel.append(trainy[i])
#        numframes.append(80)
    else:
      a = np.zeros((80-len(trainx[i]), 60),dtype = np.float32)
      trainx[i] = np.vstack((a,trainx[i]))
      traininput.append(trainx[i])
      trainlabel.append(trainy[i])
      numframes.append(len(trainx[i]))
  traininput = np.array(traininput)
  trainlabel = np.array(trainlabel)
  numframes = np.array(numframes).astype(np.float32)
  #print("len(traininput): ", len(traininput))
  #print("traininput: ", traininput)
  return traininput, trainlabel, numframes  #len(traininput):(batch-a)*30+a framenum:80 framedim:60
def SampleTestFrame(trainx, trainy):
  traininput = []
  trainlabel = []
  numframes = []
  cnt = 0
  for i in range(len(trainx)):
    if len(trainx[i]) > 80:
      cnt = cnt + 1
      index = random.randint(0, len(trainx[i])-80)
      a = trainx[i][index: index+80]
      traininput.append(a)
      trainlabel.append(trainy[i])
      numframes.append(80)
#      for j in range(30):
#        index = random.randint(0, len(trainx[i])-80)
#        a = trainx[i][index: index+80]
#        traininput.append(a)
#        trainlabel.append(trainy[i])
#        numframes.append(80)
    else:
      a = np.zeros((80-len(trainx[i]), 60),dtype = np.float32)
      trainx[i] = np.vstack((a,trainx[i]))
      traininput.append(trainx[i])
      trainlabel.append(trainy[i])
      numframes.append(len(trainx[i]))
  traininput = np.array(traininput)
  trainlabel = np.array(trainlabel)
  numframes = np.array(numframes).astype(np.float32)
  #print("len(testinput): ", len(traininput))
  #print("testinput: ", traininput)
  return traininput, trainlabel, numframes  #len(traininput):(batch-a)*30+a framenum:80 framedim:60




#def totalbatch():
#  return len(traindata) / BATCH_SIZE
#def shuffle_data():
#  b = [x[0] for x in label]
#  data_unique = np.unique([x[0] for x in label])
#  idx_rand = np.random.permutation(len(data_unique))
#  traindata_shuffle = []
#  onehotlabel_shuffle = []
#  for idx_r in idx_rand:
#    for i in range(len(label)):
#      if b[i] == data_unique[idx_r]:
#        traindata_shuffle_tmp = traindata[i]
#        traindata_shuffle.append(traindata_shuffle_tmp)
#        onehotlabel_shuffle_tmp = onehotlabel[i]
#        onehotlabel_shuffle.append(onehotlabel_shuffle_tmp)
#  traindata_shuffle = np.array(traindata_shuffle)
#  onehotlabel_shuffle = np.array(onehotlabel_shuffle)
#  return traindata_shuffle, onehotlabel_shuffle
