# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
import lstmmodel
import os
import loadframedata
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


num_frames = 80
mfccdim = 60
outputdim = 3
base_learning_rate = 0.001
decay_rate = 0.9
decay_steps = 200000
momentum = 0.9
epoch = 10
#learning_rate = tf.train.exponential_decay(
#                     base_learning_rate,
#                     step,
#                     decay_steps,
#                     decay_rate,
#                     staircase=True)

datatrain = np.load("/data/x10126.wang/SV/wxning/small/datatrain.npy")
labeltrain = np.load("/data/x10126.wang/SV/wxning/small/labeltrain.npy")
print("len(datatrain): ", len(datatrain)) 
datatest = np.load("/data/x10126.wang/SV/wxning/small/datatest.npy")
labeltest = np.load("/data/x10126.wang/SV/wxning/small/labeltest.npy") 
print("len(datatest): ", len(datatest))

#datatrain = np.load("/data/x10126.wang/SV/wxning/small/batchdata.npy")
#labeltrain = np.load("/data/x10126.wang/SV/wxning/small/batchlabel.npy")
#print("len(datatrain): ", len(datatrain)) 
#datatest = np.load("/data/x10126.wang/SV/wxning/small/batchtestdata.npy")
#labeltest = np.load("/data/x10126.wang/SV/wxning/small/batchtestlabel.npy") 
#print("len(datatest): ", len(datatest))
#print("labeltrain: ", np.argmax(labeltrain, 1))

def train():	

  x = tf.placeholder(tf.float32, shape=[None, num_frames, mfccdim], name = "input")
  y = tf.placeholder(tf.uint8, shape=[None, outputdim], name = "label")
  sequence = tf.placeholder(tf.float32, name = "lstm_sequence")
  
  net = lstmmodel.lstm(x, sequence)
  net1 = tf.nn.softmax(net)
  xx = x
  yy = y

  with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y))
    #loss = np.mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y))
    tf.summary.scalar("loss", loss)
  
  with tf.name_scope("accuracy"):   
    correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
  
  merged_summary = tf.summary.merge_all()
	
  train_op = tf.train.MomentumOptimizer(base_learning_rate, momentum).minimize(loss) 

  gpu_options = tf.GPUOptions(allow_growth=True)
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
  writer = tf.summary.FileWriter("./mylstmmodel", sess.graph)
  init = tf.global_variables_initializer()
  sess.run(init)
  
  saver = tf.train.Saver(max_to_keep=10) #saver=tf.train.Saver(max_to_keep=5),保存最近的5个模型
  TrainBatch = int(loadframedata.TotalBatch(datatrain))
  TestBatch = int(loadframedata.TotalBatch(datatest))
  print("TrainBatch: ", TrainBatch)
  print("TestBatch: ", TestBatch)
  f = open('./mylstmmodel/acc.txt','w')
  for i in range(epoch):
    acctrain = []
    acctest = []
    datax, datay = loadframedata.ShuffleData(datatrain, labeltrain)
    dataxtest, dataytest = loadframedata.ShuffleData(datatest, labeltest)
    acctrain = []
    acctest = []
    for j in range(TrainBatch):
      
      trainx, trainy = loadframedata.GetTrainBatch(datax, datay)
      x_train, y_train, trainnumframes = loadframedata.SampleTrainFrame(trainx, trainy) 
      testx, testy = loadframedata.GetTestBatch(dataxtest, dataytest)
      x_test, y_test, testnumframes = loadframedata.SampleTrainFrame(testx, testy)
      
      sess.run(train_op, feed_dict={x: x_train, y: y_train, sequence: trainnumframes})   
      
      if j % 10 == 0:
        [AccTrain, LossesTrain] = sess.run([accuracy, loss], feed_dict={x: x_train, y: y_train, sequence: trainnumframes})
        [AccTest, LossesTest] = sess.run([accuracy, loss], feed_dict={x: x_test, y: y_test, sequence: testnumframes})
        
        trainnet = sess.run(net, feed_dict={x: x_train, sequence: trainnumframes})   
        trainy = sess.run(yy, feed_dict={y: y_train}) 
        print("net: ", sess.run(tf.argmax(net, 1), feed_dict={x: x_train, sequence: trainnumframes}))
        print("trainet: ", sess.run(tf.argmax(trainnet, 1)))
        print("netsoftmax: ", sess.run(tf.argmax(net1, 1), feed_dict={x: x_train, sequence: trainnumframes}))
        print("trainy:  ", sess.run(tf.argmax(trainy, 1)))  
         
        
        testnet = sess.run(net1, feed_dict={x: x_test, sequence: testnumframes})   
        testy = sess.run(yy, feed_dict={y: y_test})  
        #print("testsoftmax: ", sess.run(net1, feed_dict={x: x_test, sequence: testnumframes}))
        print("testy:   ", sess.run(tf.argmax(testy, 1))) 
        print("testnet: ", sess.run(tf.argmax(testnet, 1)))      
                
        acctrain.append(AccTrain)       
        acctest.append(AccTest)
        summary = sess.run(merged_summary, feed_dict={x: x_train, y: y_train, sequence: trainnumframes})
        writer.add_summary(summary, i)
        
        print('epoch %d  batch %d  TrainAcc %.4g  TrainLoss %.4g  TestAcc %.4g  TestLoss %.4g' % (i, j, AccTrain, LossesTrain, AccTest, LossesTest))
        f.write('epoch ' + str(i) + ', batch ' + str(j) + ', TrainAcc '+ str(AccTrain) + ', TrainLoss: ' + str(LossesTrain) + ', TestAcc '+ str(AccTest) + ', TestLoss ' + str(LossesTest) + '\n')
    
    saver.save(sess, "./mylstmmodel/mylstmmodel.ckpt", global_step=i)
       
    acctrain = np.mean(np.array(acctrain))
    acctest = np.mean(np.array(acctest))
    print('MeanTrainAcc %.4g  MeanTestAcc %.4g' % (acctrain, acctest))
    f.write('MeanTrainAcc ' + str(acctrain) + ', MeanTestAcc ' + str(acctest))
  f.close()   

train()

#w1 = sess.graph.get_tensor_by_name("fc1/weights:0") 
#print(w1.get_shape().as_list()) 
#print(sess.run(a))
#np.savetxt("w1.txt", w1.eval())
#w2 = sess.graph.get_tensor_by_name("fc2/weights:0")
#print(w2.get_shape().as_list())
#np.savetxt("w2.txt", w2.eval())
#graph = tf.get_default_graph()
#in = graph.get_tensor_by_name("input:0")
#print("input: ", sess.run(in))

#  for var in tf.trainable_variables():
#    print("var.name: ", var.name)
  #with tf.Session() as sess:
        
#      trainx, trainy = loadframedata.GetTrainBatch(datatrain, labeltrain)
#      x_train, y_train, trainnumframes = loadframedata.SampleTrainFrame(trainx, trainy)
#      testx, testy = loadframedata.GetTestBatch(datatest, labeltest)
#      x_test, y_test, testnumframes = loadframedata.SampleTestFrame(testx, testy) 

#      datax, datay = loadframedata.GetTrainBatch(datatrain, labeltrain)
#      trainx, trainy = loadframedata.ShuffleData(datax, datay)
#      x_train, y_train, trainnumframes = loadframedata.SampleTrainFrame(trainx, trainy)
#      
#      dataxtest, dataytest = loadframedata.GetTestBatch(datatest, labeltest)
#      testx, testy = loadframedata.ShuffleTestData(dataxtest, dataytest)
#      x_test, y_test, testnumframes = loadframedata.SampleTestFrame(testx, testy)
