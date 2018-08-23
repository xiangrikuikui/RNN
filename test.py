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
momentum = 0.9
epoch = 1
datatest = np.load("/data/x10126.wang/SV/wxning/small/datatest.npy")
labeltest = np.load("/data/x10126.wang/SV/wxning/small/labeltest.npy") 
print("len(datatest): ", len(datatest))
#learning_rate = tf.train.exponential_decay(
#                     base_learning_rate,
#                     step,
#                     decay_steps,
#                     decay_rate,
#                     staircase=True)
def train():	

  x = tf.placeholder(tf.float32, shape=[None, num_frames, mfccdim], name = "input")
  y = tf.placeholder(tf.uint8, shape=[None, outputdim], name = "label")
  sequence = tf.placeholder(tf.float32, name = "lstm_sequence")
  
  net = lstmmodel.lstm(x, sequence)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y))
  optimizer = tf.train.MomentumOptimizer(base_learning_rate, momentum)
  train_op = optimizer.minimize(loss)
  current = tf.cast(tf.equal(tf.argmax(net, 1), tf.argmax(y, 1)), 'float')
  accuracy = tf.reduce_mean(current)  

  with tf.name_scope("loss"):
    tf.summary.scalar("loss", loss)
  with tf.name_scope("accuracy"):
    tf.summary.scalar("accuracy", accuracy)
  merged_summary = tf.summary.merge_all()
	
#  for var in tf.trainable_variables():
#    print("var.name: ", var.name)
  gpu_options = tf.GPUOptions(allow_growth=True)
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

  #init = tf.global_variables_initializer()
  #sess.run(init)
#  new_saver = tf.train.import_meta_graph("./mylstmmodel/mylstmmodel.ckpt-8.meta")
#  new_saver.restore(sess, tf.train.latest_checkpoint('./mylstmmodel')) 
  saver=tf.train.Saver()
  model_file=tf.train.latest_checkpoint('mylstmmodel/')
  saver.restore(sess,model_file)
  #saver.restore(sess,'./mylstmmodel/mylstmmodel.ckpt-14')
  TotalBatch = loadframedata.TotalBatch(datatest)
  for i in range(epoch):
    datax, datay = loadframedata.ShuffleData(datatest, labeltest)
    acc = 0
    for j in range(TotalBatch):
      testx, testy = loadframedata.GetTestBatch(datax, datay)
      x_test, y_test, numframes = loadframedata.SampleTrainFrame(testx, testy)
      [ACC, losses] = sess.run([accuracy, loss], feed_dict={x: x_test, y: y_test, sequence: numframes}) 
      acc = acc + ACC
      if j % 100 == 0 and j != 0:
        print('batch %d, ACC %g, losses %g' % (j, ACC, losses))
    print("mean acc: ", acc/TotalBatch)
train()

## -*- coding: utf-8 -*-
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
#
#import numpy as np
#import tensorflow as tf
#import tensorflow.contrib.slim as slim
#import tensorflow.contrib.layers as layers
#import lstmmodel
#import os
#import loadframedata
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#
#
#num_frames = 80
#mfccdim = 60
#outputdim = 3
#base_learning_rate = 0.001
#decay_rate = 0.9
#momentum = 0.9
#epoch = 1
#datatest = np.load("/data/x10126.wang/SV/wxning/test/datatest.npy")
#labeltest = np.load("/data/x10126.wang/SV/wxning/test/labeltest.npy")
#print("len(datatest): ", len(datatest))
##learning_rate = tf.train.exponential_decay(
##                     base_learning_rate,
##                     step,
##                     decay_steps,
##                     decay_rate,
##                     staircase=True)
#def train():	
#
#  x = tf.placeholder(tf.float32, shape=[None, num_frames, mfccdim], name = "input")
#  y = tf.placeholder(tf.uint8, shape=[None, outputdim], name = "label")
#  sequence = tf.placeholder(tf.float32, name = "lstm_sequence")
#  
#  net = lstmmodel.lstm(x, y, sequence)
#  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y))
#  optimizer = tf.train.MomentumOptimizer(base_learning_rate, momentum)
#  train_op = optimizer.minimize(loss)
#  current = tf.cast(tf.equal(tf.argmax(net, 1), tf.argmax(y, 1)), 'float')
#  accuracy = tf.reduce_mean(current)  
#
#  with tf.name_scope("loss"):
#    tf.summary.scalar("loss", loss)
#  with tf.name_scope("accuracy"):
#    tf.summary.scalar("accuracy", accuracy)
#  merged_summary = tf.summary.merge_all()
#	
##  for var in tf.trainable_variables():
##    print("var.name: ", var.name)
#  gpu_options = tf.GPUOptions(allow_growth=True)
#  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#  #init = tf.global_variables_initializer()
#  #sess.run(init)
#  saver = tf.train.Saver()
#  
#  ckpt=tf.train.get_checkpoint_state('mylstmmodel/')
#  print(ckpt)
#  
#  TotalBatch = loadframedata.TotalBatch(datatest)
#  
#  if ckpt and ckpt.all_model_checkpoint_paths:
#  #加载模型
#  #这一部分是有多个模型文件时，对所有模型进行测试验证
#    for path in ckpt.all_model_checkpoint_paths:
#      saver.restore(sess,path)                
#      global_step=path.split('/')[-1].split('-')[-1]
#      print("model ", global_step)
#      
#      for i in range(epoch):
#        datax, datay = loadframedata.ShuffleData(datatest, labeltest)
#        acc = 0
#        for j in range(TotalBatch):
#          testx, testy = loadframedata.GetTestBatch(datax, datay)
#          x_test, y_test, numframes = loadframedata.SampleFrame(testx, testy)
#          #sess.run(train_op, feed_dict={x: x_test, y: y_test, sequence: numframes})   
#          [ACC, losses] = sess.run([accuracy, loss], feed_dict={x: x_test, y: y_test, sequence: numframes}) 
#          acc = acc + ACC
#          if j % 10 == 0 and j != 0:
#            print("batch %d, ACC %g, losses %g" % (j, ACC, losses))
#        print("mean acc: ", acc/TotalBatch)
#  else:
#    print('No checkpoint file found')
#
#train()

