
# coding: utf-8

# In[1]:


from network import alex_net
from tfdata import *
import numpy as np
from training import accuracy_of_batch


# In[2]:


import tensorflow as tf


# In[3]:


import cv2


# In[4]:


# Dataset path
train_tfrecords = 'train.tfrecords'
#val_tfrecords='val.tfrecords'
test_tfrecords = 'test.tfrecords'
batch_size=20


# In[5]:


img,label=input_pipeline(test_tfrecords,batch_size,is_shuffle=False,is_train=False)
with tf.variable_scope('model_definition'):
    prediction=alex_net(img,train=False)
accuracy=accuracy_of_batch(prediction,label)


# In[6]:


saver=tf.train.Saver()


# In[7]:


with tf.Session() as sess:
    saver.restore(sess,'checkpoint/my-model.ckpt-2000')
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    for i in range(21):
        acc=sess.run(accuracy)
        print(acc)
    coord.request_stop()
    coord.join(threads)

