
# coding: utf-8

# In[1]:


from detailnetwork import alex_net
from tfdata import *
import numpy as np


# In[2]:


import tensorflow as tf
import os


# In[3]:


test_tfrecords = 'test.tfrecords'


# In[4]:


batch_size=20


# In[5]:


img,label=input_pipeline(test_tfrecords,batch_size,is_shuffle=False,is_train=False)


# In[6]:


def accuracy_of_batch(inputtens,logits,targets):
    # Make sure targets are integers and drop extra dimensions
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    # Get predicted values by finding which logit is the greatest
    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    # Check if they are equal across the batch
    ### tf equal只能传回来True or False,PC输出的就是一个只有1和0的向量 可满足要求
    return targets,batch_predictions


# In[7]:


with tf.variable_scope('model_definition'):
    dropout7,prediction=alex_net(img,train=False)
accuracy=accuracy_of_batch(dropout7,prediction,label)


# In[8]:


saver=tf.train.Saver()


# In[9]:


file_write_obj = open("result1.txt", 'w')


# In[11]:


with tf.Session() as sess:
    saver.restore(sess,'checkpoint/my-model.ckpt-12000')
    
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    for i in range(50):
        acc1,acc2=sess.run(accuracy)
        print(acc1)
        file_write_obj.writelines(str(acc1))
        file_write_obj.write('\n')
        
        print(acc2)
        file_write_obj.writelines(str(acc2))
        file_write_obj.write('\n')
    print('finished!')
    file_write_obj.close()
    coord.request_stop()
    coord.join(threads)
    

