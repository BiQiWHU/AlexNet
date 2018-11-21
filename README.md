# file information in detail
tfdata.py: read, encode and decode images and labels with a format of tfrecord
my_ops:offer some basic operations
network.py:an self-implementation of AlexNet
training.py:using tensorflow to train AlexNet and save models
test.py:predict labels on the test dataset
textforCM:a python file to output all labels on test dataset.

Environment:
(1) Tensorflow >=1.0
(2) Python3

Operations:
(1) use our tfdata.py to generate your own dataset. please modify it according to your own dataset, parameters such as the number of classes and image size must be changed, or it can not run. 
(2) use training.py to train AlexNet.
(3) use test.py to test your data, choose the best model and use textforCM to output the prediction labels
Enjoy!
