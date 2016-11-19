from __future__ import division, print_function, absolute_import
import numpy as np
import tflearn
import tensorflow as tf
from tflearn.datasets import cifar10
import sys

def resnet(dataset,img_prep,img_aug,X,Y,testX,testY,width,height,channel,class_num,filt,depth,epoch):
  # Building Residual Network
  layer=1
  net = tflearn.input_data(shape=[None, width, height, channel],
                           data_preprocessing=img_prep,
                           data_augmentation=img_aug)
  net = tflearn.conv_2d(net, filt, 3, regularizer='L2', weight_decay=0.0001)
  while(depth!=0):
    d_num=depth-(int)(depth/100)*100
    depth=(int)(depth/100)
    if(layer==1):
      net = tflearn.residual_block(net, d_num, filt)
    else:
      net = tflearn.residual_block(net, 1, filt, downsample=True)
      net = tflearn.residual_block(net, d_num-1, filt)
    layer=layer+1
  net = tflearn.batch_normalization(net)
  net = tflearn.activation(net, 'relu')
  net = tflearn.global_avg_pool(net)
  # Regression
  net = tflearn.fully_connected(net, class_num, activation='softmax')
  mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
  net = tflearn.regression(net, optimizer=mom,
                           loss='categorical_crossentropy')
  # Training
  model = tflearn.DNN(net, checkpoint_path=('model_resnet_'+dataset),
                      max_checkpoints=10, tensorboard_verbose=0,
                      clip_gradients=0.)
  model.fit(X, Y, n_epoch=epoch, validation_set=(testX, testY),
            snapshot_epoch=False, snapshot_step=500,
            show_metric=True, batch_size=128, shuffle=True,
            run_id=('resnet_'+dataset))
  aa=model.predict(testX)
  correct=0
  for i in range(len(aa)):
    if(aa[i].index(max(aa[i])) == np.argmax(testY[i])):
      correct=correct+1
  return correct/len(aa)

if __name__=="__main__":
  # initial setting
  #depth=1
  #step=1
  #flag=0
  #pre_acc=-1
  
  depth=int(sys.argv[1])
  step=int(sys.argv[2])
  flag=int(sys.argv[3])
  pre_acc=float(sys.argv[4])

  outflag=0
  filt=16
  epoch=200
  # cifar10 dataset
  dataset="cifar10"
  width=32
  height=32
  channel=3
  class_num=10
  (X, Y), (testX, testY) = cifar10.load_data()
  Y = tflearn.data_utils.to_categorical(Y, 10)
  testY = tflearn.data_utils.to_categorical(testY, 10)
  # Real-time data preprocessing
  img_prep = tflearn.ImagePreprocessing()
  img_prep.add_featurewise_zero_center(per_channel=True)
  # Real-time data augmentation
  img_aug = tflearn.ImageAugmentation()
  img_aug.add_random_flip_leftright()
  img_aug.add_random_crop([width, height], padding=4)
  #learning and evaluating
  acc=resnet(dataset,img_prep,img_aug,X,Y,testX,testY,width,height,channel,class_num,filt,depth,epoch)
  print(dataset+"|filter_num:"+str(filt)+"|depth:"+str(depth)+"|epoch:"+str(epoch)+"|eval accuracy:"+str(acc))
  tf.reset_default_graph()
  if(acc > pre_acc):
    depth+=step
    flag=0
    #pre_acc=acc
  else:
    if(flag==1):
      outflag=1
    else:
      depth-=step
      step*=100
      depth+=step
      flag=1
  print("Next Step Paramemters|depth:"+str(depth)+"|step:"+str(step)+"|flag:"+str(flag)+"|outflag:"+str(outflag)+"|pre_acc:"+str(pre_acc))
