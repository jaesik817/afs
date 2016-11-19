#!/bin/bash
#initial parameters
depth=50
step=50
flag=0
outflag=0
pre_acc=-1
while true
do
  if [ $outflag -eq 1 ];then
    break
  fi
  
  nohup python resnet_cifar10.py $depth $step $flag $pre_acc > resnet_cifar10.log
  echo `date` `cat resnet_cifar10.log |tail -2 |head -1`
  echo `date` `cat resnet_cifar10.log |tail -1 |head -1`
  depth=`cat resnet_cifar10.log |tail -1 |cut -f 2 -d "|"|cut -f 2 -d ":"`
  step=`cat resnet_cifar10.log |tail -1 |cut -f 3 -d "|"|cut -f 2 -d ":"`
  flag=`cat resnet_cifar10.log |tail -1 |cut -f 4 -d "|"|cut -f 2 -d ":"`
  outflag=`cat resnet_cifar10.log |tail -1 |cut -f 5 -d "|"|cut -f 2 -d ":"`
  pre_acc=`cat resnet_cifar10.log |tail -1 |cut -f 6 -d "|"|cut -f 2 -d ":"`
done
echo "AFS Done!!"
