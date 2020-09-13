from __future__ import absolute_import, division, print_function, unicode_literals
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras import layers, models
from keras.models import Sequential
from keras import layers
import os
import sys
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import random
from keras import optimizers
from keras.layers import SimpleRNN, Dense
from keras.layers import Bidirectional
import tensorflow as tf
from numpy import argmax
import argparse
import shutil

def load_data(dirname):
    if dirname[-1]!='/':
        dirname=dirname+'/'
    listfile=os.listdir(dirname)
    X = []
    Y = []
    for file in listfile:
        if "_" in file:
            continue
        if "." in file:
            continue
        wordname=file
        textlist=os.listdir(dirname+wordname)
        for text in textlist:
            if "DS_" in text:
                continue
            textname=dirname+wordname+"/"+text
            numbers=[]
            #print(textname)
            with open(textname, mode = 'r') as t:
                numbers = [float(num) for num in t.read().split()]
                while (len(numbers) > 45000):
                    del numbers[len(numbers)-1]
                for i in range(len(numbers),45000):
                    numbers.extend([0.000])
            landmark_frame=[]
            landmark_frame.extend(numbers)
            landmark_frame=np.array(landmark_frame)
            X.append(np.array(landmark_frame))
            Y.append(wordname)
    X=np.array(X)
    Y=np.array(Y)
    x_train=np.array(X)
    return x_train,Y


#prediction
def load_label():
    listfile=[]
    with open("label.txt",mode='r') as l:
        listfile=[i for i in l.read().split()]
    label = {}
    count = 1
    for l in listfile:
        if "_" in l:
            continue
        label[l] = count
        count += 1
    return label
    
def main(input_data_path,output_data_path):
    
    comp=comp='bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11   mediapipe/examples/desktop/multi_hand_tracking:multi_hand_tracking_gpu'
    #명령어 컴파일
    cmd='GLOG_logtostderr=1 /home/pi/mediapipe/bazel-bin/mediapipe/examples/desktop/multi_hand_tracking/multi_hand_tracking_gpu   --calculator_graph_config_file=/home/pi/mediapipe/mediapipe/graphs/hand_tracking/multi_hand_tracking_desktop_live.pbtxt'
    #미디어 파이프 명령어 저장
    listfile=os.listdir(input_data_path)
    output_dir=""
    filel=[]
    for file in listfile:
        if ".DS_" in file:
            continue
        word=file+'/'
        fullfilename=os.listdir(input_data_path+word)
        # 하위디렉토리의 모든 비디오들의 이름을 저장
        if not(os.path.isdir(output_data_path+"_"+word)):
            os.mkdir(output_data_path+"_"+word)
        if not(os.path.isdir(output_data_path+word)):
            os.mkdir(output_data_path+word)
        os.system(comp)
        outputfilelist=os.listdir(output_data_path+'_'+word)
        for mp4list in fullfilename:
            if ".DS_Store" in mp4list:
                continue
            filel.append(mp4list)
            inputfilen='   --input_video_path='+input_data_path+word+mp4list
            outputfilen='   --output_video_path='+output_data_path+'_'+word+mp4list
            cmdret=cmd+inputfilen+outputfilen
            os.system(cmdret)
    #mediapipe동작 작동 종료:
    output_dir=output_data_path
    
    x_test,Y=load_data(output_data_path) #output_dir
    new_model = tf.keras.models.load_model('model.h5')
    #new_model.summary()

    labels=load_label()

    #모델 사용

    xhat = x_test
    yhat = new_model.predict(xhat)
    predictions = np.array([np.argmax(pred) for pred in yhat])
    print(yhat)
    print(predictions)
    rev_labels = dict(zip(list(labels.values()), list(labels.keys())))
    filel=np.array(filel)
    txtpath=output_data_path+"result.txt" 

    with open(txtpath, "w") as f:
        f.seek(0)
        f.write(rev_labels[predictions[0]+1])
        f.write("\n")
        f.close()

    listfile=os.listdir(output_data_path)
    for file in listfile:
        if "." in file:
            continue
        else:
            shutil.rmtree(output_data_path+file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Sign language with Mediapipe')
    parser.add_argument("--input_data_path",help=" ")
    parser.add_argument("--output_data_path",help=" ")
    args=parser.parse_args()
    input_data_path=args.input_data_path
    output_data_path=args.output_data_path
    main(input_data_path,output_data_path)