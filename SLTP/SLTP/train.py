from __future__ import absolute_import, division, print_function, unicode_literals
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras import layers, models
from keras.models import Sequential
from keras import layers
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.pylab as pylab
import pandas as pd
#from sklearn.linear_model import LinearRegression
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import random
from keras import optimizers
from keras.layers import SimpleRNN, Dense
from keras.layers import Bidirectional
import os
import sys
import argparse
np.random.seed(1)

'''
#다항식 정의 함수
def fit_polynomial(x,y,degree):
    model = LinearRegression()
    model.fit(np.vander(x, degree+1),y)
    return model
#x입력시 model에의해 y출력
def apply_polynomial(model,x):
    degree = model.coef_.size - 1
    y = model.predict(np.vander(x, degree+1))
    return y
'''
def make_label(text):
    with open("label.txt", "w") as f:
         f.write(text)
    f.close()
    
def load_data(dirname):
    if dirname[-1]!='/':
        dirname=dirname+'/'
    listfile=os.listdir(dirname)
    X = [] #train data set
    Y = []
    x = [] #graph x,y
    y = []
    XT = [] #test data set (30%)
    YT = []
    for file in listfile:
        if "_" in file:
            continue
        wordname=file
        textlist=os.listdir(dirname+wordname)
        #a=len(textlist)
        #b=a//3
        k=0
        for text in textlist:
            if "DS_" in text:
                continue
            textname=dirname+wordname+"/"+text
            numbers=[]
            #print(textname)
            with open(textname, mode = 'r') as t:
                numbers = [float(num) for num in t.read().split()] #read landmark
                '''check front -> delete 0 , ~20000'''
                #while (numbers[0] == 0.0):
                #    del numbers[0]
                while (len(numbers) > 45000):
                    del numbers[len(numbers)-1]
                #for i in range(len(numbers)-1, len(numbers)-100, -1):
                #    if (numbers[i]==0.0): del numbers[i]
                for i in range(len(numbers),45000):
                    numbers.extend([0.000]) #add 0.000 ~20000
            #print('len numbers : ',len(numbers))
            row=0
            landmark_frame=[]
            '''
            for i in range(0,200):
                landmark_frame.extend(numbers[row:row+105]) #200*105(가로)
                row += 105
            '''
            landmark_frame.extend(numbers)
            landmark_frame=np.array(landmark_frame)
            #landmark_frame=list(landmark_frame.reshape(-1,105)) #200*100(세로)
            if (k%4==3):
                XT.append(np.array(landmark_frame))
                YT.append(wordname)
            else:
                X.append(np.array(landmark_frame))
                Y.append(wordname)
            k+=1

    X=np.array(X)
    Y=np.array(Y)
    XT=np.array(XT)
    YT=np.array(YT)
    print(Y.shape)
    print(YT.shape)
    print(X.shape)
    print(XT.shape)
    
    tmp = [[x,y] for x, y in zip(X, Y)]
    random.shuffle(tmp)

    tmp1 = [[xt,yt] for xt, yt in zip(XT, YT)]
    random.shuffle(tmp1)
    
    X = [n[0] for n in tmp]
    Y = [n[1] for n in tmp]
    XT = [n[0] for n in tmp1]
    YT = [n[1] for n in tmp1]

    k=set(Y)
    ks=sorted(k)
    text=""
    for i in ks:
        text=text+i+" "
    make_label(text)
    
    s = Tokenizer()
    s.fit_on_texts([text])
    encoded=s.texts_to_sequences([Y])[0]
    encoded1=s.texts_to_sequences([YT])[0]
    one_hot = to_categorical(encoded)
    one_hot2 = to_categorical(encoded1)

    #print(encoded)
    #print(one_hot)
    '''
    X=np.array(X)
    X=X.reshape(X.shape[0],45000,1)
    XT=np.array(XT)
    XT=XT.reshape(XT.shape[0],45000,1)
    '''

    (x_train, y_train) = X, one_hot
    (x_test, y_test)= XT, one_hot2
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    x_test=np.array(x_test)
    y_test=np.array(y_test)

    print(y_train.shape)
    print(y_test.shape)
    print(x_train.shape)
    print(x_test.shape)

    return x_train,y_train,x_test,y_test

def build_model(label): #multi-layer LSTM
    model = Sequential()
    '''
    #model.add(Embedding(50,20000))
    model.add(layers.LSTM(64, return_sequences=True,
                   input_shape=(200, 105)))
    #model.add(layers.LSTM(32, return_sequences=True))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(label, activation='softmax'))
    '''
    model.add(layers.Dense(128, input_shape=(45000,), activation='sigmoid'))
    #model.add(layers.Dense(64, activation='sigmoid'))
    model.add(layers.Dense(51, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',                                                          
                  metrics=['accuracy'])
    return model

def main(dirname):
    x_train,y_train,x_test,y_test=load_data(dirname)
    #num_val_samples=(x_train.shape[0])//5
    model=build_model(y_train.shape[1])
    print('Training stage')
    print('==============')
    history=model.fit(x_train,y_train,epochs=100,batch_size=50,validation_data=(x_test,y_test))
    score, acc = model.evaluate(x_test,y_test,verbose=0)
    print('Test performance: accuracy={0}, loss={1}'.format(acc, score))
    model.save('model.h5')
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training Model')
    parser.add_argument("--input_train_path",help=" ")
    args=parser.parse_args()
    input_train_path=args.input_train_path
    main(input_train_path)