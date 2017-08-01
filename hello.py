#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:45:30 2017

@author: yawen
"""
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import statsmodels.api as sm
import pandas as pd
import numpy as np
def read():
    #read raw data
    import pandas as pd
    df = pd.read_csv('data.csv')
    mag_control = df[['Time Stamp','chwp_pc','cwp_pc','ct_pc']]
    sta = df[['Time Stamp','chwp1stat','chwp2stat','chwp3stat','chwp4stat','cwp1stat','cwp2stat','cwp3stat','ch1stat','ch2stat','ch3stat','ct1stat','ct2stat']]
    time = df['Time Stamp']
    env = df[['Time Stamp','rh','drybulb','wetbulb']]
    return df,mag_control,sta,time,env

def convert_time(time):
    #input: time of type Series
    time = list(time)
    from datetime import datetime
    atime = [datetime.strptime(i,'%d/%m/%Y %H:%M') for i in time]
    return atime

def find_abnormal_time(df,name,time):
    #input:time of type datetime
    val = df[name].values
    import matplotlib.pyplot as plt
    #first look of value
#    plt.scatter(time,val)
    plt.plot(time,val)
    
def find_abnormal(df,name):
    val = df[name].values
    import matplotlib.pyplot as plt
    #first look of value
    plt.plot(val)
    
def linear(xtrain,ytrain):
    li = LinearRegression()
    li.fit(xtrain,ytrain)
    return li

def main():
    df,mag_control,sta,time,env = read()
    time = convert_time(time)
    find_abnormal(sta,'ct1stat')
    df.corr()
    
def normalTest(df):
    #正态分布检验
    from scipy.stats import kstest
    from scipy import stats
    from scipy.stats import normaltest
    
    col = df.columns
    for name in col:
        try:
            val = df[name].values
            print name,'...'
            print kstest(val, 'norm')
            print normaltest(val, axis=None)
            print stats.shapiro(val)
            print '........................'
        except:
            print name,'  type err!'
    
    
def cal_abnormal(df):
    f = open('abn.txt','w+')
    col = df.columns
    for name in col:
        try:
            arr = df[name].values
            print name,'...'
            strn = name+'...\n'
            f.write(strn)
            mean =arr.mean()
            std = arr.std()
            for i in range(len(arr)):
                temp = abs(arr[i]-mean)-3*std
                if temp > 0:
                    print i,arr[i]
                    strn = str(i)+'   '+str(arr[i])+'\n'
                    f.write(strn)
            print '........................'
            f.write('......................\n')
        except:
            print name,'  type err!'
    f.close()
    
def get_abnormal(df):
    col = df.columns
    dic = {}
    for name in col:
        try:
            arr = df[name].values
            print name,'...'
            dic[name]=[]
            mean =arr.mean()
            std = arr.std()
            for i in range(len(arr)):
                temp = abs(arr[i]-mean)-3*std
                if temp > 0:
                    print i,arr[i],'...',
                    dic[name].append((i,arr[i]))
            print '........................'
        except:
            print name,'  type err!'
        
    ind = []
    for item in dic.keys():
        temp = [i[0] for i in dic[item]]
        ind.extend(temp)
    ind1 = list(set(ind))
    ind1.sort()
    return ind1
    
def dropab(df,index):
    for item in index:
        df = df.drop(item)
    return df
    
def minmaxScale(arr):
    new = []
    mi = min(arr)
    ma = max(arr)
    for item in arr:
        if ma-mi!=0:
            temp = (item-mi)/(ma-mi)
            new.append(temp)
        else:
            print ma,mi
    return new
    
def printcol():
    alll = ['Time Stamp', 'chwrhdr', 'chwshdr', 'chwsfhdr', 'cwshdr', 'cwrhdr', 'cwsfhdr', 'chwgpmrt', 'cwgpmrt','cwgpmrt_sp', 'chgpmrt_sp', 'ct_eff_sp', 'dch']
    env = ['rh','drybulb','wetbulb']
    sta = ['ch1stat', 'ch2stat', 'ch3stat', 'chwp1stat', 'chwp2stat', 'cwp1stat', 'cwp2stat', 'cwp3stat', 'ct1stat',
 'ct2stat','chwp3stat', 'chwp4stat']
    kw = ['ch1kw', 'ch2kw', 'chwp1kw', 'chwp2kw', 'cwp1kw', 'cwp2kw', 'cwp3kw', 'ct1kw', 'ct2kw', 'ch3kw', 'chwp3kw', 'chwp4kw']
    sys = ['systotpower', 'loadsys', 'effsys', 'hbsys',]
    pc = ['cwp_pc', 'chwp_pc', 'ct_pc']
    eff = ['chiller_eff', 'cwp_eff', 'chwp_eff', 'ct_eff']
    return alll,env,sta,kw,sys,pc,eff
    
def dropcol(df,s=['chwp3stat','chwp4stat','chwp4kw','chwp3kw','cwp3kw','cwp3stat','ch3stat','ch3kw']):
    df2 = df.drop(s,axis=1)
    return df2
    
def linear2(xtrain,ytrain,xtest,ytest):
    print ytest.columns
    xtrain = sm.add_constant(xtrain)
    model = sm.OLS(ytrain,xtrain)
    results = model.fit()
    print(results.params)
    print(results.summary())
#    print '....predict'
#    xtest=sm.add_constant(xtest)
#    y_hat=results.predict(xtest)
#    label = np.array(ytest)
#    pred = np.array(y_hat)
#    print(metrics.r2_score(label, pred))
#    print(metrics.explained_variance_score(label, pred))


def nihe(co,name,train,test):
    co1 = co[[name]][co[name] > 0.5]
    s = list(co1.index)
    fei = ['ch1kw', 'ch2kw', 'chwp1kw', 'chwp2kw', 'cwp1kw', 'cwp2kw', 'cwp3kw', 'ct1kw', 'ct2kw', 'ch3kw', 'chwp3kw', 'chwp4kw','systotpower', 'loadsys', 'effsys', 'hbsys','chiller_eff', 'cwp_eff', 'chwp_eff', 'ct_eff','cwgpmrt_sp','ct_eff_sp']
    sn = [i for i in s if i not in fei]
#    print sn
    xtrain = train[sn]
    ytrain = train[[name]]
    xtest = test[sn]
    ytest = test[[name]]
    li = linear(xtrain,ytrain)
    print li.coef_
    print li.score(xtest,ytest)
    linear2(xtrain,ytrain,xtest,ytest)
    