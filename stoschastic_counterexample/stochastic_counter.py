# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from math import sqrt,exp
from random import random,seed
from scipy.signal import savgol_filter

font = {
        'family' : 'monospace',
        'size'   : 14  ,
        }
matplotlib.rc('font', **font)

def exp_normalFunction(sigma=1000,bound=100):
    bound=10
    x=np.linspace(-bound,bound,500,np.float32)
    y=np.exp(-(x**2-1)**2/5**4)
    y=x**(-1/2)*np.exp(-x/2)
    y=x**2*np.exp(-x**2/2)
    plt.figure(3)
    plt.plot(x,y)
    plt.show()

def get_gt(size):
    pass

class window():
    def __init__(self,length,beta2):
        self.length=length
        self.nextindex=0
        self.beta2=beta2
        self.windowArray=np.zeros(self.length,dtype=np.float32)
        self.wight_vector=np.array( [ beta2**(self.length-i) for i in range(1,self.length+1) ],dtype=np.float32 )
        self.wight_vector=self.wight_vector/np.sum(self.wight_vector)
        self.isFirst = True

    def append(self,element):
        if self.nextindex<self.length:
            self.windowArray[int(self.nextindex)]=element
            self.nextindex+=1
        else:
            self.isFirst = False
            self.windowArray[self.nextindex%self.length]=element
            self.nextindex=self.nextindex%self.length + 1
    def get_moving_average(self):
        if self.nextindex == 0:
            raise 'No element'
        elif self.nextindex<self.length:
            average = np.sum(self.windowArray[0:self.nextindex]*self.wight_vector[self.length-self.nextindex:self.length])+ np.sum(self.windowArray[self.nextindex:self.length] * self.wight_vector[0:self.length-self.nextindex])
        else:
            average = np.sum(self.windowArray[0:self.nextindex] * self.wight_vector[self.length - self.nextindex:self.length])
        return average

    def popTail(self):
        return self.windowArray[(self.nextindex)%self.length]


def randomGt(C,sigma=0.02):
    p=(1+sigma)/(C+1)
    q=random()
    if q<=p:
        gt=C
    else:
        gt=-1
    return gt

def adam(obj,C,start=10000,plusPro=False,plusMt=True,stepnum=100000,alpha=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8):
    c=epsilon
    mt,vt=0,0
    deta=start
    detas=[]

    seed(1)
    for i in range(stepnum+10000):
        global gt
        gt=deta-obj
        gt=randomGt(C)
        mt=beta1*mt+(1-beta1)*gt
        vt=beta2*vt+(1-beta2)*gt**2
        vt_=vt/(1-beta2**(i+1))
        if not plusMt:
            mt_=gt  #eliminate the influence of mt
        else:
            mt_=mt/(1-beta1**(i+1))
        step=mt_/sqrt(vt_+c)
        if i >= 10000:     
            deta=deta-alpha*step
            if plusPro:
                deta=deta/abs(deta) if abs(deta)>=1 else deta
            detas.append(deta)

    print('Final theta of Adam: %d'%detas[-1])
    return detas

def adamshiftmoving(obj,C,plusMt=True,plusPro=False,start=10000,stepnum=100000,alpha=0.001,beta1=0.9,beta2=0.999,keep_num=10,epsilon=1e-8):
    c=epsilon
    mt,vt=0,0
    deta=start
    detas,gts,gt_formers=[],[],[]

    seed(1)
    gtWindow = window(keep_num,beta1)
    for i in range(stepnum+10000):
        global gt
        gt=randomGt(C)
        if i < keep_num:
            gtWindow.append(gt)
            step = gtWindow.get_moving_average()
            gt_formers.append(0)
        else:
            gt_pre = gtWindow.windowArray[gtWindow.nextindex%gtWindow.length]
            gtWindow.append(gt)
            mt = gtWindow.get_moving_average()
            vt = beta2*vt + (1-beta2)*gt_pre**2
            vt_ = vt/(1-beta2**(i+1-keep_num))
            step = mt / sqrt(vt_+c)
        
        if i >= 10000:     
            deta = deta-alpha*step
            if plusPro:
                deta = deta/abs(deta) if abs(deta) >= 1 else deta
            detas.append(deta)
    print('Final theta of AdaShift: %d'%detas[-1])
    return detas,gts,gt_formers

def amsgrad(obj,C,start=10000,plusPro=False,plusMt=True,stepnum=100000,alpha=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8):
    c=epsilon
    mt,vt,vt_=0,0,0
    deta=start
    detas=[]

    seed(1)
    for i in range(stepnum+10000):
        global gt
        gt=deta-obj
        gt=randomGt(C)
        mt=beta1*mt+(1-beta1)*gt
        vt=beta2*vt+(1-beta2)*gt**2
        vt_=max(vt_,vt/(1-beta2**(i+1)))
        if not plusMt:
            mt_=gt  #eliminate the influence of mt
        else:
            mt_=mt/(1-beta1**(i+1))
        step=mt_/sqrt(vt_+c)
        
        if i >= 10000:                
            deta=deta-alpha*step
            if plusPro:
                deta=deta/abs(deta) if abs(deta)>=1 else deta

            detas.append(deta)

    print('Final theta of AMSgrad: %d'%detas[-1])
    return detas


if __name__=='__main__':
    obj=1
    alpha=0.001
    beta1=0.9
    beta2=0.999
    keep_num=10
    epsilon = 1e-8

    stepnum=10000000 
    start=0
    C=101
    plusMt=1
    plusPro=False


    detas1 = adam(obj,C,alpha=alpha,start=start,plusMt=plusMt,stepnum=stepnum,beta1=0.0,beta2=beta2,epsilon=epsilon)
    detas2 = adam(obj,C,alpha=alpha,start=start,plusMt=plusMt,stepnum=stepnum,beta1=0.9,beta2=beta2,epsilon=epsilon)
    # detas8 = adam(obj,C,alpha=alpha,start=start,plusMt=plusMt,stepnum=stepnum,beta1=0.999,beta2=beta2,epsilon=epsilon)
    # detas9 = adam(obj,C,alpha=alpha,start=start,plusMt=plusMt,stepnum=stepnum,beta1=0.9999,beta2=beta2,epsilon=epsilon)
    detas10 = adam(obj,C,alpha=alpha,start=start,plusMt=plusMt,stepnum=stepnum,beta1=0.9999,beta2=beta2,epsilon=epsilon)
    detas11 = adam(obj,C,alpha=alpha,start=start,plusMt=plusMt,stepnum=stepnum,beta1=0.9,beta2=0.9999,epsilon=epsilon)

    detas6 = amsgrad(obj,C,alpha=alpha,start=start,plusMt=plusMt,stepnum=stepnum,beta1=0.0,beta2=beta2,epsilon=epsilon)
    # detas7 = amsgrad(obj,C,alpha=alpha,start=start,plusMt=plusMt,stepnum=stepnum,beta1=0.9,beta2=beta2,epsilon=epsilon)

    detas3, gts1,gt_formers1 = adamshiftmoving(obj,C,alpha=alpha,start=start,plusMt=plusMt,stepnum=stepnum,beta1=0.0,beta2=beta2,keep_num=1, epsilon=epsilon)
    # detas4, gts2,gt_formers2 = adamshiftmoving(obj,C,alpha=alpha,start=start,plusMt=plusMt,stepnum=stepnum,beta1=0.9,beta2=beta2,keep_num=keep_num, epsilon=epsilon)
    # detas5, gts3,gt_formers3 = adamshiftmoving(obj,C,alpha=alpha,start=start,plusMt=plusMt,stepnum=stepnum,beta1=1.0,beta2=beta2,keep_num=keep_num, epsilon=epsilon)

    
    result1 = { 
        'Adam beta1:0.0': detas1,
        'AMSGrad beta1:0.0': detas6,
        'AdaShift n:1  beta1:0.0': detas3,
    }

    result2 = {
        'Adam beta1:0.9 beta2:0.999': detas2,
        'Adam beta1:0.9 beta2:0.9999': detas11,
        'Adam beta1:0.9999 beta2:0.999': detas10,
    }

    color_map={
        'Adam beta1:0.0':           '#009FCC',     
        'Adam beta1:0.9':           '#003C9D',  
        'AdaShift n:1  beta1:0.0':  '#FF3333',
        'AdaShift n:10 beta1:0.9':  '#CC0000',
        'AdaShift n:10 beta1:1.0':  '#880000',
        'AMSGrad beta1:0.0':        '#00DD00',
        'AMSGrad beta1:0.9':        '#008844',

        'Adam beta1:0.9 beta2:0.999':           '#33CCFF',  
        'Adam beta1:0.9999':                  '#0000CC',  
        'Adam beta1:0.9999 beta2:0.999':       '#CC0000',  
        'Adam beta1:0.9 beta2:0.9999':        '#220088'
    }

    
    figureNo=1
    figsize = (8,6)
    label_font = 14
    legend_prop = {'size': 14}
    plot_beg=0
    plot_stop=-1
    smooth_window=3
    ploy=3
    lw=1.5
    dotNum=1000
    X_label='iterations'
    Title='Stochastic Optimization' 

    plt.close('all')
    plt.figure(figureNo,figsize=figsize)
    plt.hlines(y=0,xmin=0,xmax=stepnum,linestyles='dashdot',lw=lw*0.3)
    result = result1
    for key in result.keys():
        x_range = np.arange(dotNum)*(len(result[key])//dotNum)
        content = np.mean(np.array(result[key]).reshape(dotNum,len(result[key])//dotNum),axis=1)
        plt.plot(x_range,content,lw=lw,color=color_map[key])

    plt.xlabel(X_label,fontsize = label_font)
    plt.ylabel(r'$ \theta_t $',fontsize = label_font)
    plt.ylim(-25,40)
    plt.title('Comparsion between Different Algorithms')
    plt.legend(result.keys(),loc='upper left',prop=legend_prop)
    plt.tight_layout()

    plt.figure(figureNo+1, figsize=figsize)
    plt.hlines(y=0, xmin=0, xmax=stepnum, linestyles='dashdot', lw=lw * 0.3)
    result = result2
    for key in result.keys():
        x_range = np.arange(dotNum) * (len(result[key]) // dotNum)
        content = np.mean(np.array(result[key]).reshape(dotNum, len(result[key]) // dotNum), axis=1)
        plt.plot(x_range, content, lw=lw, color=color_map[key])

    plt.xlabel(X_label, fontsize=label_font)
    plt.ylabel(r'$ \theta_t $', fontsize=label_font)
    plt.ylim(-25, 40)
    plt.title('Comparsion between Different beta')
    plt.legend(result.keys(), loc='upper left', prop=legend_prop)
    plt.tight_layout()

    plt.show()


