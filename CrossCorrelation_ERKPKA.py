# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 00:10:31 2022

@author: Watabe
"""

import os, glob, sys,math,random
sys.path.append(r"\\nikonti2\Users\NIS-Research\Python\PythonCodes")
import cv2
from graphfunc_watabe import cm2inch, pltsetting, pdf_font
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import signal
pdf_font(plt)

dfpath=r"\\Nikonti2\Users\NIS-Research\Documents\Users\watabe\20211204\LargeMDCK2-9_CrossCorrelation2\CYratio_BoosterRatio.csv"
df=pd.read_csv(dfpath)

resdf=pd.DataFrame()

index=0
for NewLabel in df["NewLabel"].unique():
    Eachdf=df[df["NewLabel"]==NewLabel]
    ERK=Eachdf["CYRatio"]
    PKA=Eachdf["BoosterRatio"]
    lags = signal.correlation_lags(len(ERK), len(PKA))

    # ERK=(ERK-ERK.min())/(ERK.max()-ERK.min())
    # PKA=(PKA-PKA.min())/(PKA.max()-PKA.min())

    ERK=ERK-ERK.mean()
    PKA=PKA-PKA.mean()

    corr=signal.correlate(ERK,PKA, mode='full', method='auto')

    # corr /= np.max(corr)
    # a=sm.tsa.stattools.ccf(PKA,ERK, adjusted=False)


    if index==0:
        res=np.zeros([len(df["NewLabel"].unique()),len(corr)])

    res[index,:]=corr
    index+=1


X=120
res2=res[:,X:-X]
lags2=lags[X:-X]

fig=plt.figure(figsize=(2,1.5))
plt.fill_between(lags2,res2.mean(axis=0)-res2.std(axis=0),
                 res2.mean(axis=0)+res2.std(axis=0),
                 color="cyan")
plt.plot(lags2,res2.mean(axis=0),"k-")

plt.plot([lags[res.mean(axis=0)==res.mean(axis=0).max()],
              lags[res.mean(axis=0)==res.mean(axis=0).max()]],
         [-0.16,0.22],"r--")

lag_min=int(lags[res.mean(axis=0)==res.mean(axis=0).max()])
plt.text(x=0,y=0.25,s=str(lag_min)+" min")

savepath=dfpath[:-4]+".pdf"
plt.xticks([-30,0,30])
#plt.yticks([-0.1,0,0.1,0.2])
pltsetting(plt,plt.gca())
plt.savefig(savepath)
print(savepath)
plt.show()


# test=pd.read_csv(r"E:\desktop\test.csv")
# sin=test["sin"]
# peak=test["peak"]

# sin=sin-sin.mean()
# peak=peak-peak.mean()


# corr=signal.correlate(sin, peak, mode='full', method='auto')
# corr /= np.max(corr)
# lags = signal.correlation_lags(len(sin), len(peak))

# plt.plot(sin,"g:")
# plt.plot(peak,"k")
# plt.show()
# plt.plot(lags,corr)
# plt.show()
# print(lags[corr==corr.max()][0])
# #define data
# marketing = np.array([3, 4, 5, 5, 7, 9, 13, 15, 12, 10, 8, 8])
# revenue = np.array([21, 19, 22, 24, 25, 29, 30, 34, 37, 40, 35, 30])



# #calculate cross correlation
# cc=sm.tsa.stattools.ccf(marketing, revenue, adjusted=False)

# plt.plot(marketing)
# plt.show()
# plt.plot(revenue)
# plt.show()
# plt.plot(cc)
# plt.show()