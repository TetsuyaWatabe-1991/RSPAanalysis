# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 17:54:37 2022

@author: Watabe
"""


import os, glob, sys,math,random
sys.path.append(r"C:\Users\NIS-Research\Python\PythonCodes")
import cv2
#import tifffile
from nd2reader import ND2Reader
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from PIL import Image
from matplotlib.colors import ListedColormap
from skimage.draw import circle, rectangle, circle_perimeter

mKOBG=1600
mKateBG=1600
CFPBG=2000
FRETBG=1200
mKOthreshold_afterBG=1000
mKOthreshold_afterBG_High=65535-mKOthreshold_afterBG
CFPthreshold_afterBG=1500


mKOch=2
mKate2ch=3
CFPch=0
FRETch=1

CYframeinterval=5

CY_Ratio_diameter_um=20

remove_confinement_frame=80

AnalyzeFrameWindow=150



NDPath=r"\\NIKONTI2\Users\NIS-Research\Documents\Users\watabe\20211204\LargeMDCK2-9_.nd2"

SaveFolder_suffix=f"CrossCorrelation_diameter_{CY_Ratio_diameter_um}"

DFpath=r"\\Nikonti2\Users\NIS-Research\Documents\Users\watabe\20211204\LargeMDCK2-9_test\RegionDFwithNewLabel.csv"


RegionDFwithNewLabel=pd.read_csv(DFpath)

PeakDF=RegionDFwithNewLabel[RegionDFwithNewLabel["Peak"]==1]
wellname=NDPath[NDPath.find("Well")+4:NDPath.find("_Channel")]
print(wellname)

   
SaveFolder=NDPath[:-4]+SaveFolder_suffix
os.makedirs(SaveFolder,exist_ok=True)

ND=ND2Reader(NDPath)
Onepixel_micron=ND.metadata["pixel_microns"]

CY_Ratio_radius_pixel=0.5*(CY_Ratio_diameter_um/Onepixel_micron)


df=pd.DataFrame()

for index,items in PeakDF.iterrows():
    y=PeakDF.at[index,"y"]
    x=PeakDF.at[index,"x"]
    NewLabel=PeakDF.at[index,"NewLabel"]
    frame=PeakDF.at[index,"Frame"]

    startframe=(int(frame-AnalyzeFrameWindow/2)//CYframeinterval)*CYframeinterval
    endframe=(1+int(frame+AnalyzeFrameWindow/2)//CYframeinterval)*CYframeinterval

    if remove_confinement_frame>startframe or endframe>ND.sizes["t"]:
        continue

    yr,xr=circle(y,x,CY_Ratio_radius_pixel,ND.frame_shape)

    for Nthframe in range(startframe,endframe):
    
        CYframe=(Nthframe//CYframeinterval)*CYframeinterval
        
        print(Nthframe)

        if Nthframe==CYframe:

            CFPsubBG=cv2.subtract(ND.get_frame_2D(c=CFPch,z=CYframe), CFPBG)
            FRETsubBG=cv2.subtract(ND.get_frame_2D(c=FRETch,z=CYframe), FRETBG)

            CFPintensity=CFPsubBG[yr,xr].sum()
            FRETintensity=FRETsubBG[yr,xr].sum()

            CYRatio=(FRETintensity/CFPintensity)


        mKOsubBG=cv2.subtract(ND.get_frame_2D(c=mKOch,z=Nthframe), mKOBG)
        mKatesubBG=cv2.subtract(ND.get_frame_2D(c=mKate2ch,z=Nthframe), mKateBG)

        mKOintensity=mKOsubBG[yr,xr].sum()
        mKateTintensity=mKatesubBG[yr,xr].sum()

        BoosterRatio=(mKateTintensity/mKOintensity)


        df=df.append({
                    "NewLabel":NewLabel,
                    "Frame":Nthframe,
                    "CYRatio":CYRatio,
                    "BoosterRatio":BoosterRatio
                    },ignore_index=True)


    plotdf=df[df["NewLabel"]==NewLabel]

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    plt.plot(plotdf["Frame"],plotdf["CYRatio"])

    ax2 = fig.add_subplot(2, 1, 2)
    plt.plot(plotdf["Frame"],plotdf["BoosterRatio"])

    figpath=os.path.join(SaveFolder,str(NewLabel).zfill(4)+".png")

    fig.savefig(figpath,dpi=150, bbox_inches="tight")

    plt.clf();plt.close();

df.to_csv(os.path.join(SaveFolder,"CYratio_BoosterRatio.csv"))

# df.to_csv(os.path.join(SaveFolder,"ResultDF.csv"))
# randomdf.to_csv(os.path.join(SaveFolder,"RandomDF.csv"))

# print("END")