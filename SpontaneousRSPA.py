# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 03:16:09 2022

@author: Watabe
"""

import os
import cv2
from nd2reader import ND2Reader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.draw import circle_perimeter
from skimage.measure import label, regionprops

def main():
    NDPathList=[
            r"\\nikonti2\Jobs\jobsdb Projects\Watabe\MultiMode20211207\20211207_183109_666\ChannelCFP2,FRET_2,mKOk_bin2,mKateFRET_bin2,4ｘTransBin4x_Seq0010.nd2",
            r"\\nikonti2\Jobs\jobsdb Projects\Watabe\MultiMode20211207\20220118_132237_527\ChannelCFP2,FRET_2,mKOk_bin2,mKateFRET_bin2,4ｘTransBin4x_Seq0000.nd2",
            r"F:\watabe\20220201\Density.nd2",
            r"\\NIKONTI2\Users\NIS-Research\Documents\Users\watabe\20220118\MDCK2-9inhibitors.nd2",
            r"F:\watabe\20220131\ERKPKA2-9_Inhibitors.nd2",
            r"F:\watabe\20220201\Inhibitors.nd2",
            r"\\NIKONTI2\F\watabe\20220313\ERKPKAdrug.nd2"
            ]

    SaveFolder_suffix_list=[
                        "Density_FixedThreshold130",
                        "Density_FixedThreshold130",
                        "Density_FixedThreshold130",
                        "Drug_FixedThreshold130",
                        "Drug_FixedThreshold130",
                        "Drug_FixedThreshold130",
                        "Drug_FixedThreshold130"]

    for i in range(7):
        NDPath=NDPathList[i]
        SaveFolder_suffix=SaveFolder_suffix_list[i]
        main2(NDPath,SaveFolder_suffix)

def main2(NDPath,SaveFolder_suffix):
    mKOBG=600
    mKateBG=700

    mKOthreshold_afterBG=400
    mKOthreshold_afterBG_High=65535-mKOthreshold_afterBG
    
    kernel = np.ones((11,11),np.uint8)
    mKOch=2
    mKate2ch=3
    
    ColMp=cm.bwr
    ColMp.set_under(color="k")

    FrameInterval=1
    
    Normalization_Frames=20
    
    Threshold=1.3    
    ########################## parameters ##########################
    
           
    SaveFolder=NDPath[:-4]+SaveFolder_suffix
    os.makedirs(SaveFolder,exist_ok=True)
    print(SaveFolder)
    RawRatioSaveFolder=os.path.join(SaveFolder,"RawRatio")
    os.makedirs(RawRatioSaveFolder,exist_ok=True)
    ND=ND2Reader(NDPath)
    Onepixel_per_micron=ND.metadata["pixel_microns"]
    OnemmPixel=int(1000/Onepixel_per_micron)
    
    grid_y, grid_x = np.mgrid[0:ND.sizes["y"]:1,0:ND.sizes["x"]:1]
    
    for eacharea in range(ND.sizes["v"]):
    
        EachAreaSaveFolder=os.path.join(RawRatioSaveFolder,str(eacharea))
        os.makedirs(EachAreaSaveFolder,exist_ok=True)
        
        FrameRead=0
        first=True
        
        df=pd.DataFrame()
    
        
        for Nthframe in range(ND.sizes["t"]):
            
            mKOsubBG=cv2.subtract(ND.get_frame_2D(c=mKOch,t=Nthframe,v=eacharea), mKOBG)
            mKatesubBG=cv2.subtract(ND.get_frame_2D(c=mKate2ch,t=Nthframe,v=eacharea), mKateBG)
            KOmedian = cv2.medianBlur(mKOsubBG,ksize=3)
            Katemedian = cv2.medianBlur(mKatesubBG,ksize=3)
            
            BoosterRatio=(Katemedian/KOmedian)
            BoosterRatio=BoosterRatio*(KOmedian>mKOthreshold_afterBG)*(KOmedian<mKOthreshold_afterBG_High)
            
            np.nan_to_num(BoosterRatio, copy=False)
            
            if first:
                first=False
                BeforeFrameArray=np.zeros([mKOsubBG.shape[0],
                                           mKOsubBG.shape[1],
                                           Normalization_Frames],dtype=float)
                
            BeforeFrameArray[:,:,FrameRead%Normalization_Frames]=BoosterRatio
            
            FrameRead+=1
            
            if FrameRead<Normalization_Frames:
                continue
            
            BoosterRatio_Denominator=np.amin(BeforeFrameArray,2)  
            
            NormalizedBoosterRatio=(BoosterRatio/BoosterRatio_Denominator)*(BoosterRatio>0.3)
            NormalizedBoosterRatio[NormalizedBoosterRatio==np.inf]=np.nan
            NormalizedBoosterRatio=np.nan_to_num(NormalizedBoosterRatio)
    
            median=cv2.GaussianBlur((NormalizedBoosterRatio*100).astype(np.uint8),(21,21),0)
            blur = cv2.GaussianBlur(median/100,(21,21),0)
            
            
            IM=(blur>Threshold).astype(np.uint8)
            IM2 = cv2.morphologyEx(IM, cv2.MORPH_OPEN, kernel)
            IM3 = cv2.morphologyEx(IM2, cv2.MORPH_CLOSE, kernel)
     
            
            label_img = label(IM3)
            regions = regionprops(label_img)
            
            BoosterRatio2=NormalizedBoosterRatio.copy()
            
            for eachregion in regions:
                
                x=eachregion.centroid[1]
                y=eachregion.centroid[0]
                
                diameter=eachregion.equivalent_diameter
                
                yr,xr=circle_perimeter(int(y),int(x),radius=int(diameter/2),
                                       method= "bresenham",shape=IM3.shape)
                
                BoosterRatio2[yr,xr]+=100000
                
                df=df.append({"Frame":Nthframe,
                              "Label":eachregion.label,
                              "y":y,
                              "x":x,
                              "equivalent_diameter":eachregion.equivalent_diameter,
                              },ignore_index=True)

                
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.axis("off")
            ax1.plot([IM3.shape[0]*0.95-OnemmPixel/2,IM3.shape[0]*0.95],[IM3.shape[1]*0.95,IM3.shape[1]*0.95], color='black', linestyle='-')
            ax1.text(10, -300, str(Nthframe*FrameInterval)+" min")
            ax1.text(IM3.shape[0]*0.75, IM3.shape[1]*1.1, "500 um")
            plt.imshow(NormalizedBoosterRatio,vmin=0.9,vmax=1.4*2,cmap="bwr")
        
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.axis("off")
            plt.imshow(BoosterRatio2,vmin=0.9,vmax=1.4*2,cmap="bwr")
            
            fig.savefig(os.path.join(EachAreaSaveFolder,"CheckTheCircle_"+str(Nthframe).zfill(3)+".png"),dpi=200,
                        bbox_inches="tight")

            plt.clf()
            plt.close()
        
        df.to_csv(os.path.join(SaveFolder,f"{str(eacharea).zfill(3)}ResultDF.csv"))


if __name__ == '__main__':
    main()

    print("END")