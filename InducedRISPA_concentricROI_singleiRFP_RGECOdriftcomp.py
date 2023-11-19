# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:30:13 2022

@author: Watabe
"""
import os, glob, sys,math,random,time,pathlib
sys.path.append(r"\\nikonti2\Users\NIS-Research\Python\PythonCodes")
from codeautosave import autosave
import cv2
#import tifffile
from nd2reader import ND2Reader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from matplotlib.colors import ListedColormap
from skimage.draw import disk, rectangle, circle_perimeter
from skimage.measure import label, regionprops
from skimage.registration import phase_cross_correlation
from PIL import Image,ImageDraw,ImageFont

from scipy.signal import butter, lfilter, freqz
def butter_lowpass_filter3d(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data,axis=0)
    return y


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a



mKOch=0
mKate2ch=1
iRFPch=2
GFPch=3

iRFPBG=3000
mKateBG=3000
mKOBG=2300

# mKOBG=np.array(Image.open(mKOBGpath),(np.uint16))
# mKateBG=np.array(Image.open(mKateBGpath),(np.uint16))
# iRFPBG=np.array(Image.open(iRFPBGpath),(np.uint16))

iRFPthreshold=2000
RatioThreshold=1.10
Percentile=50

RISPA_beforeFrame=10

BaseFolder=r"\\nikonti2\Jobs\jobsdb Projects\Watabe\RISPAinduce\20220119_200717_501"

suffix="analysis3"

seqdict={0:[3,6],
          1:[3,6],
          # 2:[3,6]}
          21:[2,6],}
#           22:[2,6]}

# seqdict={21:[2,6]}


for start in seqdict:

    resultdf=pd.DataFrame()
    EachSaveFolder=os.path.join(BaseFolder,f"Seq{str(start).zfill(4)}_{suffix}")
    os.makedirs(EachSaveFolder,exist_ok=True)

    print("\n\n----- ",start)
    for NthTime in range(seqdict[start][1],-1,-1):
        seq=start + NthTime*seqdict[start][0]
        
        # if seq==21:
        #     continue
        

        NDList=glob.glob(os.path.join(BaseFolder,f"*_Seq{str(seq).zfill(4)}.nd2"))
        if len(NDList)!=1:
            print("ERROR: ",seq)
            continue

        filename=NDList[0]
        print(filename)

        ND=ND2Reader(filename)

        Onepixel_micron=ND.metadata["pixel_microns"]
        
        for Nthfield in range(ND.sizes["v"]):
            
            # if Nthfield!=1:
            #     continue
            
            EachFieldFolder=os.path.join(EachSaveFolder,str(Nthfield))
            os.makedirs(EachFieldFolder,exist_ok=True)
        
            print("field: ",Nthfield)

            mKOiRFPRatiostack=np.zeros([ND.sizes["y"],ND.sizes["x"],ND.sizes["t"]],dtype=np.float32)
            iRFPstack=np.zeros([ND.sizes["y"],ND.sizes["x"],ND.sizes["t"]],dtype=np.uint16)

            shift_list=[]

            first=True
            for Nthframe in range(ND.sizes["t"]):
                mKOsubBG=cv2.subtract(ND.get_frame_2D(c=mKOch,t=Nthframe,v=Nthfield), mKOBG)
                iRFPsubBG=cv2.subtract(ND.get_frame_2D(c=iRFPch,t=0,v=Nthfield), iRFPBG)
                
                mKOiRFPRatiostack[:,:,Nthframe]=mKOsubBG/iRFPsubBG
                iRFPstack[:,:,Nthframe]=iRFPsubBG

                if first==True:
                    first=False
                    FirstmKO=mKOsubBG.copy()

                shift, error, diffphase = phase_cross_correlation(FirstmKO, mKOsubBG)
                shift_list.append(shift)


            iRFPmin=np.amin(iRFPstack,axis=2)
            iRFPmax=np.amax(iRFPstack,axis=2)

            iRFPbinary=(iRFPmin>iRFPthreshold)

            iRFP_RevBinary=cv2.erode(np.uint8(iRFPmax<iRFPthreshold),
                                     kernel=np.ones([3,3]),iterations=2)

            iRFP_label_img = label(iRFPbinary)
            iRFP_regions = regionprops(iRFP_label_img)

            iRFP_erode_regionList=[]
            
            iRFPerodebase=np.zeros(iRFPbinary.shape,dtype=np.uint8)
            
            NthRegion=0
            for eachiRFP in range(1,np.max(iRFP_label_img)+1):
                eachiRFPimg=np.uint8(iRFP_label_img==eachiRFP)
                eachErodeimg=cv2.erode(eachiRFPimg,kernel=np.ones([3,3]),iterations=1)
                        
                ## if erode filter separates single ROI into more than two,
                ## this procedure will deteriorate the analysis
                if eachErodeimg.sum() < 2:
                    continue
                else:
                    NthRegion+=1
                
                if eachErodeimg.sum() > 10:
                    while True:
                        # print(eachErodeimg.sum())
                        ErodeErodeimg=cv2.erode(eachErodeimg,kernel=np.ones([3,3]),iterations=1)
                        if ErodeErodeimg.sum()<10:
                            break
                        else:
                            eachErodeimg=ErodeErodeimg

                iRFPerodebase[eachErodeimg==1]=NthRegion
                
                # eachErode_label_img = label(eachErodeimg)
                # eachErode_regions=regionprops(eachErode_label_img)
                

            iRFP_erode_regionList=regionprops(iRFPerodebase)



            # iRFPregion=iRFP_erode_regionList[10]

            # for Nthframe in range(ND.sizes["t"]):
            #     mKOsubBG=cv2.subtract(ND.get_frame_2D(c=mKOch,t=Nthframe,v=Nthfield), mKOBG)
            #     y, x = iRFPregion.coords[:,0], iRFPregion.coords[:,1]
            #     y_shift=np.uint16(y - shift_list[Nthframe][0])
            #     x_shift=np.uint16(x - shift_list[Nthframe][1])
                
            #     plt.imshow(mKOsubBG[328:378,473:523],vmin=0,vmax=15000,cmap="gray")#[328:378,473:523]
            #     plt.show()
            #     mKOsubBG[y_shift,x_shift]=65535
            #     plt.imshow(mKOsubBG[328:378,473:523],vmin=0,vmax=15000,cmap="gray")#[328:378,473:523]
            #     plt.show()
                # break
                


            NthRegion=0
            for iRFPregion in iRFP_erode_regionList:
                
                NthRegion+=1
                
                y, x = iRFPregion.coords[:,0], iRFPregion.coords[:,1]
                RGECOratio=np.zeros(ND.sizes["t"])
 
                for Nthframe in range(ND.sizes["t"]):
                    y_shift=np.uint16(y - shift_list[Nthframe][0])
                    x_shift=np.uint16(x - shift_list[Nthframe][1])
                    goodshift_x=[]
                    goodshift_y=[]
                    
                    for nthxy in range(len(y_shift)):
                        if y_shift[nthxy]<ND.sizes["y"] and x_shift[nthxy]<ND.sizes["x"]:
                            goodshift_x.append(x_shift[nthxy])
                            goodshift_y.append(y_shift[nthxy])
                    
                    goodshift_x=np.array(goodshift_x)
                    goodshift_y=np.array(goodshift_y)
                    
                    if len(goodshift_x)==0:
                        goodshift_x=x
                        goodshift_y=y
                    
                    RGECOratio[Nthframe]=np.median(mKOiRFPRatiostack[goodshift_y,goodshift_x,Nthframe])
                
                # plt.plot(RGECOratio,"k-")
                # savename=os.path.join(EachFieldFolder,str(seq).zfill(4)+"Region_"+str(NthRegion).zfill(2)+"_RGECO_driftcomp"+".png")
                # plt.savefig(savename,dpi=300,bbox_inches="tight")
                # # plt.show()
                # plt.close();plt.clf()

                np.save(os.path.join(EachFieldFolder,str(seq).zfill(4)+"Region_"+str(NthRegion).zfill(2)+"_RGECOratio_driftcomp"),RGECOratio)