# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 09:12:40 2022

@author: NIS-Research
"""

## Check the nearest center of RISPA
##
import os, glob, sys,math,random,time,pathlib
sys.path.append(r"\\nikonti2\Users\NIS-Research\Python\PythonCodes")
import pandas as pd
from codeautosave import autosave
from nd2reader import ND2Reader
import numpy as np


def main():
    
    BaseFolder=r"\\Nikonti2\Jobs\jobsdb Projects\Watabe\RISPAinduce\20220119_200717_501"
    suffix="analysis3"
    threshold_length_pixel=30

    seqdict={0:[3,6],
              1:[3,6],
              21:[2,6]}

    for start in seqdict:
        
        EachSaveFolder=os.path.join(BaseFolder,f"Seq{str(start).zfill(4)}_{suffix}")
        ResultDFsavePath=os.path.join(EachSaveFolder,"ResultDF.csv")
                
        assignNewLabel(ResultDFsavePath,threshold_length_pixel)
    

def assignNewLabel(csvpath,threshold_length_pixel):
    
    ResultDF=pd.read_csv(csvpath,index_col=0)
    for Nthfield in ResultDF["Nthfield"].unique():
        print("Nthfield ",Nthfield)
    
        FirstFrameDF=ResultDF[(ResultDF["Nthframe"]==0)&(ResultDF["Nthfield"]==Nthfield)]
    
        FirstTime=int(FirstFrameDF["NthTime"].min())
        LastTime=int(FirstFrameDF["NthTime"].max())
    
        Center_analyzeDF=pd.DataFrame()
       
        for NthTime in range(FirstTime,LastTime):
            BaseDF=FirstFrameDF[FirstFrameDF["NthTime"]==NthTime]
            NextDF=FirstFrameDF[FirstFrameDF["NthTime"]==NthTime+1]
            
            for BaseRegion in BaseDF["regionnum"].unique():
                base=BaseDF[BaseDF["regionnum"]==BaseRegion]
                x_center=base["x"].values[0]
                y_center=base["y"].values[0]

                # if abs(ImageSize[1]/2 - x_center)>ImageSize[1]/2-IgnoreEdge_pixel:
                #     continue
                # elif abs(ImageSize[0]/2 - y_center)>ImageSize[0]/2-IgnoreEdge_pixel:
                #     continue       
        
                for CandidateRegion in NextDF["regionnum"].unique():
                    Next=NextDF[(NextDF["regionnum"]==CandidateRegion)]
                    Next_x_center=Next["x"].values[0]
                    Next_y_center=Next["y"].values[0]
                    
                    distance=((x_center-Next_x_center)**2 + (y_center-Next_y_center)**2)**0.5
                    
                    if distance < threshold_length_pixel:
                        Center_analyzeDF=Center_analyzeDF.append({
                                                            "NthTime":NthTime,
                                                            "regionnum":BaseRegion,
                                                            "Nextregionnum":CandidateRegion,
                                                            "distance":distance,
                                                            "y_center":y_center,
                                                            "x_center":x_center,
                                                            },ignore_index=True)
                        
        ##### Duplication check    
    
        
        NumberOfError=0
        for NthTime in range(FirstTime,LastTime):
            NthDF=Center_analyzeDF[Center_analyzeDF["NthTime"]==NthTime]
            
            for EachRegionLabel in NthDF["regionnum"].unique():
                if (len(NthDF[NthDF["regionnum"]==EachRegionLabel]))>1:
                    # print("Check the center of RISPA. -- 1")
                    # print(NthDF[NthDF["regionnum"]==EachRegionLabel],"\n")
        
                    DeleteIndexlist=list(NthDF[NthDF["regionnum"]==EachRegionLabel].index)
                    preserve_index=NthDF[NthDF["regionnum"]==EachRegionLabel]["distance"].idxmin()
        
                    DeleteIndexlist.remove(preserve_index)
                    Center_analyzeDF.drop(DeleteIndexlist, axis=0, inplace=True)
                    NumberOfError+=1
        
            ## Re-define of NthDF is MUST, because even after the dropping procedure has changed the Center_analyzeDF NthDF do not change.
            NthDF=Center_analyzeDF[Center_analyzeDF["NthTime"]==NthTime]
            for EachRegionLabel in NthDF["Nextregionnum"].unique():
                if (len(NthDF[NthDF["Nextregionnum"]==EachRegionLabel]))>1:
                    # print("Check the center of RISPA. -- 2")
                    # print(NthDF[NthDF["Nextregionnum"]==EachRegionLabel],"\n")
        
                    DeleteIndexlist=list(NthDF[NthDF["Nextregionnum"]==EachRegionLabel].index)
                    preserve_index=NthDF[NthDF["Nextregionnum"]==EachRegionLabel]["distance"].idxmin()
        
                    DeleteIndexlist.remove(preserve_index)
                    Center_analyzeDF.drop(DeleteIndexlist, axis=0, inplace=True)
                    NumberOfError+=1
        
        
        print(Center_analyzeDF)
        print(f"The number of detected duplicated counts: {NumberOfError}")
        
        FirstFrameDF.at[:,"Newregionnum"]=0
        FirstFrameDF.at[:,"continuity"]=0
        
        ## Assigning new label
        Newregionnum=0
        for i,items in Center_analyzeDF.iterrows():
            NthTime = Center_analyzeDF.at[i,"NthTime"]
            RegionNum = Center_analyzeDF.at[i,"regionnum"]
            NextRegionNum = Center_analyzeDF.at[i,"Nextregionnum"]
                
            NthTimeResultDF = FirstFrameDF[(FirstFrameDF["NthTime"]==NthTime)&(FirstFrameDF["regionnum"]==RegionNum)]
            NplusOneTimeResultDF = FirstFrameDF[(FirstFrameDF["NthTime"]==NthTime+1)&(FirstFrameDF["regionnum"]==NextRegionNum)]
            
            # print(" ",len(NthTimeResultDF),len(NplusOneTimeResultDF))
            if len(NthTimeResultDF)*len(NplusOneTimeResultDF)!=1:
                print(f"ERROR, NthTime={NthTime}, regionnum={RegionNum}")
                continue
    
    
            if FirstFrameDF.at[NthTimeResultDF.index[0],"Newregionnum"]==0:
                if NthTime==FirstTime:
                    Newregionnum+=1
                    FirstFrameDF.at[NthTimeResultDF.index[0],"Newregionnum"]=Newregionnum
                else:
                    continue
    
            Newregionnum=FirstFrameDF.at[NthTimeResultDF.index[0],"Newregionnum"]
            FirstFrameDF.at[NplusOneTimeResultDF.index[0],"Newregionnum"]=Newregionnum
    
    
        #Check whether each region is detected through the timecourse
        for Newregionnum in FirstFrameDF["Newregionnum"].unique():
    
            continuity=len(FirstFrameDF[FirstFrameDF["Newregionnum"]==Newregionnum])
            FirstFrameDF.loc[FirstFrameDF[FirstFrameDF["Newregionnum"]==Newregionnum].index,"continuity"]=continuity
    
        # ResultDF.at[:,"Newregionnum"]=0
        for i,items in FirstFrameDF.iterrows():
            NthTime = FirstFrameDF.at[i,"NthTime"]
            RegionNum = FirstFrameDF.at[i,"regionnum"]
            Nthfield = FirstFrameDF.at[i,"Nthfield"]
            Newregionnum=FirstFrameDF.at[i,"Newregionnum"]
            continuity=FirstFrameDF.at[i,"continuity"]
        
            ChangeIndex=ResultDF[(ResultDF["NthTime"]==NthTime)&(ResultDF["regionnum"]==RegionNum)&(ResultDF["Nthfield"]==Nthfield)].index
            ResultDF.loc[ChangeIndex,"Newregionnum"]=Newregionnum
            ResultDF.loc[ChangeIndex,"continuity"]=continuity
        # ResultDF[(FirstFrameDF["NthTime"]==NthTime)&(FirstFrameDF["regionnum"]==RegionNum)]
    
    
    
    print("Finish assigning new label")
    
    savepath=csvpath[:-4]+"newregion_assigned.csv"
    ResultDF.to_csv(savepath)
    
if __name__ == '__main__':
    main()
    autosave(__file__)