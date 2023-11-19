# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 09:53:30 2022

@author: NIS-Research
"""

import os, glob, sys,math,random,time,pathlib
sys.path.append(r"\\nikonti2\Users\NIS-Research\Python\PythonCodes")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from codeautosave import autosave
# BaseFolder=r"\\NIKONTI2\Users\NIS-Research\Documents\Users\watabe\20211204\PKA-BoosterAMPK001analysis"


def main():
    # BaseFolder=r"\\nikonti2\F\job\t_watabe Projects\Watabe\RISPAinduce\20220129_113942_125"
    # suffix="analysis3"
    # suffix2="newregion_assigned.csv"
    # threshold_length_pixel=30
    
    # seqdict={0:[5,6],
    #           1:[5,6],
    #           2:[5,6],
    #           3:[5,6],
    #           4:[5,6]}


    BaseFolder=r"\\Nikonti2\Jobs\jobsdb Projects\Watabe\RISPAinduce\20220119_200717_501"
    suffix="analysis3"
    suffix2="newregion_assigned.csv"
    threshold_length_um=27

    seqdict={0:[3,6],
              1:[3,6],
              2:[3,6],
              21:[2,6],
              22:[2,6]}

    for start in seqdict:
        EachSaveFolder=os.path.join(BaseFolder,f"Seq{str(start).zfill(4)}_{suffix}")
        ResultDFsavePath=os.path.join(EachSaveFolder,"ResultDF.csv")
        
        graphsavepath=os.path.join(EachSaveFolder,"graph_FperF0")
        graphexport(ResultDFsavePath,suffix2,threshold_length_um,graphsavepath)



def graphexport(csvpath,suffix,threshold_length_um,graphsavepath):

    csvpath_suffix=csvpath[:-4]+suffix
    ResultDF=pd.read_csv(csvpath_suffix,index_col=0)
    os.makedirs(graphsavepath,exist_ok=True)

    
    for Nthfield in ResultDF["Nthfield"].unique():
        print("Nthfield ",Nthfield)
    
        NthFieldDF=ResultDF[ResultDF["Nthfield"]==Nthfield]

        ROIlist=NthFieldDF["Newregionnum"].unique()
    
        # print(RIPSApositiveROIlist)
    
        for Newregionnum in ROIlist:
            if Newregionnum==0:
                continue
            RegionDF=NthFieldDF[NthFieldDF["Newregionnum"]==Newregionnum]
            RegionDF["NthTimestr"]=RegionDF["NthTime"].astype(str)
            
            ModifiedDF=pd.DataFrame()
            for seq in RegionDF["seq"].unique():
                EachSeqDF=RegionDF[RegionDF["seq"]==seq]
                if EachSeqDF["Rad_um"].max()<threshold_length_um:
                    EachSeqDF["Radius_um"]=0
                else:
                    EachSeqDF["Radius_um"]=EachSeqDF["Rad_um"]
                ModifiedDF=ModifiedDF.append(EachSeqDF)
                
            SeqNum=str(RegionDF["seq"].min()).zfill(4)
            
            sns.lineplot(x="Nthframe",y="RGECOratio",hue="NthTimestr",data=ModifiedDF)
            plt.title(f"RGECO  Seq{SeqNum}  Field: {int(Nthfield)}  Region: {int(Newregionnum)}")
    
            filename=os.path.join(graphsavepath,SeqNum+"_"+str(Nthfield)+"_"+str(int(Newregionnum))+"_RGECO.png")
            plt.savefig(filename,dpi=300,bbox_inches="tight")
            plt.show()

            sns.lineplot(x="Nthframe",y="Radius_um",hue="NthTimestr",data=ModifiedDF[ModifiedDF["Nthframe"]>0])
   
            plt.title(f"RISPA radius  Seq{SeqNum}  Field: {int(Nthfield)}  Region: {int(Newregionnum)}")
    
            filename=os.path.join(graphsavepath,SeqNum+"_"+str(Nthfield)+"_"+str(int(Newregionnum))+"_RISPA.png")
            plt.savefig(filename,dpi=300,bbox_inches="tight")
            plt.show()

    
if __name__ == '__main__':
    d=main()
    autosave(__file__)