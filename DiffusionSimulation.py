# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 14:43:10 2021

@author: Watabe
"""

import os, sys, datetime,time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import seaborn as sns
from numba import jit


def main():
    start = time.time()
    
    Zlength=1600
    RadiusLength=1600
    DifCo=400

    dz=10
    dr=10
    dt=0.05
    
    TotalTime=60*10
    
    InitialBinFromOrigin_Z,InitialBinFromOrigin_R=1,1
    
    ConcThrshold_nM=1
    
    
    TotalPGE_fmol_list = [0.001*60]
    
    savefolder=r"E:\OLDDatas\202109\20210925_python_simulation"
    GetDataEveryNsec=2
    
    IncreasingDuration_Sec,PlateauDuration_Sec,DecreasingDuration_Sec=0,60,30
    
    now = datetime.datetime.now()
    filename_base = os.path.join(savefolder, "Diffusion_"+now.strftime('%Y%m%d_%H%M%S'))
    Num_Z,Num_R,Num_T=Coordination_check(RadiusLength,Zlength,TotalTime,dr,dz,dt)
    InitialVolume,EachDonut,EachVolume,EachRadiusPosition=GetCoordinateParams(RadiusLength,dr,dz,InitialBinFromOrigin_Z,
                                                                              InitialBinFromOrigin_R,Num_R,Num_Z)
    resultdf=pd.DataFrame()
    PGEpattern=pd.DataFrame()
    

    
    for TotalPGE_fmol in TotalPGE_fmol_list:  
        print(f"PGE2 (fmol): {TotalPGE_fmol}")
        Temp_ZR=np.zeros([Num_Z+2,Num_R+2])
        
        resultdf,PGEpattern=SpecialFunctionPGE2(Num_T,dz,dt,dr,DifCo,IncreasingDuration_Sec,PlateauDuration_Sec,DecreasingDuration_Sec,
                        TotalPGE_fmol,RadiusLength,Num_R,EachVolume,ConcThrshold_nM,PGEpattern,
                        EachRadiusPosition,InitialVolume,Temp_ZR,InitialBinFromOrigin_Z,InitialBinFromOrigin_R,GetDataEveryNsec,resultdf)
  
    print ("elapsed_time:{0}".format(time.time() - start) + "[sec]")
    
    diffusion_to_CSV(resultdf,filename_base)
    
    plot_PGE_sumTotal(PGEpattern,filename_base)
    plot_PGEpattern(PGEpattern,filename_base)
    plot_radius(resultdf,filename_base,ConcThrshold_nM)    
    
    
    
    ##Valables used in this simulation will be saved
    f = open(filename_base+"_condition.txt", 'w', encoding='utf-8', newline='\n')
    resulttext=""
    
    for variable_name, value in locals().items():
        if type(value) not in [int, float,str]:
            if type(value)==list:
                if len(value)<10:
                    f.write(f"{variable_name},{value}\n")
        else:
            f.write(f"{variable_name},{value}\n")
    f.close()

    
    
    

def SpecialFunctionPGE2(Num_T,dz,dt,dr,DifCo,IncreasingDuration_Sec,PlateauDuration_Sec,DecreasingDuration_Sec,
                        TotalPGE_fmol,RadiusLength,Num_R,EachVolume,ConcThrshold_nM,PGEpattern,
                        EachRadiusPosition,InitialVolume,Temp_ZR,InitialBinFromOrigin_Z,InitialBinFromOrigin_R,GetDataEveryNsec,resultdf):

    PlateauPGE_PerSec=2*(TotalPGE_fmol)/(IncreasingDuration_Sec+2*PlateauDuration_Sec+DecreasingDuration_Sec)
    
    TotalReleasingDuration_Sec=IncreasingDuration_Sec+PlateauDuration_Sec+DecreasingDuration_Sec
    print("TotalReleasingDuration_Sec",TotalReleasingDuration_Sec)
    
    Released_TotalPGE2=0
    
    for t in range(Num_T):

        
        if t*dt<TotalReleasingDuration_Sec:
            if t*dt<IncreasingDuration_Sec:
                delta_PGE_Amount= (PlateauPGE_PerSec/IncreasingDuration_Sec)*(t*dt)*dt
            elif t*dt<IncreasingDuration_Sec+PlateauDuration_Sec:
                delta_PGE_Amount= (PlateauPGE_PerSec)*dt
            else:
                delta_PGE_Amount= (-(PlateauPGE_PerSec/(DecreasingDuration_Sec))*(t*dt)+\
                    PlateauPGE_PerSec*(TotalReleasingDuration_Sec)/DecreasingDuration_Sec)*dt

            delta_PGE_Conc=delta_PGE_Amount/InitialVolume
            Temp_ZR[1:1+InitialBinFromOrigin_Z,1:1+InitialBinFromOrigin_R]+=delta_PGE_Conc
        
        else:
            delta_PGE_Amount=0
        
        Temp_ZR=BoundaryCondition(Temp_ZR)
        Temp_ZR=DiffusionEquation(Temp_ZR,dz,dr,dt,DifCo,RadiusLength,Num_R)
        
        Released_TotalPGE2+=delta_PGE_Amount
        
                      
        if (t)%int(GetDataEveryNsec/dt)==0:
                resultdf=AppendResultDataFrame(Temp_ZR,EachVolume,t,dt,resultdf,ConcThrshold_nM,TotalPGE_fmol,EachRadiusPosition)
                PGEpattern=AppendPGEpattern(t,dt,delta_PGE_Amount,PGEpattern,IncreasingDuration_Sec,PlateauDuration_Sec,
                                            DecreasingDuration_Sec,TotalPGE_fmol,Released_TotalPGE2)
    
    return resultdf,PGEpattern

    


def ConstantPGE2(Num_T,dz,dt,dr,DifCo,ReleasingDuration_Sec,TotalPGE_fmol,RadiusLength,Num_R,EachVolume,ConcThrshold_nM,
                 EachRadiusPosition,InitialVolume,Temp_ZR,InitialBinFromOrigin_Z,InitialBinFromOrigin_R,GetDataEveryNsec,resultdf):

    for t in range(Num_T):
        if t*dt<ReleasingDuration_Sec:
            delta_PGE_Amount=TotalPGE_fmol/(ReleasingDuration_Sec/dt)
            delta_PGE_Conc=delta_PGE_Amount/InitialVolume
            
            Temp_ZR[1:1+InitialBinFromOrigin_Z,1:1+InitialBinFromOrigin_R]+=delta_PGE_Conc
            
        Temp_ZR=BoundaryCondition(Temp_ZR)
        Temp_ZR=DiffusionEquation(Temp_ZR,dz,dr,dt,DifCo,RadiusLength,Num_R)
                      
        if (t)%int(GetDataEveryNsec/dt)==0:
                resultdf=AppendResultDataFrame(Temp_ZR,EachVolume,t,dt,resultdf,ConcThrshold_nM,TotalPGE_fmol,EachRadiusPosition)
                
    return resultdf



def GetCoordinateParams(RadiusLength,dr,dz,InitialBinFromOrigin_Z,InitialBinFromOrigin_R,Num_R,Num_Z):
    
    InitialVolume=(np.pi)*(dz*InitialBinFromOrigin_Z)*((dr*InitialBinFromOrigin_R)**2)
    print("Initial Volume: "+str(InitialVolume)+" fL"+"\n")
    
    #Calculate the volume of each donut
    TwoNminusOne=np.arange(1,2*Num_R,step=2)
    EachDonut=(np.pi)*dz*(dr**2)*TwoNminusOne
    EachVolume=np.tile(EachDonut,[Num_Z,1])    
    EachRadiusPosition=np.linspace(dr/2,RadiusLength-dr/2,Num_R)
    
    return InitialVolume,EachDonut,EachVolume,EachRadiusPosition
    
    
def AppendPGEpattern(t,dt,delta_PGE_Amount,PGEpattern,IncreasingDuration_Sec,
                     PlateauDuration_Sec,DecreasingDuration_Sec,TotalPGE_fmol,Released_TotalPGE2):
    
    PGEpattern=PGEpattern.append({"delta_PGE_Amount_fmol_per_sec":delta_PGE_Amount/dt,
                    "time":t*dt,
                    "TotalPGE_fmol":TotalPGE_fmol,
                    "Released_TotalPGE2":Released_TotalPGE2,
                    "IncreasingDuration_Sec":IncreasingDuration_Sec,
                    "PlateauDuration_Sec":PlateauDuration_Sec,
                    "DecreasingDuration_Sec":DecreasingDuration_Sec
                    },ignore_index=True)
    
    return PGEpattern      



def AppendResultDataFrame(Temp_ZR,EachVolume,t,dt,resultdf,ConcThrshold_nM,TotalPGE_fmol,EachRadiusPosition):
    # if (t)%int(GetDataEveryNsec/dt)==0:
    TotalPGE2WithinCylinder=(Temp_ZR[1:-1,1:-1]*EachVolume).sum()
    
    if (t*dt)%(100)==0:
        print(int(t*dt), " sec,  Total PGE2 within the cylinder (fmol): ",TotalPGE2WithinCylinder)
    # plt.imshow(Temp_ZR[1:-1,1:-1],vmin=0,vmax=PlotMaxConc_nM*10**(-9))
    # plt.show()
    
    Z_zero=Temp_ZR[1,1:-1]
    MoreThanThreshold=Z_zero*10**9>ConcThrshold_nM
    MaxRadius=(MoreThanThreshold*EachRadiusPosition).max()
    
    resultdf=resultdf.append({"TotalPGE_fmol":TotalPGE_fmol,
                    "TotalPGE2WithinCylinder":TotalPGE2WithinCylinder,
                    "time":t*dt,
                    "MaxRadius":MaxRadius
                    },ignore_index=True)
    
    return resultdf


def Coordination_check(RadiusLength,Zlength,TotalTime,dr,dz,dt):
    if RadiusLength%dr + Zlength%dz >0:
        print("error")
        sys.exit()
        
    else:
        Num_Z=int(Zlength/dz)
        Num_R=int(RadiusLength/dr)
        Num_T=int(TotalTime/dt)
        
        return Num_Z,Num_R,Num_T

@jit
def BoundaryCondition(Temp_ZR):
    Temp_ZR[0,:]=Temp_ZR[1,:]   #neumann
    Temp_ZR[-1,:]=0             #Dirichlet
    # Temp_ZR[-1,:]=Temp_ZR[-2,:]
    
    Temp_ZR[:,0]=Temp_ZR[:,1]   #neumann
    Temp_ZR[:,-1]=0            #Dirichlet
    # Temp_ZR[:,-1]=Temp_ZR[:,-2]
    return Temp_ZR

@jit
def DiffusionEquation(Temp_ZR,dz,dr,dt,DifCo,RadiusLength,Num_R):
    #d2c/dr2
    d2c_dr2 = (1/(dr*dr))*(Temp_ZR[1:-1,2:  ]
                        -2*Temp_ZR[1:-1,1:-1]
                          +Temp_ZR[1:-1,0:-2])
    
    #(1/r)*(dc/dr)
    dc_dr = (1/ (2*dr))*(Temp_ZR[1:-1,2:]
                       - Temp_ZR[1:-1,0:-2]);
    DivR=1./np.linspace(dr/2,RadiusLength-dr/2,Num_R)
    dc_dr_DivR=dc_dr*DivR

    #d2c/dz2
    d2c_dz2 = (1/ (dr*dr))*(Temp_ZR[2:  ,1:-1] 
                        - 2*Temp_ZR[1:-1,1:-1]
                           +Temp_ZR[0:-2,1:-1])
    Temp_ZR[1:-1,1:-1] =  dt*DifCo*(d2c_dr2+dc_dr_DivR+d2c_dz2) + Temp_ZR[1:-1,1:-1]
    
    Temp_ZR=BoundaryCondition(Temp_ZR)
    
    return Temp_ZR



def diffusion_to_CSV(resultdf,filename_base):
    filename=filename_base+"_result.csv"
    resultdf.to_csv(filename)
    


def plot_radius(resultdf,filename_base,ConcThrshold_nM):
    filename=filename_base+".png"
    #resultdf = resultdf[resultdf["TotalPGE_fmol"]<11]
    resultdf['PGE'] = [f"{x}" for x in resultdf['TotalPGE_fmol']] 
    
    fig, ax = plt.subplots(1,1,figsize=(cm2inch(50), cm2inch(40)))
    resultdf["Time(min)"]=resultdf["time"]/60
    ax = sns.lineplot(x="Time(min)", y="MaxRadius",hue="PGE", data=resultdf,palette="nipy_spectral_r")
    plt.legend(bbox_to_anchor=(1.05, 0.9), loc=2, borderaxespad=0.)
    plt.ylim(-10,510)
    plt.yticks([0,100,200,300,400,500])
    
    plt.ylabel("Radius of PGE conc > "+str(ConcThrshold_nM) +" nM")
    plt.xlabel("Time (min)")
    pltsetting(plt,ax)
    plt.savefig(filename[:-4]+".png",format="png",dpi=600, transparent=True,bbox_inches='tight')
    plt.show()
    
    
def plot_PGEpattern(PGEpattern,filename_base):
    filename=filename_base+"_PGEpattern.png"
    #resultdf = resultdf[resultdf["TotalPGE_fmol"]<11]
    PGEpattern['PGE'] = [f"{x}" for x in PGEpattern['TotalPGE_fmol']] 
    
    PGEpattern['delta_PGE_Amount_attomol_per_sec'] = PGEpattern["delta_PGE_Amount_fmol_per_sec"]*1000

    PGEpattern['PGEmolecules'] = PGEpattern["delta_PGE_Amount_fmol_per_sec"]*6.02*10**(23-15)
    PGEpattern["Time(min)"]=PGEpattern["time"]/60
    
    fig, ax = plt.subplots(1,1,figsize=(cm2inch(50), cm2inch(40)))
    ax = sns.lineplot(x="Time(min)", y="PGEmolecules",hue="PGE", data=PGEpattern,palette="nipy_spectral_r")
    plt.legend(bbox_to_anchor=(1.05, 0.9), loc=2, borderaxespad=0.)
    
    plt.ylabel("PGE2(molecules)/sec")
    plt.xlabel("Time (min)")

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    pltsetting(plt,ax)
    plt.savefig(filename[:-4]+".png",format="png",dpi=600, transparent=True,bbox_inches='tight')
    plt.show()

def plot_PGE_sumTotal(PGEpattern,filename_base):
    filename=filename_base+"_PGE_totalReleased.png"
    #resultdf = resultdf[resultdf["TotalPGE_fmol"]<11]
    PGEpattern['PGE'] = [f"{x}" for x in PGEpattern['TotalPGE_fmol']] 

    PGEpattern["Time(min)"]=PGEpattern["time"]/60
    
    fig, ax = plt.subplots(1,1,figsize=(cm2inch(50), cm2inch(40)))
    ax = sns.lineplot(x="Time(min)", y="Released_TotalPGE2",hue="PGE", data=PGEpattern,palette="nipy_spectral_r")
    plt.legend(bbox_to_anchor=(1.05, 0.9), loc=2, borderaxespad=0.)
    
    plt.ylabel("PGE2 fmol/cell")
    plt.xlabel("Time (min)")
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    pltsetting(plt,ax)
    plt.savefig(filename[:-4]+".png",format="png",dpi=600, transparent=True,bbox_inches='tight')
    plt.show()
    
def pltsetting(plt,ax):
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_color('#000000')
    plt.gca().spines['bottom'].set_linewidth("1")
    plt.gca().spines['right'].set_color('#000000')
    plt.gca().spines['left'].set_linewidth("1")
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 8
    
    ax.xaxis.label.set_color('#000000')
    ax.yaxis.label.set_color('#000000')
    ax.xaxis.set_tick_params(width=1)
    ax.yaxis.set_tick_params(width=1)
    
def cm2inch(value):
    return value/25.4
    
if __name__ == '__main__':
    main()