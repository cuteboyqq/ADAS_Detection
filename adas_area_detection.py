import glob
import os
import shutil
from config.config import get_VLA_args, get_DCA_args, get_VPA_args, \
    get_DUA_args, get_DLA_args,get_MA_args, get_CRA_args
import cv2
from tasks.VPA import VPA
from tasks.VLA import VLA
from tasks.DCA import DCA
from tasks.DUA import DUA
from tasks.DLA import DLA
from tasks.CRA import CRA
from tasks.MA import MultiAreaTask
from engine.dataset import BaseDataset


if __name__=="__main__":
   
    Get_VLA = False
    Get_DCA = False
    Get_VPA = False
    Get_DUA  = False
    Get_DLA = False
    Get_MA = False
    Get_CRA = True
    if Get_VLA:
        args_vla = get_VLA_args()
        vla = VLA(args_vla)
        vla.Add_Vanish_Line_Area_Yolo_Txt_Labels()

    if Get_DCA:
        args_dca = get_DCA_args()
        dca = DCA(args_dca)
        dca.Get_DCA_Yolo_Txt_Labels()

    
    
    if Get_DUA:
        args_dua = get_DUA_args()
        dua = DUA(args_dua)
        dua.Get_DUA_Yolo_Txt_Labels(Get_VPA=False,Get_DUA=False,Get_Two_DUA=False,Get_Three_DUA=True)
    
    if Get_DLA:
        args_dla = get_DLA_args()
        dla = DLA(args_dla)
        dla.Get_DLA_Yolo_Txt_Labels()
    
    if Get_MA:
        args_ma = get_MA_args()
        ma = MultiAreaTask(args_ma)
        ma.Get_Multi_Area_Tasks_Yolo_Txt_Labels()

    if Get_VPA:
        args_vpa = get_VPA_args()
        vpa = VPA(args_vpa)
        vpa.Get_VPA_Yolo_Txt_Labels()
    
    if Get_CRA:
        args_cra = get_CRA_args()
        cra = CRA(args_cra)
        cra.Get_CRA_Yolo_Txt_Labels()


