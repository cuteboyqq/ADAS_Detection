import glob
import os
import shutil
from config.config import get_VLA_args, get_DCA_args, get_VPA_args, get_LA_args
import cv2
from tasks.VPA import VPA
from tasks.VLA import VLA
from tasks.DCA import DCA
from tasks.LA import LA
from engine.dataset import BaseDataset


if __name__=="__main__":
   
    Get_VLA = False
    Get_VPA = True
    Get_DCA = False
    Get_LA  = False
    if Get_VLA:
        args_vla = get_VLA_args()
        vla = VLA(args_vla)
        vla.Add_Vanish_Line_Area_Yolo_Txt_Labels()

    if Get_DCA:
        args_dca = get_DCA_args()
        dca = DCA(args_dca)
        dca.Get_DCA_Yolo_Txt_Labels()

    if Get_VPA:
        args_vpa = get_VPA_args()
        vpa = VPA(args_vpa)
        vpa.Get_VPA_Yolo_Txt_Labels()
    
    if Get_LA:
        args_la = get_LA_args()
        la = LA(args_la)
        la.Get_LA_Yolo_Txt_Labels()



