import glob
import os
import shutil
import cv2

## =======Set parameters for all get_XXX_args()==================================================
SHOW_IMAGE = False
SAVE_IMAGE = False
DATA_NUM = 10000
DATA_TYPE = "val"
IMG_DIR = "/home/ali/Projects/datasets/bdd100k_data_0.8/images/100k/val"
SAVE_TXT_DIR = "/home/ali/Projects/datasets/BDD100K_0.8_Train_VLA_DCA_DUA3_label_Txt_2023-12-29-Test-------------------------"
DATA_DIR = "/home/ali/Projects/datasets/bdd100k_data_0.8"
VLA_LABEL = 12
DCA_LABEL = 13
VPA_LABEL = 14
DUA_UP_LABEL = 15
DUA_MIDDLE_LABEL = 16
DUA_DOWN_LABEL = 17
DLA_LEFT_LABEL = 18
DLA_RIGHT_LABEL = 19
## ===============Parsing detection folder================================================================================
DCA_PARSE_DET_FOLDER  = "detection-VLA"
'''2023-12-29 parsing step 2'''
VPA_PARSE_DET_FOLDER  = "detection-VLA-DCA-DUA3"
DUA_PARSE_DET_FOLDER  = "detection-VLA-DCA-VPA"
DLA_PARSE_DET_FOLDER = "detection-VLA-DCA-VPA-DUA"
'''2023-12-29 parsing step 1'''
MA_PARSE_DET_FOLDER = "detection-ori"

DCA_SAVE_TXT_DIR = "/home/ali/Projects/datasets/bdd100k_data_0.8/labels/detection-VLA-DCA/val"
'''2023-12-29 parsing step 2'''
VPA_SAVE_TXT_DIR = "/home/ali/Projects/datasets/bdd100k_data_0.8/labels/detection-VLA-DCA-DUA3-VPA/val"
DUA_SAVE_TXT_DIR = "/home/ali/Projects/datasets/bdd100k_data_0.8/labels/detection-VLA-DCA-VPA-DUA/val"
'''2023-12-29 parsing step 1'''
MA_SAVE_TXT_DIR = "/home/ali/Projects/datasets/bdd100k_data_0.8/labels/detection-VLA-DCA-VPA-DUA3/val"

## =================MA Multi Area Task=========================================================================================
ENABLE_VLA = True
ENABLE_DCA = True
ENABLE_VPA = True
ENABLE_DUA_UP = True
ENABLE_DUA_MID = True
ENABLE_DUA_DOWN = True

'''
    VLA parameters
'''
def get_VLA_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-imdir','--im-dir',help='image directory',\
                        default=IMG_DIR)
    parser.add_argument('-savedir','--save-dir',help='save image directory',\
                        default="/home/ali/Projects/datasets/BDD100K_Val_VLA_DCA_VPA_label_Txt_h80_2023-11-24")
    parser.add_argument('-datadir','--data-dir',help='dataset directory',\
                        default=DATA_DIR)


    parser.add_argument('-savetxtdir','--save-txtdir',help='save txt directory',\
                        default=SAVE_TXT_DIR)
    parser.add_argument('-vlalabel','--vla-label',type=int,help='VLA label',default=VLA_LABEL)
    parser.add_argument('-dcalabel','--dca-label',type=int,help='DCA label',default=DCA_LABEL)
    parser.add_argument('-vpalabel','--vpa-label',type=int,help='DCA label',default=VPA_LABEL)
    parser.add_argument('-duauplabel','--dua-uplabel',type=int,help='VUA up label',default=DUA_UP_LABEL)
    parser.add_argument('-duamiddlelabel','--dua-middlelabel',type=int,help='VUA middle label',default=DUA_MIDDLE_LABEL)
    parser.add_argument('-duadownlabel','--dua-downlabel',type=int,help='VUA down label',default=DUA_DOWN_LABEL)
    parser.add_argument('-dlaleftlabel','--dla-leftlabel',type=int,help='DLA label',default=DLA_LEFT_LABEL)
    parser.add_argument('-dlarightlabel','--dla-rightlabel',type=int,help='DLA label',default=DLA_RIGHT_LABEL)
    parser.add_argument('-saveimg','--save-img',type=bool,help='save images',default=SAVE_IMAGE)

    parser.add_argument('-datatype','--data-type',help='data type',default=DATA_TYPE)
    parser.add_argument('-datanum','--data-num',type=int,help='number of images to crop',default=DATA_NUM)


    ## ========================MA parameters=====================================================================
    parser.add_argument('-enableval','--enable-vla',type=bool,help='enable VLA',default=ENABLE_VLA)
    parser.add_argument('-enabledca','--enable-dca',type=bool,help='enable DCA',default=ENABLE_DCA)
    parser.add_argument('-enablevpa','--enable-vpa',type=bool,help='enable VPA',default=ENABLE_VPA)
    parser.add_argument('-enableduaup','--enable-duaup',type=bool,help='enable DUA up',default=ENABLE_DUA_UP)
    parser.add_argument('-enableduamid','--enable-duamid',type=bool,help='enable DUA mid',default=ENABLE_DUA_MID)
    parser.add_argument('-enableduadown','--enable-duadown',type=bool,help='enable DUA down',default=ENABLE_DUA_DOWN)
    ## ========================================================================================================
    parser.add_argument('-showim','--show-im',type=bool,help='show images',default=SHOW_IMAGE)
    parser.add_argument('-showimcrop','--show-imcrop',type=bool,help='show crop images',default=True)
    parser.add_argument('-showvanishline','--show-vanishline',type=bool,help='show vanish line in image',default=False)
    parser.add_argument('-saveimcrop','--save-imcrop',type=bool,help='save  crop images',default=True)

    parser.add_argument('-multicrop','--multi-crop',type=bool,help='save multiple vanish area crop images',default=True)
    parser.add_argument('-multinum','--multi-num',type=int,help='number of multiple vanish area crop images',default=6)
    parser.add_argument('-shiftpixel','--shift-pixels',type=int,help='number of multiple crop images shift pixels',default=2)

    parser.add_argument('-splitnum','--split-num',type=int,help='split number',default=10)
    parser.add_argument('-splitheight','--split-height',type=int,help='split image height',default=80)
    parser.add_argument('-dataset','--dataset',help='dataset directory',default="/home/ali/Projects/datasets/CULane/driver_161_90frame_crop_2cls/train")
    return parser.parse_args()


'''
    DCA parameters
'''
def get_DCA_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-imdir','--im-dir',help='image directory',\
                        default=IMG_DIR)
    parser.add_argument('-savedir','--save-dir',help='save image directory',\
                        default="/home/ali/Projects/datasets/BDD100K_Val_VLA_DCA_VPA_label_Txt_h80_2023-11-24")
    parser.add_argument('-datadir','--data-dir',help='dataset directory',\
                        default=DATA_DIR)

    parser.add_argument('-detfolder','--det-folder',help='detection folder',\
                        default=DCA_PARSE_DET_FOLDER)

    parser.add_argument('-savetxtdir','--save-txtdir',help='save txt directory',\
                        default=DCA_SAVE_TXT_DIR)
    parser.add_argument('-vlalabel','--vla-label',type=int,help='VLA label',default=VLA_LABEL)
    parser.add_argument('-dcalabel','--dca-label',type=int,help='DCA label',default=DCA_LABEL)
    parser.add_argument('-vpalabel','--vpa-label',type=int,help='DCA label',default=VPA_LABEL)
    parser.add_argument('-duauplabel','--dua-uplabel',type=int,help='VUA up label',default=DUA_UP_LABEL)
    parser.add_argument('-duamiddlelabel','--dua-middlelabel',type=int,help='VUA middle label',default=DUA_MIDDLE_LABEL)
    parser.add_argument('-duadownlabel','--dua-downlabel',type=int,help='VUA down label',default=DUA_DOWN_LABEL)
    parser.add_argument('-dlaleftlabel','--dla-leftlabel',type=int,help='DLA label',default=DLA_LEFT_LABEL)
    parser.add_argument('-dlarightlabel','--dla-rightlabel',type=int,help='DLA label',default=DLA_RIGHT_LABEL)
    parser.add_argument('-saveimg','--save-img',type=bool,help='save images',default=SAVE_IMAGE)

    parser.add_argument('-datatype','--data-type',help='data type',default=DATA_TYPE)
    parser.add_argument('-datanum','--data-num',type=int,help='number of images to crop',default=DATA_NUM)

    ## ========================MA parameters=====================================================================
    parser.add_argument('-enableval','--enable-vla',type=bool,help='enable VLA',default=ENABLE_VLA)
    parser.add_argument('-enabledca','--enable-dca',type=bool,help='enable DCA',default=ENABLE_DCA)
    parser.add_argument('-enablevpa','--enable-vpa',type=bool,help='enable VPA',default=ENABLE_VPA)
    parser.add_argument('-enableduaup','--enable-duaup',type=bool,help='enable DUA up',default=ENABLE_DUA_UP)
    parser.add_argument('-enableduamid','--enable-duamid',type=bool,help='enable DUA mid',default=ENABLE_DUA_MID)
    parser.add_argument('-enableduadown','--enable-duadown',type=bool,help='enable DUA down',default=ENABLE_DUA_DOWN)
    ## ========================================================================================================



    parser.add_argument('-showim','--show-im',type=bool,help='show images',default=SHOW_IMAGE)
    parser.add_argument('-showimcrop','--show-imcrop',type=bool,help='show crop images',default=True)
    parser.add_argument('-showvanishline','--show-vanishline',type=bool,help='show vanish line in image',default=False)
    parser.add_argument('-saveimcrop','--save-imcrop',type=bool,help='save  crop images',default=True)

    parser.add_argument('-multicrop','--multi-crop',type=bool,help='save multiple vanish area crop images',default=True)
    parser.add_argument('-multinum','--multi-num',type=int,help='number of multiple vanish area crop images',default=6)
    parser.add_argument('-shiftpixel','--shift-pixels',type=int,help='number of multiple crop images shift pixels',default=2)

    parser.add_argument('-splitnum','--split-num',type=int,help='split number',default=10)
    parser.add_argument('-splitheight','--split-height',type=int,help='split image height',default=80)
    parser.add_argument('-dataset','--dataset',help='dataset directory',default="/home/ali/Projects/datasets/CULane/driver_161_90frame_crop_2cls/train")
    return parser.parse_args()


'''
    VPA parameters
'''
def get_VPA_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-imdir','--im-dir',help='image directory',\
                        default=IMG_DIR)
    parser.add_argument('-savedir','--save-dir',help='save image directory',\
                        default="/home/ali/Projects/datasets/BDD100K_Val_VLA_DCA_VPA_label_Txt_h80_2023-11-24")
    parser.add_argument('-datadir','--data-dir',help='dataset directory',\
                        default=DATA_DIR)

    parser.add_argument('-detfolder','--det-folder',help='detection folder',\
                        default=VPA_PARSE_DET_FOLDER)

    parser.add_argument('-savetxtdir','--save-txtdir',help='save txt directory',\
                        default=VPA_SAVE_TXT_DIR)
    parser.add_argument('-vlalabel','--vla-label',type=int,help='VLA label',default=VLA_LABEL)
    parser.add_argument('-dcalabel','--dca-label',type=int,help='DCA label',default=DCA_LABEL)
    parser.add_argument('-vpalabel','--vpa-label',type=int,help='DCA label',default=VPA_LABEL)
    parser.add_argument('-duauplabel','--dua-uplabel',type=int,help='VUA up label',default=DUA_UP_LABEL)
    parser.add_argument('-duamiddlelabel','--dua-middlelabel',type=int,help='VUA middle label',default=DUA_MIDDLE_LABEL)
    parser.add_argument('-duadownlabel','--dua-downlabel',type=int,help='VUA down label',default=DUA_DOWN_LABEL)
    parser.add_argument('-dlaleftlabel','--dla-leftlabel',type=int,help='DLA label',default=DLA_LEFT_LABEL)
    parser.add_argument('-dlarightlabel','--dla-rightlabel',type=int,help='DLA label',default=DLA_RIGHT_LABEL)
    parser.add_argument('-saveimg','--save-img',type=bool,help='save images',default=SAVE_IMAGE)

    parser.add_argument('-datatype','--data-type',help='data type',default=DATA_TYPE)
    parser.add_argument('-datanum','--data-num',type=int,help='number of images to crop',default=DATA_NUM)
    ## ========================MA parameters=====================================================================
    parser.add_argument('-enableval','--enable-vla',type=bool,help='enable VLA',default=ENABLE_VLA)
    parser.add_argument('-enabledca','--enable-dca',type=bool,help='enable DCA',default=ENABLE_DCA)
    parser.add_argument('-enablevpa','--enable-vpa',type=bool,help='enable VPA',default=ENABLE_VPA)
    parser.add_argument('-enableduaup','--enable-duaup',type=bool,help='enable DUA up',default=ENABLE_DUA_UP)
    parser.add_argument('-enableduamid','--enable-duamid',type=bool,help='enable DUA mid',default=ENABLE_DUA_MID)
    parser.add_argument('-enableduadown','--enable-duadown',type=bool,help='enable DUA down',default=ENABLE_DUA_DOWN)
    ## ========================================================================================================


    parser.add_argument('-showim','--show-im',type=bool,help='show images',default=SHOW_IMAGE)
    parser.add_argument('-showimcrop','--show-imcrop',type=bool,help='show crop images',default=True)
    parser.add_argument('-showvanishline','--show-vanishline',type=bool,help='show vanish line in image',default=False)
    parser.add_argument('-saveimcrop','--save-imcrop',type=bool,help='save  crop images',default=True)

    parser.add_argument('-multicrop','--multi-crop',type=bool,help='save multiple vanish area crop images',default=True)
    parser.add_argument('-multinum','--multi-num',type=int,help='number of multiple vanish area crop images',default=6)
    parser.add_argument('-shiftpixel','--shift-pixels',type=int,help='number of multiple crop images shift pixels',default=2)

    parser.add_argument('-splitnum','--split-num',type=int,help='split number',default=10)
    parser.add_argument('-splitheight','--split-height',type=int,help='split image height',default=80)
    parser.add_argument('-dataset','--dataset',help='dataset directory',default="/home/ali/Projects/datasets/CULane/driver_161_90frame_crop_2cls/train")
    return parser.parse_args()

'''
    DUA parameters
'''
def get_DUA_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-imdir','--im-dir',help='image directory',\
                        default=IMG_DIR)
    parser.add_argument('-savedir','--save-dir',help='save image directory',\
                        default="/home/ali/Projects/datasets/BDD100K_Val_VLA_DCA_VPA_label_Txt_h80_2023-11-24")
    parser.add_argument('-datadir','--data-dir',help='dataset directory',\
                        default=DATA_DIR)

    parser.add_argument('-detfolder','--det-folder',help='detection folder',\
                        default=DUA_PARSE_DET_FOLDER)

    parser.add_argument('-savetxtdir','--save-txtdir',help='save txt directory',\
                        default=DUA_SAVE_TXT_DIR)
    parser.add_argument('-vlalabel','--vla-label',type=int,help='VLA label',default=VLA_LABEL)
    parser.add_argument('-dcalabel','--dca-label',type=int,help='DCA label',default=DCA_LABEL)
    parser.add_argument('-vpalabel','--vpa-label',type=int,help='VPA label',default=VPA_LABEL)
    parser.add_argument('-duauplabel','--dua-uplabel',type=int,help='VUA up label',default=DUA_UP_LABEL)
    parser.add_argument('-duamiddlelabel','--dua-middlelabel',type=int,help='VUA middle label',default=DUA_MIDDLE_LABEL)
    parser.add_argument('-duadownlabel','--dua-downlabel',type=int,help='VUA down label',default=DUA_DOWN_LABEL)
    parser.add_argument('-dlaleftlabel','--dla-leftlabel',type=int,help='DLA label',default=DLA_LEFT_LABEL)
    parser.add_argument('-dlarightlabel','--dla-rightlabel',type=int,help='DLA label',default=DLA_RIGHT_LABEL)
    parser.add_argument('-saveimg','--save-img',type=bool,help='save images',default=SAVE_IMAGE)

    parser.add_argument('-datatype','--data-type',help='data type',default=DATA_TYPE)
    parser.add_argument('-datanum','--data-num',type=int,help='number of images to crop',default=DATA_NUM)
    ## ========================MA parameters=====================================================================
    parser.add_argument('-enableval','--enable-vla',type=bool,help='enable VLA',default=ENABLE_VLA)
    parser.add_argument('-enabledca','--enable-dca',type=bool,help='enable DCA',default=ENABLE_DCA)
    parser.add_argument('-enablevpa','--enable-vpa',type=bool,help='enable VPA',default=ENABLE_VPA)
    parser.add_argument('-enableduaup','--enable-duaup',type=bool,help='enable DUA up',default=ENABLE_DUA_UP)
    parser.add_argument('-enableduamid','--enable-duamid',type=bool,help='enable DUA mid',default=ENABLE_DUA_MID)
    parser.add_argument('-enableduadown','--enable-duadown',type=bool,help='enable DUA down',default=ENABLE_DUA_DOWN)
    ## ========================================================================================================


    parser.add_argument('-showim','--show-im',type=bool,help='show images',default=SHOW_IMAGE)
    parser.add_argument('-showimcrop','--show-imcrop',type=bool,help='show crop images',default=True)
    parser.add_argument('-showvanishline','--show-vanishline',type=bool,help='show vanish line in image',default=False)
    parser.add_argument('-saveimcrop','--save-imcrop',type=bool,help='save  crop images',default=True)

    parser.add_argument('-multicrop','--multi-crop',type=bool,help='save multiple vanish area crop images',default=True)
    parser.add_argument('-multinum','--multi-num',type=int,help='number of multiple vanish area crop images',default=6)
    parser.add_argument('-shiftpixel','--shift-pixels',type=int,help='number of multiple crop images shift pixels',default=2)

    parser.add_argument('-splitnum','--split-num',type=int,help='split number',default=10)
    parser.add_argument('-splitheight','--split-height',type=int,help='split image height',default=80)
    parser.add_argument('-dataset','--dataset',help='dataset directory',default="/home/ali/Projects/datasets/CULane/driver_161_90frame_crop_2cls/train")
    return parser.parse_args()

'''
    DLA parameters
'''
def get_DLA_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-imdir','--im-dir',help='image directory',\
                        default=IMG_DIR)
    parser.add_argument('-savedir','--save-dir',help='save image directory',\
                        default="/home/ali/Projects/datasets/BDD100K_Val_VLA_DCA_VPA_label_Txt_h80_2023-11-24")
    parser.add_argument('-datadir','--data-dir',help='dataset directory',\
                        default=DATA_DIR)

    parser.add_argument('-detfolder','--det-folder',help='detection folder',\
                        default=DLA_PARSE_DET_FOLDER)

    parser.add_argument('-savetxtdir','--save-txtdir',help='save txt directory',\
                        default=SAVE_TXT_DIR)
    parser.add_argument('-vlalabel','--vla-label',type=int,help='VLA label',default=VLA_LABEL)
    parser.add_argument('-dcalabel','--dca-label',type=int,help='DCA label',default=DCA_LABEL)
    parser.add_argument('-vpalabel','--vpa-label',type=int,help='VPA label',default=VPA_LABEL)
    parser.add_argument('-duauplabel','--dua-uplabel',type=int,help='VUA up label',default=DUA_UP_LABEL)
    parser.add_argument('-duamiddlelabel','--dua-middlelabel',type=int,help='VUA middle label',default=DUA_MIDDLE_LABEL)
    parser.add_argument('-duadownlabel','--dua-downlabel',type=int,help='VUA down label',default=DUA_DOWN_LABEL)
    parser.add_argument('-dlaleftlabel','--dla-leftlabel',type=int,help='DLA label',default=DLA_LEFT_LABEL)
    parser.add_argument('-dlarightlabel','--dla-rightlabel',type=int,help='DLA label',default=DLA_RIGHT_LABEL)
    parser.add_argument('-saveimg','--save-img',type=bool,help='save images',default=SAVE_IMAGE)

    parser.add_argument('-datatype','--data-type',help='data type',default=DATA_TYPE)
    parser.add_argument('-datanum','--data-num',type=int,help='number of images to crop',default=DATA_NUM)

    ## ========================MA parameters=====================================================================
    parser.add_argument('-enableval','--enable-vla',type=bool,help='enable VLA',default=ENABLE_VLA)
    parser.add_argument('-enabledca','--enable-dca',type=bool,help='enable DCA',default=ENABLE_DCA)
    parser.add_argument('-enablevpa','--enable-vpa',type=bool,help='enable VPA',default=ENABLE_VPA)
    parser.add_argument('-enableduaup','--enable-duaup',type=bool,help='enable DUA up',default=ENABLE_DUA_UP)
    parser.add_argument('-enableduamid','--enable-duamid',type=bool,help='enable DUA mid',default=ENABLE_DUA_MID)
    parser.add_argument('-enableduadown','--enable-duadown',type=bool,help='enable DUA down',default=ENABLE_DUA_DOWN)
    ## ========================================================================================================

    parser.add_argument('-showim','--show-im',type=bool,help='show images',default=SHOW_IMAGE)
    parser.add_argument('-showimcrop','--show-imcrop',type=bool,help='show crop images',default=True)
    parser.add_argument('-showvanishline','--show-vanishline',type=bool,help='show vanish line in image',default=False)
    parser.add_argument('-saveimcrop','--save-imcrop',type=bool,help='save  crop images',default=True)

    parser.add_argument('-multicrop','--multi-crop',type=bool,help='save multiple vanish area crop images',default=True)
    parser.add_argument('-multinum','--multi-num',type=int,help='number of multiple vanish area crop images',default=6)
    parser.add_argument('-shiftpixel','--shift-pixels',type=int,help='number of multiple crop images shift pixels',default=2)

    parser.add_argument('-splitnum','--split-num',type=int,help='split number',default=10)
    parser.add_argument('-splitheight','--split-height',type=int,help='split image height',default=80)
    parser.add_argument('-dataset','--dataset',help='dataset directory',default="/home/ali/Projects/datasets/CULane/driver_161_90frame_crop_2cls/train")
    return parser.parse_args()


'''
    MA(Multi Area) parameters
'''
def get_MA_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-imdir','--im-dir',help='image directory',\
                        default=IMG_DIR)
    parser.add_argument('-savedir','--save-dir',help='save image directory',\
                        default="/home/ali/Projects/datasets/BDD100K_Val_VLA_DCA_VPA_label_Txt_h80_2023-11-24")
    parser.add_argument('-datadir','--data-dir',help='dataset directory',\
                        default=DATA_DIR)

    parser.add_argument('-detfolder','--det-folder',help='detection folder',\
                        default=MA_PARSE_DET_FOLDER)

    parser.add_argument('-savetxtdir','--save-txtdir',help='save txt directory',\
                        default=MA_SAVE_TXT_DIR)
    parser.add_argument('-vlalabel','--vla-label',type=int,help='VLA label',default=VLA_LABEL)
    parser.add_argument('-dcalabel','--dca-label',type=int,help='DCA label',default=DCA_LABEL)
    parser.add_argument('-vpalabel','--vpa-label',type=int,help='VPA label',default=VPA_LABEL)
    parser.add_argument('-duauplabel','--dua-uplabel',type=int,help='VUA up label',default=DUA_UP_LABEL)
    parser.add_argument('-duamiddlelabel','--dua-middlelabel',type=int,help='VUA middle label',default=DUA_MIDDLE_LABEL)
    parser.add_argument('-duadownlabel','--dua-downlabel',type=int,help='VUA down label',default=DUA_DOWN_LABEL)
    parser.add_argument('-dlaleftlabel','--dla-leftlabel',type=int,help='DLA label',default=DLA_LEFT_LABEL)
    parser.add_argument('-dlarightlabel','--dla-rightlabel',type=int,help='DLA label',default=DLA_RIGHT_LABEL)
    parser.add_argument('-saveimg','--save-img',type=bool,help='save images',default=SAVE_IMAGE)

    parser.add_argument('-datatype','--data-type',help='data type',default=DATA_TYPE)
    parser.add_argument('-datanum','--data-num',type=int,help='number of images to crop',default=DATA_NUM)

    ## ========================MA parameters=====================================================================
    parser.add_argument('-enableval','--enable-vla',type=bool,help='enable VLA',default=ENABLE_VLA)
    parser.add_argument('-enabledca','--enable-dca',type=bool,help='enable DCA',default=ENABLE_DCA)
    parser.add_argument('-enablevpa','--enable-vpa',type=bool,help='enable VPA',default=ENABLE_VPA)
    parser.add_argument('-enableduaup','--enable-duaup',type=bool,help='enable DUA up',default=ENABLE_DUA_UP)
    parser.add_argument('-enableduamid','--enable-duamid',type=bool,help='enable DUA mid',default=ENABLE_DUA_MID)
    parser.add_argument('-enableduadown','--enable-duadown',type=bool,help='enable DUA down',default=ENABLE_DUA_DOWN)
    ## ========================================================================================================

    parser.add_argument('-showim','--show-im',type=bool,help='show images',default=SHOW_IMAGE)
    parser.add_argument('-showimcrop','--show-imcrop',type=bool,help='show crop images',default=True)
    parser.add_argument('-showvanishline','--show-vanishline',type=bool,help='show vanish line in image',default=False)
    parser.add_argument('-saveimcrop','--save-imcrop',type=bool,help='save  crop images',default=True)

    parser.add_argument('-multicrop','--multi-crop',type=bool,help='save multiple vanish area crop images',default=True)
    parser.add_argument('-multinum','--multi-num',type=int,help='number of multiple vanish area crop images',default=6)
    parser.add_argument('-shiftpixel','--shift-pixels',type=int,help='number of multiple crop images shift pixels',default=2)

    parser.add_argument('-splitnum','--split-num',type=int,help='split number',default=10)
    parser.add_argument('-splitheight','--split-height',type=int,help='split image height',default=80)
    parser.add_argument('-dataset','--dataset',help='dataset directory',default="/home/ali/Projects/datasets/CULane/driver_161_90frame_crop_2cls/train")
    return parser.parse_args()


