import glob
import os
import shutil
import cv2

## =======Set parameters for all get_XXX_args()==================================================
SHOW_IMAGE = False
SAVE_IMAGE = False
DATA_NUM = 10000
DATA_TYPE = "val"
IMG_DIR = "/home/ali/Projects/datasets/bdd100k_data_0.9/images/100k/val"
SAVE_TXT_DIR = "/home/ali/Projects/datasets/BDD100K_Val_DCA_VPA_DUA_DLA_label_Txt_2023-12-26"
DATA_DIR = "/home/ali/Projects/datasets/bdd100k_data_0.9"
VLA_LABEL = 12
DCA_LABEL = 13
VPA_LABEL = 14
DUA_LABEL = 15
DLA_LABEL = 16
## ===============================================================================================
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
    parser.add_argument('-vpamlabel','--vpam-label',type=int,help='VPA middle label',default=DUA_LABEL)
    parser.add_argument('-dlalabel','--dla-label',type=int,help='DLA label',default=DLA_LABEL)
    parser.add_argument('-saveimg','--save-img',type=bool,help='save images',default=SAVE_IMAGE)

    parser.add_argument('-datatype','--data-type',help='data type',default=DATA_TYPE)
    parser.add_argument('-datanum','--data-num',type=int,help='number of images to crop',default=DATA_NUM)



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


    parser.add_argument('-savetxtdir','--save-txtdir',help='save txt directory',\
                        default=SAVE_TXT_DIR)
    parser.add_argument('-vlalabel','--vla-label',type=int,help='VLA label',default=VLA_LABEL)
    parser.add_argument('-dcalabel','--dca-label',type=int,help='DCA label',default=DCA_LABEL)
    parser.add_argument('-vpalabel','--vpa-label',type=int,help='DCA label',default=VPA_LABEL)
    parser.add_argument('-vpamlabel','--vpam-label',type=int,help='VPA middle label',default=DUA_LABEL)
    parser.add_argument('-dlalabel','--dla-label',type=int,help='DLA label',default=DLA_LABEL)
    parser.add_argument('-saveimg','--save-img',type=bool,help='save images',default=SAVE_IMAGE)

    parser.add_argument('-datatype','--data-type',help='data type',default=DATA_TYPE)
    parser.add_argument('-datanum','--data-num',type=int,help='number of images to crop',default=DATA_NUM)



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


    parser.add_argument('-savetxtdir','--save-txtdir',help='save txt directory',\
                        default=SAVE_TXT_DIR)
    parser.add_argument('-vlalabel','--vla-label',type=int,help='VLA label',default=VLA_LABEL)
    parser.add_argument('-dcalabel','--dca-label',type=int,help='DCA label',default=DCA_LABEL)
    parser.add_argument('-vpalabel','--vpa-label',type=int,help='DCA label',default=VPA_LABEL)
    parser.add_argument('-vpamlabel','--vpam-label',type=int,help='VPA middle label',default=DUA_LABEL)
    parser.add_argument('-dlalabel','--dla-label',type=int,help='DLA label',default=DLA_LABEL)
    parser.add_argument('-saveimg','--save-img',type=bool,help='save images',default=SAVE_IMAGE)

    parser.add_argument('-datatype','--data-type',help='data type',default=DATA_TYPE)
    parser.add_argument('-datanum','--data-num',type=int,help='number of images to crop',default=DATA_NUM)



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


    parser.add_argument('-savetxtdir','--save-txtdir',help='save txt directory',\
                        default=SAVE_TXT_DIR)
    parser.add_argument('-vlalabel','--vla-label',type=int,help='VLA label',default=VLA_LABEL)
    parser.add_argument('-dcalabel','--dca-label',type=int,help='DCA label',default=DCA_LABEL)
    parser.add_argument('-vpalabel','--vpa-label',type=int,help='VPA label',default=VPA_LABEL)
    parser.add_argument('-vpamlabel','--vpam-label',type=int,help='DUA label',default=DUA_LABEL)
    parser.add_argument('-dlalabel','--dla-label',type=int,help='DLA label',default=DLA_LABEL)
    parser.add_argument('-saveimg','--save-img',type=bool,help='save images',default=SAVE_IMAGE)

    parser.add_argument('-datatype','--data-type',help='data type',default=DATA_TYPE)
    parser.add_argument('-datanum','--data-num',type=int,help='number of images to crop',default=DATA_NUM)



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


    parser.add_argument('-savetxtdir','--save-txtdir',help='save txt directory',\
                        default=SAVE_TXT_DIR)
    parser.add_argument('-vlalabel','--vla-label',type=int,help='VLA label',default=VLA_LABEL)
    parser.add_argument('-dcalabel','--dca-label',type=int,help='DCA label',default=DCA_LABEL)
    parser.add_argument('-vpalabel','--vpa-label',type=int,help='VPA label',default=VPA_LABEL)
    parser.add_argument('-vpamlabel','--vpam-label',type=int,help='DUA label',default=DUA_LABEL)
    parser.add_argument('-dlalabel','--dla-label',type=int,help='DLA label',default=DLA_LABEL)
    parser.add_argument('-saveimg','--save-img',type=bool,help='save images',default=SAVE_IMAGE)

    parser.add_argument('-datatype','--data-type',help='data type',default=DATA_TYPE)
    parser.add_argument('-datanum','--data-num',type=int,help='number of images to crop',default=DATA_NUM)



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





