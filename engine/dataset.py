import glob
import os
import shutil
import cv2

class BaseDataset:

    def __init__(self,args):

        ## data directory
        self.dataset_dir =  args.dataset
        self.save_dir = args.save_dir
        self.im_dir = args.im_dir
        self.dataset_dir = args.data_dir
        
        ## split crop image detail
        self.split_num = args.split_num
        self.split_height = args.split_height
        
        self.save_imcrop = args.save_imcrop

        ## Augmentation using shift cropping
        self.multi_crop = args.multi_crop
        self.multi_num = args.multi_num
        self.shift_pixels = args.shift_pixels
        
        ## yolo.txt parameter
        self.save_txtdir = args.save_txtdir
        self.vla_label = args.vla_label
        self.dca_label = args.dca_label
        self.vpa_label = args.vpa_label
        self.save_img = args.save_img

        ## parse image detail
        self.data_type = args.data_type
        self.data_num = args.data_num
        self.wanted_label_list = [2,3,4]
        # 0: pedestrian
        # 1: rider
        # 2: car
        # 3: truck
        # 4: bus
        # 5: train
        # 6: motorcycle
        # 7: bicycle
        # 8: traffic light
        # 9: traffic sign
        ## view result
        self.show_vanishline = args.show_vanishline
        self.show_imcrop = args.show_imcrop
        self.show_im = args.show_im

    def parse_path(self,path,type="val"):
        file = path.split(os.sep)[-1]
        file_name = file.split(".")[0]
        drivable_file  = file_name + ".png"
        lane_file  = file_name + ".png"
        detection_file = file_name + ".txt"
        drivable_path = os.path.join(self.dataset_dir,"labels","drivable","colormaps",type,drivable_file)
        lane_path = os.path.join(self.dataset_dir,"labels","lane","colormaps",type,lane_file)
        detection_path = os.path.join(self.dataset_dir,"labels","detection",type,detection_file)
        return drivable_path,lane_path,detection_path
    
    def parse_path_ver2(self,path,type="val",detect_folder="detection"):
        file = path.split(os.sep)[-1]
        file_name = file.split(".")[0]
        drivable_file  = file_name + ".png"
        lane_file  = file_name + ".png"
        detection_file = file_name + ".txt"
        drivable_path = os.path.join(self.dataset_dir,"labels","drivable","colormaps",type,drivable_file)
        drivable_mask_path = os.path.join(self.dataset_dir,"labels","drivable","masks",type,drivable_file)
        lane_path = os.path.join(self.dataset_dir,"labels","lane","colormaps",type,lane_file)
        detection_path = os.path.join(self.dataset_dir,"labels",detect_folder,type,detection_file)
        return drivable_path,drivable_mask_path,lane_path,detection_path


    def find_max_value(self,a,b,c):
        max = None
        index = None
        if a>b:
            max=a
            index=1
        else:
            max=b
            index=2
        if c>max:
            max=c
            index=3
        return max,index

    def find_min_value(self,a,b,c):
        min=None
        index=None
        if a<b:
            min=a
            index=1
        else:
            min=b
            index=2
        
        if c<min:
            min=c
            index=3

        return min,index
        

    def Get_Min_y_In_Drivable_Area(self,drivable_path):
        if not os.path.exists(drivable_path):
            drivable_img = cv2.imread(drivable_path)
            return int(drivable_img.shape[0]/2.0)
        else:
            # print("drivable_path exists!")
            drivable_img = cv2.imread(drivable_path)
            if self.show_im:
                cv2.imshow("drivable",drivable_img)
                cv2.waitKey(200)
            drivable_h,drivable_w = drivable_img.shape[0],drivable_img.shape[1]
            # print(f"drivable_h:{drivable_h},drivable_w:{drivable_w}")
            p1_w =  int(drivable_w / 3.0)
            p2_w =  int(drivable_w / 2.0)
            p3_w =  int( (drivable_w*2.0) / 3.0)
            p1y,p2y,p3y = 0,0,0
            y = 0
            find_small_y = False
            while(y<=drivable_h-1 and find_small_y == False):
                if(drivable_img[y][p1_w][0]!=0):
                    find_small_y=True
                    p1y = y
                    break
                else:
                    y+=1

            y = 0
            find_small_y = False
            while(y<=drivable_h-1 and find_small_y == False):
                if(drivable_img[y][p2_w][0]!=0):
                    find_small_y=True
                    p2y = y
                    break
                else:
                    y+=1
            
            y = 0
            find_small_y = False
            while(y<=drivable_h-1 and find_small_y == False):
                if(drivable_img[y][p3_w][0]!=0):
                    find_small_y=True
                    p3y = y
                    break
                else:
                    y+=1

            # print(f"p1y:{p1y},p2y:{p2y},p3y:{p3y}")
            min,index = self.find_min_value(p1y,p2y,p3y)
            if min==0:
                # print(f"min={min} special case~~~~~")
                # if p1y==0 and p2y==0 and p3y==0:
                #     min = int(drivable_img.shape[0]/2.0)
                # elif not all([p1y,p2y,p3y]):
                #     min,index = self.find_max_value(p1y,p2y,p3y)
                min=None
                    
            # print(f"min = {min}, index={index}")

            return min,index
        
        
            
    def Find_Min_Y_Among_All_Vehicle_Bounding_Boxes(self,min,detection_path,img_h,img_w):
        # print(f"h:{img_h} w:{img_w}")
        min_rea = 999999
        find_min_area=False
        min_x=99999
        min_w=99999
        min_h=99999
        with open(detection_path,"r") as f:
            lines = f.readlines()
            for line in lines:
                find_min_area=False
                #print(line)
                la = line.split(" ")[0]
                x = int(float(line.split(" ")[1])*img_w)
                y = int(float(line.split(" ")[2])*img_h)
                w = int(float(line.split(" ")[3])*img_w)
                h = int(float(line.split(" ")[4])*img_h)
                #print(f"{la} {x} {y} {w} {h}")
                if w*h < min_rea and int(la) in self.wanted_label_list:
                    # print(f"w*h={w*h},min_rea={min_rea},x:{x},y:{y}")
                    min_rea = w*h
                    find_min_area=True
                    # print(f"find_min_area :{find_min_area} ")
                    
                if min is not None:
                    if int(la) in self.wanted_label_list and find_min_area:
                        # print(f"y:{y} min:{min}")
                        min=y
                        min_x=x
                        min_w=w
                        min_h=h
                else:
                    if int(la) in self.wanted_label_list and find_min_area:
                        # print(f"y:{y} min:{min}")
                        min=y
                        min_x=x
                        min_w=w
                        min_h=h
        if min is None:
            min = int(img_h/2.0)
        return min
        #return min,min_x,min_w,min_h
    

    def Find_Min_Y_Among_All_Vehicle_Bounding_Boxes_Ver2(self,min,detection_path,img_h,img_w):
        # print(f"h:{img_h} w:{img_w}")
        min_rea = 999999
        find_min_area=False
        min_x=99999
        min_w=99999
        min_h=99999
        with open(detection_path,"r") as f:
            lines = f.readlines()
            for line in lines:
                find_min_area=False
                #print(line)
                la = line.split(" ")[0]
                x = int(float(line.split(" ")[1])*img_w)
                y = int(float(line.split(" ")[2])*img_h)
                w = int(float(line.split(" ")[3])*img_w)
                h = int(float(line.split(" ")[4])*img_h)
                #print(f"{la} {x} {y} {w} {h}")
                if w*h < min_rea and int(la) in self.wanted_label_list:
                    # print(f"w*h={w*h},min_rea={min_rea},x:{x},y:{y}")
                    min_rea = w*h
                    find_min_area=True
                    # print(f"find_min_area :{find_min_area} ")
                    
                if min is not None:
                    if int(la) in self.wanted_label_list and find_min_area:
                        # print(f"y:{y} min:{min}")
                        min=y
                        min_x=x
                        min_w=w
                        min_h=h
                else:
                    if int(la) in self.wanted_label_list and find_min_area:
                        # print(f"y:{y} min:{min}")
                        min=y
                        min_x=x
                        min_w=w
                        min_h=h
        if min is None:
            min = int(img_h/2.0)
        return (min,min_x,min_w,min_h)
        #return min,min_x,min_w,min_h

    

    
   
