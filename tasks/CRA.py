import glob
import os
import shutil
import cv2
from engine.dataset import BaseDataset

class CRA(BaseDataset):

    def Get_CRA_Yolo_Txt_Labels(self, version=1):
        '''
            func: 
                Get Cross Road Area
            Purpose : 
                parsing the images in given directory, 
                find the center of drivable area (DCA: Drivable Center Area)
                and add bounding box information x,y,w,h 
                into label.txt of yolo format.
            input :
                self.im_dir : the image directory
                self.dataset_dir : the dataset directory
                self.save_dir : save crop image directory
            output:
                the label.txt with Drivable Center Area (DCA) bounding box
        '''
        im_path_list = glob.glob(os.path.join(self.im_dir,"*.jpg"))

        if self.data_num<len(im_path_list):
            final_wanted_img_count = self.data_num
        else:
            final_wanted_img_count = len(im_path_list)

        print(f"final_wanted_img_count = {final_wanted_img_count}")
        min_final_2 = None
        
        for i in range(final_wanted_img_count):
            im_path = r"C:/datasets/bdd100k_data_0.9/images/100k/val/b1d0a191-06deb55d.jpg"
            print(f"im_path = {im_path}")
            drivable_path,drivable_mask_path,lane_path,detection_path = self.parse_path_ver2(im_path,type=self.data_type,detect_folder=self.det_folder)
            # print(f"{i}:{im_path_list[i]}")
            # im = cv2.imread(im_path_list[i])
            # h,w = im.shape[0],im.shape[1]
            print(f"detection_path = {detection_path}")
            print(f"{i}:{im_path}")
           
            xywh,h,w = self.Get_CRA_XYWH(im_path,return_type=1)
         
            x,y = xywh[0],xywh[1]
         
            success = self.Add_CRA_Yolo_Txt_Label(xywh,detection_path,h,w,im_path)

    

    def Add_CRA_Yolo_Txt_Label(self,xywh,detection_path,h,w,im_path):
        success = 0
        xywh_not_None = True
        CRA_lxywh = None
        if xywh[0] is not None and xywh[1] is not None:
            xywh_not_None = True
        else:
            xywh_not_None = False
        # print(f"xywh[0]:{xywh[0]},xywh[1]:{xywh[1]},xywh[2]:{xywh[2]},xywh[3]:{xywh[3]},w:{w},h:{h}")
        if os.path.exists(detection_path):
            if xywh_not_None == True:
                x = float((int(float(xywh[0]/w)*1000000))/1000000)
                y = float((int(float(xywh[1]/h)*1000000))/1000000)
                w = float((int(float(xywh[2]/w)*1000000))/1000000)
                h = float((int(float(xywh[3]/h)*1000000))/1000000)
                la = self.cra_label
                # print(f"la = {la}")
                CRA_lxywh = str(la) + " " \
                            +str(x) + " " \
                            +str(y) + " " \
                            + str(w) + " " \
                            + str(h) 
            
            # print(f"x:{x},y:{y},w:{w},h:{h}")
            if not os.path.exists(self.save_txtdir):
                os.makedirs(self.save_txtdir,exist_ok=True)

            label_txt_file = detection_path.split(os.sep)[-1]
            save_label_path = os.path.join(self.save_txtdir,label_txt_file)
            
            # Copy the original label.txt into the save_label_path
            if not os.path.exists(save_label_path):
                shutil.copy(detection_path,save_label_path)
            else:
                print(f"File exists ,PASS! : {save_label_path}")
                return success
            

            if self.save_img:
                shutil.copy(im_path,self.save_txtdir)

            if CRA_lxywh is not None:
                # Add DCA label into Yolo label.txt
                with open(save_label_path,'a') as f:
                    f.write(CRA_lxywh)
                    f.write("\n")

            # print(f"{la}:{x}:{y}:{w}:{h}")
            success = 1
        else:
            success = 0
            print(f"detection_path:{detection_path} does not exists !! PASS~~~~~")
            return success

        return success

    def Get_CRA_XYWH(self,im_path,return_type=1):
        '''
        BDD100K Drivable map label :
        0: Main Lane
        1: Alter Lane
        2: BackGround
        '''
        
        drivable_path,drivable_mask_path,lane_path,detection_path = self.parse_path_ver2(im_path,type=self.data_type,detect_folder=self.det_folder)

        h = 0
        w = 0
        if os.path.exists(im_path):
            im = cv2.imread(im_path)
            im_h,im_w = im.shape[0],im.shape[1]
        #     # print(f"h:{h}, w:{w}")
        if not os.path.exists(detection_path):
            print(f"{detection_path} is not exists !! PASS~~~")
            return (None,None,None,None),None,None

       

        # dri_map = {"MainLane": 0, "AlterLane": 1, "BackGround":2}
        # lane_mapping: {0: 3,            1: 2,           2: 2,           3: 4,           4: 5, 
        #       5: 2,             6: 2,           7: 4,           8: lane_bg,     9: lane_bg,
        #       10: lane_bg,      11: lane_bg,    12: lane_bg,    13: lane_bg,    14: lane_bg,
        #       15: lane_bg,      16: 3,          17: 1,          18: 1,          19: 4,
        #       20: 5,            21: 1,          22: 1,          23: 4,          24: lane_bg,
        #       25: lane_bg,      26: lane_bg,    27: lane_bg,    28: lane_bg,    29: lane_bg,
        #       30: lane_bg,      31: lane_bg,    32: 3,          33: 2,          34: 2,  
        #       35: 4,            36: 5,          37: 2,          38: 2,          39: 4,
        #       40: lane_bg,      41: lane_bg,    42: lane_bg,    43: lane_bg,    44: lane_bg,
        #       45: lane_bg,      46: lane_bg,    47: lane_bg,    48: 3,          49: 1,
        #       50: 1,            51: 4,          52: 5,          53: 1,          54: 1,
        #       55: 4,
        #       255: lane_bg}
        lane_label = {0, 16, 32, 48}

        # names_lane:
        # 0: background
        # 1: vertical dash white line
        # 2: vertical solid white line
        # 3: crosswalk
        # 4: yellow line
        # 5: road curb

        xywh = (None,None,None,None)
        if os.path.exists(lane_path):
            
            im_lane = cv2.imread(lane_path)
            
            
            
            tmp_left_x = 9999
            tmp_right_x = 0
            tmp_top_y = 9999
            tmp_down_y = 0

            final_left_x = 9999
            final_right_x = 9999
            final_top_y = 9999
            final_down_y = 9999
            ## Find the lowest X of Main lane drivable map
            for i in range(int(h)):
                for j in range(int(w)):
                    if im_lane[i][j][0] != 0:
                        print(im_lane[i][j][0])
                        input()
                    if im_lane[i][j][0] in lane_label:
                        if j < tmp_left_x:
                            j = tmp_left_x
                        if j > tmp_right_x:
                            j = tmp_right_x
                        if i < tmp_top_y:
                            i = tmp_top_y
                        if i > tmp_down_y:
                            i = tmp_down_y
                        print("find cross road label !")

            final_left_x = tmp_left_x
            final_right_x = tmp_right_x
            final_top_y = tmp_top_y
            final_down_y = tmp_down_y
            
            w = (final_right_x - final_left_x)
            h = (final_down_y - final_top_y)
            x = final_left_x + int(w/2.0)
            y = final_top_y +  int(h/2.0)

            xywh = (x,y,w,h)
           
        else:
            xywh = (None,None,None,None)
            w,h = None,None
        
        if self.show_im and xywh[0] is not None:
            x1 = xywh[0] - int(xywh[2] / 2.0)
            x2 = xywh[0] + int(xywh[2] / 2.0)
            y1 = xywh[1] - int(xywh[3] / 2.0)
            y2 = xywh[1] + int(xywh[3] / 2.0)
            p1 = (x1,x2)
            p2 = (y1,y2)
            cv2.rectangle(im,p1,p2,(255,0,0),cv2.LINE_AA)
            cv2.rectangle(im_lane,p1,p2,(255,0,0),cv2.LINE_AA)
            cv2.imshow("image",im)
            cv2.imshow("im_lane",im_lane)
            cv2.waitKey()

        
        return xywh,h,w



                        

                    
                
              
                    

            
    








