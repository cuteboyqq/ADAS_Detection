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
            
            drivable_path,drivable_mask_path,lane_path,detection_path = self.parse_path_ver2(im_path_list[i],type=self.data_type,detect_folder=self.det_folder)
            print(f"{i}:{im_path_list[i]}")
            im = cv2.imread(im_path_list[i])
            h,w = im.shape[0],im.shape[1]
            # print(f"detection_path = {detection_path}")
            # print(f"{i}:{im_path}")
           
            xywh,im_h,im_w = self.Get_CRA_XYWH(im_path_list[i],return_type=1)
         
            x,y = xywh[0],xywh[1]
         
            success = self.Add_CRA_Yolo_Txt_Label(xywh,detection_path,im_h,im_w,im_path_list[i])

    

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
            # light green  (219,211,86) --> road curb
            # Green (127,219,86)
            # Purpol (219,86,160)
            # (178,96,219)
            #Orange (86,94,219) --> cross road
            R = 0
            G = 0
            B = 0

            for i in range(0,int(im_h),3):
                for j in range(0,int(im_w),3):
                    if im_lane[i][j][0] == 86 and im_lane[i][j][1] == 94 and im_lane[i][j][2] == 219:
                        # print(im_lane[i][j][0])
                        # input()
                        if j < tmp_left_x:
                            tmp_left_x = j
                        if j > tmp_right_x:
                            tmp_right_x = j
                        if i < tmp_top_y:
                            tmp_top_y = i
                        if i > tmp_down_y:
                            tmp_down_y = i 

            final_left_x = tmp_left_x
            final_right_x = tmp_right_x
            final_top_y = tmp_top_y
            final_down_y = tmp_down_y
            print(f"{final_left_x},{final_right_x},{final_top_y},{final_down_y}")

            if final_left_x!=9999:
                w = abs(final_right_x - final_left_x)
                h = abs(final_down_y - final_top_y)
                x = final_left_x + int(w/2.0)
                y = final_top_y +  int(h/2.0)
                print(f"xywh = {x},{y},{w},{h}")
                xywh = (x,y,w,h)
            else:
                xywh = (None,None,None,None)
                w,h = None,None
           
        else:
            xywh = (None,None,None,None)
            w,h = None,None
        
        if self.show_im and xywh[0] is not None:
            x1 = xywh[0] - int(xywh[2] / 2.0)
            x2 = xywh[0] + int(xywh[2] / 2.0)
            y1 = xywh[1] - int(xywh[3] / 2.0)
            y2 = xywh[1] + int(xywh[3] / 2.0)
            p1 = (x1,y1)
            p2 = (x2,y2)
            cv2.rectangle(im,p1,p2,(86,94,219),2,cv2.LINE_AA)
            cv2.rectangle(im_lane,p1,p2,(86,94,219),2,cv2.LINE_AA)
            cv2.imshow("image",im)
            cv2.imshow("im_lane",im_lane)
            cv2.waitKey()

        
        return xywh,im_h,im_w