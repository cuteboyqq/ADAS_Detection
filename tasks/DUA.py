import glob
import os
import shutil
import cv2
from engine.dataset import BaseDataset

DETECTTION_FOLDER = "detection-DCA-VPA_ver3"

class DUA(BaseDataset):

    def Get_DUA_Yolo_Txt_Labels(self, Get_VPA = False, Get_DUA=False, Get_Two_DUA=False, Get_Three_DUA=True):
        '''
            func: 
                Get Drivable Upper Area
            Purpose : 
                parsing the images in given directory, 
                find the Line Area (LA: Line Area)
                and add bounding box information x,y,w,h 
                into label.txt of yolo format.
            input :
                self.im_dir : the image directory
                self.dataset_dir : the dataset directory
                self.save_dir : save crop image directory
            output:
                the label.txt with Drivable Upper Area (DUA: Drivable Upper Area) bounding box
        '''
        im_path_list = glob.glob(os.path.join(self.im_dir,"*.jpg"))

        if self.data_num<len(im_path_list):
            final_wanted_img_count = self.data_num
        else:
            final_wanted_img_count = len(im_path_list)

        print(f"final_wanted_img_count = {final_wanted_img_count}")
        min_final_2 = None
        
        for i in range(final_wanted_img_count):
            drivable_path,drivable_mask_path,lane_path,detection_path = self.parse_path_ver2(im_path_list[i],type=self.data_type,detect_folder=DETECTTION_FOLDER)
            print(f"{i}:{im_path_list[i]}")
            im = cv2.imread(im_path_list[i])
            h,w = im.shape[0],im.shape[1]
            xywh = (None,None,None,None)
            xywh_m = (None,None,None,None)
          
            detection_file = detection_path.split(os.sep)[-1]
            save_txt_path = os.path.join(self.save_txtdir,detection_file)
            if os.path.exists(save_txt_path):
                print("save_txt_path exists , PASS~~~!!")
                success = 1
                continue
            if Get_VPA and Get_DUA:
                Down = self.Get_DCA_XYWH(im_path_list[i],return_type=2) #(Left_X,Right_X,Search_line_H,min_final_2)
                #(Left_X,Right_X,Search_line_H,VL_X,VL_Y,Left_M_X,Right_M_X,Search_M_line_H)
                
                Up_Mid = self.Get_Two_VPA_XYWH(im_path_list[i],return_type=2) 
                (Left_X,Right_X,Search_line_H,VL_X,VL_Y,Left_M_X,Right_M_X,Search_M_line_H) = Up_Mid
                # print(f"Left_X={Left_X},Right_X={Right_X},Search_line_H={Search_line_H},VL_X={VL_X},VL_Y={VL_Y}")
                # print(f"Left_M_X={Left_M_X},Right_M_X={Right_M_X},Search_M_line_H={Search_M_line_H}")
                
                if Down[0] is not None and Down[1] is not None and Left_X is not None and Right_X is not None \
                    and isinstance(Down[0],int)\
                    and isinstance(Down[1],int)\
                    and isinstance(Left_X,int)\
                    and isinstance(Right_X,int):
                    VP_x,VP_y,New_W,New_H = self.Get_VPA(im_path_list[i],Up_Mid,Down,use_vehicle_info=False)
                # else:
                #     print("None value, return !!")
                #     continue
                    x, y  = VP_x, VP_y
                    xywh = (VP_x,VP_y,New_W,New_H)
            

                if Up_Mid[5] is not None:
                    x_m = int((Up_Mid[5] + Up_Mid[6])/2.0)
                    y_m = int(Up_Mid[7]/2.0)
                    w_m = int(Up_Mid[6] - Up_Mid[5])
                    h_m = int(Up_Mid[7])
                    xywh_m = (x_m,y_m,w_m,h_m)
                    # print(f"x_m={xywh_m[0]}, y_m={xywh_m[1]}, w_m={xywh_m[2]}, h_m={xywh_m[3]}")
                else:
                    xywh_m = (None,None,None,None)
                    # print(f"x_m={xywh_m[0]}, y_m={xywh_m[1]}, w_m={xywh_m[2]}, h_m={xywh_m[3]}")
            
            elif Get_DUA:
                DUA_xywh,h,w = self.Get_DUA_XYWH(im_path_list[i],return_type=1) #(DUA_X,DUA_Y,DUA_W,DUA_H),h,w
            elif Get_Two_DUA:
                DUA_xywh_up,DUA_xywh_down,h,w = self.Get_Two_DUA_XYWH(im_path_list[i],return_types=1) 
                #(DUA_X,DUA_Y,DUA_W,DUA_H),(DUA_X,DUA_Y,DUA_W,DUA_H),h,w
            elif Get_Three_DUA:
                DUA_xywh_up,DUA_xywh_middle,DUA_xywh_down,h,w = self.Get_Three_DUA_XYWH(im_path_list[i],return_types=1) 

            if Get_VPA and Get_DUA:
                success = self.Add_Two_VPA_Yolo_Txt_Label(xywh,xywh_m,detection_path,h,w,im_path_list[i],add_VPA=True,add_VMA=True)
            elif Get_DUA:
                success = self.Add_DUA_Yolo_Txt_Label(DUA_xywh,detection_path,h,w,im_path_list[i])
            elif Get_Two_DUA:
                success = self.Add_Two_DUA_Yolo_Txt_Label(DUA_xywh_up,DUA_xywh_down,detection_path,h,w,im_path_list[i])
            elif Get_Three_DUA:
                success = self.Add_Three_DUA_Yolo_Txt_Label(DUA_xywh_up,DUA_xywh_middle,DUA_xywh_down,detection_path,h,w,im_path_list[i])
    
    def Get_Three_DUA_XYWH(self,im_path,return_types=1):
        '''
        func: Get DUA XYWH (Drivable Upper Area)

        BDD100K Drivable map label :
        0: Main Lane
        1: Alter Lane
        2: BackGround

        Purpose : Get the bounding box that include below:
                    1. Sky
                    2. Vanish point
                    3. Upper Drivable area of main lane
                and this bounding box information xywh for detection label (YOLO label.txt)
        input parameter : 
                    im_path : image directory path
        output :
                type 1:
                    (Middle_X,Middle_Y,DCA_W,DCA_H),(Middle_X,Middle_Y,DCA_W,DCA_H),h,w

                    Middle_X : bounding box center X
                    Middle_Y : bounding box center Y
                    DCA_W    : bounding box W
                    DCA_H    : bounding box H
                    h : image height
                    w : image width
                type 2:
                    (Left_X,Right_X,Search_line_H,VL_X,VL_Y),(Left_M_X,Right_M_X,Search_M_line_H)
                    Left_X (Left_M_X)               : VPA bounding box left x (M: Middle)
                    Right_X (Right_M_X)             : VPA bounding box right x (M: Middle)
                    Search_line_H (Search_M_line_H) : the y of the min width (width = Right_X - Left_X) (M: Middle)
                    VL_X          : min vehicle coordinate x
                    VL_Y          : mi coordinate y

        '''
        if os.path.exists(im_path):
            drivable_path,drivable_mask_path,lane_path,detection_path = self.parse_path_ver2(im_path,type=self.data_type,detect_folder=DETECTTION_FOLDER)
            im_h = 0
            im_w = 0
            if os.path.exists(drivable_path):
                im_dri = cv2.imread(drivable_mask_path)
                im_h,im_w = im_dri.shape[0],im_dri.shape[1]
                # print(f"h:{h}, w:{w}")
            if not os.path.exists(detection_path):
                print(f"{detection_path} is not exists !! PASS~~~")
                if return_types==1:
                    return (None,None,None,None),(None,None,None,None),(None,None,None,None),None,None
                else:
                    return (None,None,None)
        else:
            if return_types==1:
                return (None,None,None,None),(None,None,None,None),(None,None,None,None),None,None
            else:
                return (None,None,None)
            
        if os.path.exists(drivable_path) and os.path.exists(im_path):
            im_dri_cm = cv2.imread(drivable_path)
            im = cv2.imread(im_path)
            # DUA_down,h,w = self.Get_DUA_XYWH(im_path,return_type = 2, w_min=200, w_max=500, h_min=80, h_max=140,force_show_im=False)
            # #(Left_M_X,Right_M_X,Search_M_line_H,VL_Y),h,w
            # DUA_middle,h,w = self.Get_DUA_XYWH(im_path,return_type = 2, w_min=50, w_max=200, h_min=40, h_max=80,force_show_im=False)
            # DUA_up,h,w = self.Get_DUA_XYWH(im_path,return_type = 2, w_min=50, w_max=200, h_min=20, h_max=40,force_show_im=False)
            DUA_up,DUA_middle,DUA_down,h,w = self.Get_DUA3_XYWH(im_path,return_type=2, h_upper=(20,40), h_middel=(40,80), h_down=(80,140),force_show_im=False)
            if DUA_down[0] is not None:
                left_x = DUA_down[0]
                right_x = DUA_down[1]
                
                y_down = DUA_down[2]
                VL_Y = DUA_down[3]
                x = int((left_x + right_x) / 2.0)
                y = int((VL_Y + y_down)/2.0)
                w = int(right_x - left_x)
                h = int(y_down - VL_Y)
                DUA_xywh_down = (x,y,w,h)
            else:
                DUA_xywh_down = (None,None,None,None)

            if DUA_middle[0] is not None:
                left_x = DUA_middle[0]
                right_x = DUA_middle[1]
                
                y_down = DUA_middle[2]
                VL_Y = DUA_middle[3]
                x = int((left_x + right_x) / 2.0)
                y = int((VL_Y + y_down)/2.0)
                w = int(right_x - left_x)
                h = int(y_down - VL_Y)
                DUA_xywh_middle = (x,y,w,h)
            else:
                DUA_xywh_middle = (None,None,None,None)


            if DUA_up[0] is not None:
                left_x = DUA_up[0]
                right_x = DUA_up[1]
                
                y_down = DUA_up[2]
                VL_Y = DUA_up[3]
                x = int((left_x + right_x) / 2.0)
                y = int((VL_Y + y_down)/2.0)
                w = int(right_x - left_x)
                h = int(y_down - VL_Y)
                DUA_xywh_up = (x,y,w,h)
            else:
                DUA_xywh_up = (None,None,None,None)

            if self.show_im and return_types==1:
                if DUA_down[0] is not None:
                    left_x = DUA_down[0]
                    right_x = DUA_down[1]
                    
                    y_down = DUA_down[2]

                    # donw left point
                    cv2.circle(im_dri_cm,(left_x,y_down), 10, (0, 255, 255), 3)
                    cv2.circle(im,(right_x,y_down), 10, (0, 255, 255), 3)

                    # donw right point
                  
                   
                    cv2.circle(im_dri_cm,(right_x,y_down), 10, (255, 0, 255), 3)
                    cv2.circle(im,(right_x,y_down), 10, (255, 0, 255), 3)


                VL_Y = DUA_down[3]
                if VL_Y is not None:
                    start_point = (0,VL_Y)
                    end_point = (im_w-1,VL_Y)
                    color = (255,127,0)
                    thickness = 4
                    cv2.line(im_dri_cm, start_point, end_point, color, thickness)
                    cv2.line(im, start_point, end_point, color, thickness)


                if DUA_down[0] is not None:
                    left_x = DUA_down[0]
                    right_x = DUA_down[1]
                    
                    y_down = DUA_down[2]
                    VL_Y = DUA_down[3]
                    p1 =(left_x,y_down)
                    p2 = (right_x,VL_Y)
                    cv2.rectangle(im_dri_cm, p1, p2, (0,127,127) , 3, cv2.LINE_AA)
                    cv2.rectangle(im, p1, p2, (0,127,127) , 3, cv2.LINE_AA)

                if DUA_middle[0] is not None:
                    left_x = DUA_middle[0]
                    right_x = DUA_middle[1]
                    
                    y_down = DUA_middle[2]
                    VL_Y = DUA_middle[3]
                    p1 =(left_x,y_down)
                    p2 = (right_x,VL_Y)
                    cv2.rectangle(im_dri_cm, p1, p2, (127,127,0) , 3, cv2.LINE_AA)
                    cv2.rectangle(im, p1, p2, (127,127,0) , 3, cv2.LINE_AA)

                if DUA_up[0] is not None:
                    left_x = DUA_up[0]
                    right_x = DUA_up[1]
                    
                    y_down = DUA_up[2]
                    VL_Y = DUA_up[3]
                    p1 =(left_x,y_down)
                    p2 = (right_x,VL_Y)
                    cv2.rectangle(im_dri_cm, p1, p2, (50,200,127) , 3, cv2.LINE_AA)
                    cv2.rectangle(im, p1, p2, (50,200,127) , 3, cv2.LINE_AA)

              

                cv2.imshow("drivable image",im_dri_cm)
                cv2.imshow("image",im)
                cv2.waitKey()
        else:
            if return_types==1:
                return (None,None,None,None),(None,None,None,None),(None,None,None,None),None,None
            else:
                return (None,None,None)
            

        return DUA_xywh_up,DUA_xywh_middle,DUA_xywh_down,im_h,im_w
        return NotImplemented

    def Add_Three_DUA_Yolo_Txt_Label(self,DUA_xywh_up,DUA_xywh_middle,DUA_xywh_down,detection_path,h,w,im_path):
        success = 0
        im_w = w
        im_h = h
        # print(f"im_w={im_w},im_h ={im_h}")
       
        DUA_lxywh_up = None
        DUA_lxywh_middle = None
        DUA_lxywh_down = None
        
        xywh_up_not_None = True
        if DUA_xywh_up[0] is not None and DUA_xywh_up[1] is not None:
            xywh_up_not_None = True
        else:
            xywh_up_not_None = False

        xywh_middle_not_None = True
        if DUA_xywh_middle[0] is not None and DUA_xywh_middle[1] is not None:
            xywh_middle_not_None = True
        else:
            xywh_middle_not_None = False

        xywh_down_not_None = True
        if DUA_xywh_down[0] is not None and DUA_xywh_down[1] is not None:
            xywh_down_not_None = True
        else:
            xywh_down_not_None = False

        # print(f"xywh[0]:{xywh[0]},xywh[1]:{xywh[1]},xywh[2]:{xywh[2]},xywh[3]:{xywh[3]},w:{w},h:{h}")
        if os.path.exists(detection_path):
            
            if xywh_up_not_None==True:
                # middle VPA bounding box
                # print(f"xywh_m[0] = {xywh_m[0]}, xywh_m[1]={xywh_m[1]}, xywh_m[2]={xywh_m[2]}. xywh_m[3]={xywh_m[3]}")
                # print(f"w={w}, h={h}")
                x_up = float((int(float(DUA_xywh_up[0]/im_w)*1000000))/1000000)
                y_up = float((int(float(DUA_xywh_up[1]/im_h)*1000000))/1000000)
                w_up = float((int(float(DUA_xywh_up[2]/im_w)*1000000))/1000000)
                h_up = float((int(float(DUA_xywh_up[3]/im_h)*1000000))/1000000)
                la_up = self.dua_up_label
                # print(f"la = {la}")
                DUA_lxywh_up = str(la_up) + " " \
                            + str(x_up) + " " \
                            + str(y_up) + " " \
                            + str(w_up) + " " \
                            + str(h_up)
            
            if xywh_middle_not_None==True:
                # middle VPA bounding box
                # print(f"xywh_m[0] = {xywh_m[0]}, xywh_m[1]={xywh_m[1]}, xywh_m[2]={xywh_m[2]}. xywh_m[3]={xywh_m[3]}")
                # print(f"w={w}, h={h}")
                x_mid = float((int(float(DUA_xywh_middle[0]/im_w)*1000000))/1000000)
                y_mid = float((int(float(DUA_xywh_middle[1]/im_h)*1000000))/1000000)
                w_mid = float((int(float(DUA_xywh_middle[2]/im_w)*1000000))/1000000)
                h_mid = float((int(float(DUA_xywh_middle[3]/im_h)*1000000))/1000000)
                la_mid = self.dua_mid_label
                # print(f"la = {la}")
                DUA_lxywh_middle = str(la_mid) + " " \
                            + str(x_mid) + " " \
                            + str(y_mid) + " " \
                            + str(w_mid) + " " \
                            + str(h_mid)


            if xywh_down_not_None==True:
                # middle VPA bounding box
                # print(f"xywh_m[0] = {xywh_m[0]}, xywh_m[1]={xywh_m[1]}, xywh_m[2]={xywh_m[2]}. xywh_m[3]={xywh_m[3]}")
                # print(f"w={w}, h={h}")
                x_down = float((int(float(DUA_xywh_down[0]/im_w)*1000000))/1000000)
                y_down = float((int(float(DUA_xywh_down[1]/im_h)*1000000))/1000000)
                w_down = float((int(float(DUA_xywh_down[2]/im_w)*1000000))/1000000)
                h_down = float((int(float(DUA_xywh_down[3]/im_h)*1000000))/1000000)
                la_down = self.dua_down_label
                # print(f"la = {la}")
                DUA_lxywh_down = str(la_down) + " " \
                            + str(x_down) + " " \
                            + str(y_down) + " " \
                            + str(w_down) + " " \
                            + str(h_down)
            
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


            if DUA_lxywh_up is not None:
                # Add VPA Middle label into Yolo label.txt
                with open(save_label_path,'a') as f:
                    f.write("\n")
                    f.write(DUA_lxywh_up)
                    if DUA_lxywh_middle is not None:
                        f.write("\n")
                        f.write(DUA_lxywh_middle)
                    if DUA_lxywh_down is not None:
                        f.write("\n")
                        f.write(DUA_lxywh_down)

            # if DUA_lxywh_middle is not None:
            #     # Add VPA Middle label into Yolo label.txt
            #     with open(save_label_path,'a') as f:
            #         f.write("\n")
            #         f.write(DUA_lxywh_middle)
            
            # if DUA_lxywh_down is not None:
            #     # Add VPA Middle label into Yolo label.txt
            #     with open(save_label_path,'a') as f:
            #         f.write("\n")
            #         f.write(DUA_lxywh_down)
            # print(f"{la}:{x}:{y}:{w}:{h}")           
            success = 1
        else:
            success = 0
            print(f"detection_path:{detection_path} does not exists !! PASS~~~~~")
            return success

        return success

    def Get_Two_DUA_XYWH(self,im_path,return_types=1):
        '''
        func: Get DUA XYWH (Drivable Upper Area)

        BDD100K Drivable map label :
        0: Main Lane
        1: Alter Lane
        2: BackGround

        Purpose : Get the bounding box that include below:
                    1. Sky
                    2. Vanish point
                    3. Upper Drivable area of main lane
                and this bounding box information xywh for detection label (YOLO label.txt)
        input parameter : 
                    im_path : image directory path
        output :
                type 1:
                    (Middle_X,Middle_Y,DCA_W,DCA_H),(Middle_X,Middle_Y,DCA_W,DCA_H),h,w

                    Middle_X : bounding box center X
                    Middle_Y : bounding box center Y
                    DCA_W    : bounding box W
                    DCA_H    : bounding box H
                    h : image height
                    w : image width
                type 2:
                    (Left_X,Right_X,Search_line_H,VL_X,VL_Y),(Left_M_X,Right_M_X,Search_M_line_H)
                    Left_X (Left_M_X)               : VPA bounding box left x (M: Middle)
                    Right_X (Right_M_X)             : VPA bounding box right x (M: Middle)
                    Search_line_H (Search_M_line_H) : the y of the min width (width = Right_X - Left_X) (M: Middle)
                    VL_X          : min vehicle coordinate x
                    VL_Y          : mi coordinate y

        '''
        if os.path.exists(im_path):
            drivable_path,drivable_mask_path,lane_path,detection_path = self.parse_path_ver2(im_path,type=self.data_type,detect_folder=DETECTTION_FOLDER)
            im_h = 0
            im_w = 0
            if os.path.exists(drivable_path):
                im_dri = cv2.imread(drivable_mask_path)
                im_h,im_w = im_dri.shape[0],im_dri.shape[1]
                # print(f"h:{h}, w:{w}")
            if not os.path.exists(detection_path):
                print(f"{detection_path} is not exists !! PASS~~~")
                if return_types==1:
                    return (None,None,None,None),(None,None,None,None),None,None
                else:
                    return (None,None,None)
        else:
            if return_types==1:
                return (None,None,None,None),(None,None,None,None),None,None
            else:
                return (None,None,None)
            
        if os.path.exists(drivable_path) and os.path.exists(im_path):
            im_dri_cm = cv2.imread(drivable_path)
            im = cv2.imread(im_path)
            DUA_down,h,w = self.Get_DUA_XYWH(im_path,return_type = 2, w_min=200, w_max=500, h_min=40, h_max=80,force_show_im=False)
            #(Left_M_X,Right_M_X,Search_M_line_H,VL_Y),h,w
            DUA_up,h,w = self.Get_DUA_XYWH(im_path,return_type = 2, w_min=50, w_max=200, h_min=20, h_max=40,force_show_im=False)

            if DUA_down[0] is not None:
                left_x = DUA_down[0]
                right_x = DUA_down[1]
                
                y_down = DUA_down[2]
                VL_Y = DUA_down[3]
                x = int((left_x + right_x) / 2.0)
                y = int((VL_Y + y_down)/2.0)
                w = int(right_x - left_x)
                h = int(y_down - VL_Y)
                DUA_xywh_down = (x,y,w,h)
            else:
                DUA_xywh_down = (None,None,None,None)

            if DUA_up[0] is not None:
                left_x = DUA_up[0]
                right_x = DUA_up[1]
                
                y_down = DUA_up[2]
                VL_Y = DUA_up[3]
                x = int((left_x + right_x) / 2.0)
                y = int((VL_Y + y_down)/2.0)
                w = int(right_x - left_x)
                h = int(y_down - VL_Y)
                DUA_xywh_up = (x,y,w,h)
            else:
                DUA_xywh_up = (None,None,None,None)

            if self.show_im and return_types==1:
                if DUA_down[0] is not None:
                    left_x = DUA_down[0]
                    right_x = DUA_down[1]
                    
                    y_down = DUA_down[2]

                    # donw left point
                    cv2.circle(im_dri_cm,(left_x,y_down), 10, (0, 255, 255), 3)
                    cv2.circle(im,(right_x,y_down), 10, (0, 255, 255), 3)

                    # donw right point
                  
                   
                    cv2.circle(im_dri_cm,(right_x,y_down), 10, (255, 0, 255), 3)
                    cv2.circle(im,(right_x,y_down), 10, (255, 0, 255), 3)


                    # middle vertical line
                    # start_point = (Middle_X,0)
                    # end_point = (Middle_X,h)
                    # color = (255,127,0)
                    # thickness = 4
                    # cv2.line(im_dri_cm, start_point, end_point, color, thickness)
                    # cv2.line(im, start_point, end_point, color, thickness)
                VL_Y = DUA_down[3]
                if VL_Y is not None:
                    start_point = (0,VL_Y)
                    end_point = (im_w-1,VL_Y)
                    color = (255,127,0)
                    thickness = 4
                    cv2.line(im_dri_cm, start_point, end_point, color, thickness)
                    cv2.line(im, start_point, end_point, color, thickness)
                if DUA_down[0] is not None:
                    left_x = DUA_down[0]
                    right_x = DUA_down[1]
                    
                    y_down = DUA_down[2]
                    VL_Y = DUA_down[3]
                    p1 =(left_x,y_down)
                    p2 = (right_x,VL_Y)
                    cv2.rectangle(im_dri_cm, p1, p2, (0,127,127) , 3, cv2.LINE_AA)
                    cv2.rectangle(im, p1, p2, (0,127,127) , 3, cv2.LINE_AA)
                
                if DUA_up[0] is not None:
                    left_x = DUA_up[0]
                    right_x = DUA_up[1]
                    
                    y_down = DUA_up[2]
                    VL_Y = DUA_up[3]
                    p1 =(left_x,y_down)
                    p2 = (right_x,VL_Y)
                    cv2.rectangle(im_dri_cm, p1, p2, (50,200,127) , 3, cv2.LINE_AA)
                    cv2.rectangle(im, p1, p2, (50,200,127) , 3, cv2.LINE_AA)

                # if DUA_xywh_down[0] is not None:
                #     x = DUA_xywh_down[0]
                #     y = DUA_xywh_down[1]
                #     w = DUA_xywh_down[2]
                #     h = DUA_xywh_down[3]
                #     p1_x = x - int(w/2.0)
                #     p1_y = y - int(h/2.0)
                #     p1 = (p1_x,p1_y)

                #     p2_x = x + int(w/2.0)
                #     p2_y = y + int(h/2.0)
                #     p2 = (p2_x,p2_y) 
                #     cv2.rectangle(im_dri_cm, p1, p2, (0,255,127) , 3, cv2.LINE_AA)
                #     cv2.rectangle(im, p1, p2, (0,255,127) , 3, cv2.LINE_AA)

                cv2.imshow("drivable image",im_dri_cm)
                cv2.imshow("image",im)
                cv2.waitKey()
        else:
            return None,None,im_h,im_w
            

        return DUA_xywh_up,DUA_xywh_down,im_h,im_w
    
    def Add_Two_DUA_Yolo_Txt_Label(self,DUA_xywh_up,DUA_xywh_down,detection_path,h,w,im_path):
        success = 0
        im_w = w
        im_h = h
        # print(f"im_w={im_w},im_h ={im_h}")
       
        DUA_lxywh_up = None
        DUA_lxywh_down = None
        
        xywh_up_not_None = True
        if DUA_xywh_up[0] is not None and DUA_xywh_up[1] is not None:
            xywh_up_not_None = True
        else:
            xywh_up_not_None = False


        xywh_down_not_None = True
        if DUA_xywh_down[0] is not None and DUA_xywh_down[1] is not None:
            xywh_down_not_None = True
        else:
            xywh_down_not_None = False
        # print(f"xywh[0]:{xywh[0]},xywh[1]:{xywh[1]},xywh[2]:{xywh[2]},xywh[3]:{xywh[3]},w:{w},h:{h}")
        if os.path.exists(detection_path):
            
            if xywh_up_not_None==True:
                # middle VPA bounding box
                # print(f"xywh_m[0] = {xywh_m[0]}, xywh_m[1]={xywh_m[1]}, xywh_m[2]={xywh_m[2]}. xywh_m[3]={xywh_m[3]}")
                # print(f"w={w}, h={h}")
                x_up = float((int(float(DUA_xywh_up[0]/im_w)*1000000))/1000000)
                y_up = float((int(float(DUA_xywh_up[1]/im_h)*1000000))/1000000)
                w_up = float((int(float(DUA_xywh_up[2]/im_w)*1000000))/1000000)
                h_up = float((int(float(DUA_xywh_up[3]/im_h)*1000000))/1000000)
                la_up = self.dua_up_label
                # print(f"la = {la}")
                DUA_lxywh_up = str(la_up) + " " \
                            + str(x_up) + " " \
                            + str(y_up) + " " \
                            + str(w_up) + " " \
                            + str(h_up)
            

            if xywh_down_not_None==True:
                # middle VPA bounding box
                # print(f"xywh_m[0] = {xywh_m[0]}, xywh_m[1]={xywh_m[1]}, xywh_m[2]={xywh_m[2]}. xywh_m[3]={xywh_m[3]}")
                # print(f"w={w}, h={h}")
                x_down = float((int(float(DUA_xywh_down[0]/im_w)*1000000))/1000000)
                y_down = float((int(float(DUA_xywh_down[1]/im_h)*1000000))/1000000)
                w_down = float((int(float(DUA_xywh_down[2]/im_w)*1000000))/1000000)
                h_down = float((int(float(DUA_xywh_down[3]/im_h)*1000000))/1000000)
                la_down = self.dua_down_label
                # print(f"la = {la}")
                DUA_lxywh_down = str(la_down) + " " \
                            + str(x_down) + " " \
                            + str(y_down) + " " \
                            + str(w_down) + " " \
                            + str(h_down)
            
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


            if DUA_lxywh_up is not None:
                # Add VPA Middle label into Yolo label.txt
                with open(save_label_path,'a') as f:
                    f.write("\n")
                    f.write(DUA_lxywh_up)
            
            if DUA_lxywh_down is not None:
                # Add VPA Middle label into Yolo label.txt
                with open(save_label_path,'a') as f:
                    f.write("\n")
                    f.write(DUA_lxywh_down)
            # print(f"{la}:{x}:{y}:{w}:{h}")           
            success = 1
        else:
            success = 0
            print(f"detection_path:{detection_path} does not exists !! PASS~~~~~")
            return success

        return success
    
    def Add_DUA_Yolo_Txt_Label(self,xywh_m,detection_path,h,w,im_path):
        success = 0
        im_w = w
        im_h = h
      
       
        VPA_lxywh_M = None
      
        
        xywh_m_not_None = True
        if xywh_m[0] is not None and xywh_m[1] is not None:
            xywh_m_not_None = True
        else:
            xywh_m_not_None = False
        # print(f"xywh[0]:{xywh[0]},xywh[1]:{xywh[1]},xywh[2]:{xywh[2]},xywh[3]:{xywh[3]},w:{w},h:{h}")
        if os.path.exists(detection_path):
            
            if xywh_m_not_None==True:
                # middle VPA bounding box
                # print(f"xywh_m[0] = {xywh_m[0]}, xywh_m[1]={xywh_m[1]}, xywh_m[2]={xywh_m[2]}. xywh_m[3]={xywh_m[3]}")
                # print(f"w={w}, h={h}")
                x_m = float((int(float(xywh_m[0]/im_w)*1000000))/1000000)
                y_m = float((int(float(xywh_m[1]/im_h)*1000000))/1000000)
                w_m = float((int(float(xywh_m[2]/im_w)*1000000))/1000000)
                h_m = float((int(float(xywh_m[3]/im_h)*1000000))/1000000)
                la_m = self.vpam_label
                # print(f"la = {la}")
                VPA_lxywh_M = str(la_m) + " " \
                            +str(x_m) + " " \
                            +str(y_m) + " " \
                            + str(w_m) + " " \
                            + str(h_m)
                # print(f"VPA_lxywh_M = {VPA_lxywh_M}")
            
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


            if VPA_lxywh_M is not None:
                # Add VPA Middle label into Yolo label.txt
                with open(save_label_path,'a') as f:
                    f.write("\n")
                    f.write(VPA_lxywh_M)
            # print(f"{la}:{x}:{y}:{w}:{h}")           
            success = 1
        else:
            success = 0
            print(f"detection_path:{detection_path} does not exists !! PASS~~~~~")
            return success

        return success

    def Add_Two_VPA_Yolo_Txt_Label(self,xywh,xywh_m,detection_path,h,w,im_path, add_VPA=False,add_VMA=True):
        success = 0
        im_w = w
        im_h = h
        xywh_not_None = True
        VPA_lxywh = None
        VPA_lxywh_M = None
        if xywh[0] is not None and xywh[1] is not None:
            xywh_not_None = True
        else:
            xywh_not_None = False
        
        xywh_m_not_None = True
        if xywh_m[0] is not None and xywh_m[1] is not None:
            xywh_m_not_None = True
        else:
            xywh_m_not_None = False
        # print(f"xywh[0]:{xywh[0]},xywh[1]:{xywh[1]},xywh[2]:{xywh[2]},xywh[3]:{xywh[3]},w:{w},h:{h}")
        if os.path.exists(detection_path):
            if xywh_not_None == True and add_VPA==True:
                x = float((int(float(xywh[0]/im_w)*1000000))/1000000)
                y = float((int(float(xywh[1]/im_h)*1000000))/1000000)
                w = float((int(float(xywh[2]/im_w)*1000000))/1000000)
                h = float((int(float(xywh[3]/im_h)*1000000))/1000000)
                la = self.vpa_label
                # print(f"la = {la}")
                VPA_lxywh = str(la) + " " \
                            +str(x) + " " \
                            +str(y) + " " \
                            + str(w) + " " \
                            + str(h)
                
            if xywh_m_not_None==True and add_VMA==True:
                # middle VPA bounding box
                # print(f"xywh_m[0] = {xywh_m[0]}, xywh_m[1]={xywh_m[1]}, xywh_m[2]={xywh_m[2]}. xywh_m[3]={xywh_m[3]}")
                # print(f"w={w}, h={h}")
                x_m = float((int(float(xywh_m[0]/im_w)*1000000))/1000000)
                y_m = float((int(float(xywh_m[1]/im_h)*1000000))/1000000)
                w_m = float((int(float(xywh_m[2]/im_w)*1000000))/1000000)
                h_m = float((int(float(xywh_m[3]/im_h)*1000000))/1000000)
                la_m = self.vpam_label
                # print(f"la = {la}")
                VPA_lxywh_M = str(la_m) + " " \
                            +str(x_m) + " " \
                            +str(y_m) + " " \
                            + str(w_m) + " " \
                            + str(h_m)
                # print(f"VPA_lxywh_M = {VPA_lxywh_M}")
            
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

            if VPA_lxywh is not None:
                # Add VPA label into Yolo label.txt
                with open(save_label_path,'a') as f:
                    f.write("\n")
                    f.write(VPA_lxywh)

            if VPA_lxywh_M is not None:
                # Add VPA Middle label into Yolo label.txt
                with open(save_label_path,'a') as f:
                    f.write("\n")
                    f.write(VPA_lxywh_M)
            # print(f"{la}:{x}:{y}:{w}:{h}")
                    
            success = 1
        else:
            success = 0
            print(f"detection_path:{detection_path} does not exists !! PASS~~~~~")
            return success

        return success
    

    def Get_DUA3_XYWH(self,im_path,return_type=1, h_upper=(20,40), h_middel=(40,80), h_down=(80,140),force_show_im=True):
        '''
        func: Get DUA XYWH (Drivable Upper Area)

        Purpose : Get the bounding box that include below:
                    1. Vanish point
                    2. Upper Drivable area of main lane
                and this bounding box information xywh for detection label (YOLO label.txt)
        input parameter : 
                    im_path : image directory path
        output :
                type 1:
                    (Middle_X,Middle_Y,DCA_W,DCA_H),h,w

                    Middle_X : bounding box center X
                    Middle_Y : bounding box center Y
                    DCA_W    : bounding box W
                    DCA_H    : bounding box H
                    h : image height
                    w : image width
                type 2:
                    (Left_X,Right_X,Search_line_H,VL_X,VL_Y),(Left_M_X,Right_M_X,Search_M_line_H)
                    Left_X (Left_M_X)               : VPA bounding box left x (M: Middle)
                    Right_X (Right_M_X)             : VPA bounding box right x (M: Middle)
                    Search_line_H (Search_M_line_H) : the y of the min width (width = Right_X - Left_X) (M: Middle)
                    VL_X          : min vehicle coordinate x
                    VL_Y          : mi coordinate y
        '''
        drivable_path,drivable_mask_path,lane_path,detection_path = self.parse_path_ver2(im_path,type=self.data_type,detect_folder=DETECTTION_FOLDER)
        h = 0
        w = 0
        if os.path.exists(drivable_path):
            im_dri = cv2.imread(drivable_mask_path)
            h,w = im_dri.shape[0],im_dri.shape[1]
            # print(f"h:{h}, w:{w}")
        if not os.path.exists(detection_path):
            print(f"{detection_path} is not exists !! PASS~~~")
            if return_type==1:
                return (None,None,None,None),(None,None,None,None),(None,None,None,None),None,None
            else:
                return (None,None,None,None),(None,None,None,None),(None,None,None,None),None,None

        min_final,index = self.Get_Min_y_In_Drivable_Area(drivable_path)    
        VL = self.Find_Min_Y_Among_All_Vehicle_Bounding_Boxes_Ver2(min_final,detection_path,h,w)
        VL_Y,VL_X,VL_W,VL_H = VL
        # print(f"VL_Y:{VL_Y},VL_X:{VL_X},VL_W:{VL_W},VL_H:{VL_H}")
        dri_map = {"MainLane": 0, "AlterLane": 1, "BackGround":2}
        
        if os.path.exists(drivable_path):
            
            im_dri_cm = cv2.imread(drivable_path)
            im = cv2.imread(im_path)

            Lowest_H = 0
            Left_tmp_X = 0
            Right_tmp_X = 0
            find_left_tmp_x = False
            find_right_tmp_x = False
    
            Top_Y = 0
            temp_y = 0
            find_top_y = False
            DCA_W = 0
            DCA_H = 0


            # initialize upper parameters
            main_lane_width_up = 0
            main_lane_upper_width = 9999
            Final_Up_Left_X = 0
            Final_Up_Right_X = 0
            Search_Up_line_H = 0

            # initialize middle parameters
            main_lane_width_mid = 0
            main_lane_middle_width = 9999
            Final_Mid_Left_X = 0
            Final_Mid_Right_X = 0
            Search_Mid_line_H = 0

            # initialize down parameters
            main_lane_width_down = 0
            main_lane_down_width = 9999
            Final_Down_Left_X = 0
            Final_Down_Right_X = 0
            Search_Down_line_H = 0

            ## Find the lowest X of Main lane drivable map
            for i in range(int(h)):
                find_left_tmp_x = False
                find_right_tmp_x = False
                for j in range(int(w)):

                    if int(im_dri[i][j][0]) == dri_map["MainLane"]:
                        if i>Lowest_H:
                            Lowest_H = i
                        if find_top_y==False:
                            Top_Y = i
                            find_top_y = True
                    if int(im_dri[i][j][0]) == dri_map["MainLane"] and find_left_tmp_x==False:
                        Left_tmp_X = j
                        find_left_tmp_x = True

                    if int(im_dri[i][j][0]) == dri_map["BackGround"] and find_right_tmp_x==False and find_left_tmp_x==True:
                        Right_tmp_X = j
                        find_right_tmp_x = True
                        temp_y = i
                
                # print(f"find_left_tmp_x:{find_left_tmp_x}")
                # print(f"find_right_tmp_x:{find_right_tmp_x}")
                tmp_main_lane_width = abs(Right_tmp_X - Left_tmp_X)


                '''
                Find the upper main lane drivable area width, 
                and not at the vanish point
                '''
                if tmp_main_lane_width>main_lane_width_up\
                    and find_left_tmp_x==True \
                    and find_right_tmp_x==True \
                    and abs(i-Top_Y)>h_upper[0] and abs(i-Top_Y) < h_upper[1]:
                 
                    main_lane_width_up = tmp_main_lane_width
                    Final_Up_Left_X = Left_tmp_X
                    Final_Up_Right_X = Right_tmp_X
                    Search_Up_line_H = i
                '''
                Find the middle main lane drivable area width, 
                and not at the vanish point
                '''
                if tmp_main_lane_width>main_lane_width_mid\
                    and find_left_tmp_x==True \
                    and find_right_tmp_x==True \
                    and abs(i-Top_Y)>h_middel[0] and abs(i-Top_Y) < h_middel[1]:
                 
                    main_lane_width_mid = tmp_main_lane_width
                    Final_Mid_Left_X = Left_tmp_X
                    Final_Mid_Right_X = Right_tmp_X
                    Search_Mid_line_H = i

                '''
                Find the down main lane drivable area width, 
                and not at the vanish point
                '''
                if tmp_main_lane_width>main_lane_width_down\
                    and find_left_tmp_x==True \
                    and find_right_tmp_x==True \
                    and abs(i-Top_Y)>h_down[0] and abs(i-Top_Y) < h_down[1]:
                 
                    main_lane_width_down = tmp_main_lane_width
                    Final_Down_Left_X = Left_tmp_X
                    Final_Down_Right_X = Right_tmp_X
                    Search_Down_line_H = i
                     
            '''
            Get Upper bounding box "
            1. left x
            2. right x
            3. lower bound y 
            '''
            if Final_Up_Left_X==0 and Final_Up_Right_X==0:
                Left_Up_X = None
                Right_Up_X = None
            else:
                Left_Up_X = Final_Up_Left_X
                Right_Up_X = Final_Up_Right_X
                # print(f"Left_M_X = {Left_M_X}")
                # print(f"Right_M_X = {Right_M_X}")
                # print(f"Search_M_line_H = {Search_M_line_H}")
            
            '''
            Get Middle bounding box
            1. left x
            2. right x
            3. lower bound y 
            '''
            if Final_Mid_Left_X==0 and Final_Mid_Right_X==0:
                Left_Mid_X = None
                Right_Mid_X = None
            else:
                Left_Mid_X = Final_Mid_Left_X
                Right_Mid_X = Final_Mid_Right_X

            '''
            Get Down bounding box
            1. left x
            2. right x
            3. lower bound y 
            '''
            if Final_Down_Left_X==0 and Final_Down_Right_X==0:
                Left_Down_X = None
                Right_Down_X = None
            else:
                Left_Down_X = Final_Down_Left_X
                Right_Down_X = Final_Down_Right_X

                # print(f"line Y :{Search_line_H} Left_X:{Left_X}, Right_X:{Right_X} Middle_X:{Middle_X}")

            '''
            Get Upper DUA xywh
            '''
            if Left_Up_X is not None and Right_Up_X is not None:
                # Middle bounding box
                DUA_Up_X = int((Left_Up_X + Right_Up_X)/2.0)
                DUA_Up_Y = int(Search_Up_line_H + VL_Y /2.0)
                DUA_Up_W = abs(Right_Up_X - Left_Up_X)
                DUA_Up_H = abs(int(Search_Up_line_H - VL_Y + 1))
            else:
                DUA_Up_X = None
                DUA_Up_Y = None
                DUA_Up_W = None
                DUA_Up_H = None
                Search_Up_line_H = None

            '''
            Get Middle DUA xywh
            '''
            if Left_Mid_X is not None and Right_Mid_X is not None:
                # Middle bounding box
                DUA_Mid_X = int((Left_Mid_X + Right_Mid_X)/2.0)
                DUA_Mid_Y = int(Search_Mid_line_H + VL_Y /2.0)
                DUA_Mid_W = abs(Right_Mid_X - Left_Mid_X)
                DUA_Mid_H = abs(int(Search_Mid_line_H - VL_Y + 1))
            else:
                DUA_Mid_X = None
                DUA_Mid_Y = None
                DUA_Mid_W = None
                DUA_Mid_H = None
                Search_Mid_line_H = None

            '''
            Get Down DUA xywh
            '''
            if Left_Down_X is not None and Right_Down_X is not None:
                # Middle bounding box
                DUA_Down_X = int((Left_Down_X + Right_Down_X)/2.0)
                DUA_Down_Y = int(Search_Down_line_H + VL_Y /2.0)
                DUA_Down_W = abs(Right_Down_X - Left_Down_X)
                DUA_Down_H = abs(int(Search_Down_line_H - VL_Y + 1))
            else:
                DUA_Down_X = None
                DUA_Down_Y = None
                DUA_Down_W = None
                DUA_Down_H = None
                Search_Down_line_H = None
            
            
            if self.show_im and return_type==1 and force_show_im:
                
                # # upper X M
                # cv2.circle(im_dri_cm,(Left_M_X,Search_M_line_H), 10, (0, 255, 255), 3)
                # cv2.circle(im,(Left_M_X,Search_M_line_H), 10, (0, 255, 255), 3)
                # # right X M
                # cv2.circle(im_dri_cm,(Right_M_X,Search_M_line_H), 10, (255, 0, 255), 3)
                # cv2.circle(im,(Right_M_X,Search_M_line_H), 10, (255, 0, 255), 3)

                
                if Left_Mid_X is not None:
                    cv2.rectangle(im_dri_cm, (Left_Mid_X, 0), (Right_Mid_X, Search_Mid_line_H), (0,127,127) , 3, cv2.LINE_AA)
                    cv2.rectangle(im, (Left_Mid_X, 0), (Right_Mid_X, Search_Mid_line_H), (0,127,127) , 3, cv2.LINE_AA)

                cv2.imshow("drivable image",im_dri_cm)
                cv2.imshow("image",im)
                cv2.waitKey()
        else:
            if return_type==1:
                return (None,None,None,None),(None,None,None,None),(None,None,None,None),None,None
            if return_type==2:
                return (None,None,None,None),(None,None,None,None),(None,None,None,None),None,None
        
        # print(f"Middle_X:{Middle_X},Middle_Y:{Middle_Y},DCA_W:{DCA_W},DCA_H:{DCA_H}")
        if return_type==1:
            return (DUA_Up_X,DUA_Up_Y,DUA_Up_W,DUA_Up_H),(DUA_Mid_X,DUA_Mid_Y,DUA_Mid_W,DUA_Mid_H),(DUA_Down_X,DUA_Down_Y,DUA_Down_W,DUA_Down_H),h,w
        if return_type==2:
            return (Left_Up_X,Right_Up_X,Search_Up_line_H,VL_Y),(Left_Mid_X,Right_Mid_X,Search_Mid_line_H,VL_Y),(Left_Down_X,Right_Down_X,Search_Down_line_H,VL_Y),h,w

    def Get_DUA_XYWH(self,im_path,return_type=1, w_min=200, w_max=400, h_min=50, h_max=150,force_show_im=True):
        '''
        func: Get DUA XYWH (Drivable Upper Area)

        BDD100K Drivable map label :
        0: Main Lane
        1: Alter Lane
        2: BackGround

        Purpose : Get the bounding box that include below:
                    1. Sky
                    2. Vanish point
                    3. Upper Drivable area of main lane
                and this bounding box information xywh for detection label (YOLO label.txt)
        input parameter : 
                    im_path : image directory path
        output :
                type 1:
                    (Middle_X,Middle_Y,DCA_W,DCA_H),h,w

                    Middle_X : bounding box center X
                    Middle_Y : bounding box center Y
                    DCA_W    : bounding box W
                    DCA_H    : bounding box H
                    h : image height
                    w : image width
                type 2:
                    (Left_X,Right_X,Search_line_H,VL_X,VL_Y),(Left_M_X,Right_M_X,Search_M_line_H)
                    Left_X (Left_M_X)               : VPA bounding box left x (M: Middle)
                    Right_X (Right_M_X)             : VPA bounding box right x (M: Middle)
                    Search_line_H (Search_M_line_H) : the y of the min width (width = Right_X - Left_X) (M: Middle)
                    VL_X          : min vehicle coordinate x
                    VL_Y          : mi coordinate y
        '''
        drivable_path,drivable_mask_path,lane_path,detection_path = self.parse_path_ver2(im_path,type=self.data_type,detect_folder=DETECTTION_FOLDER)
        h = 0
        w = 0
        if os.path.exists(drivable_path):
            im_dri = cv2.imread(drivable_mask_path)
            h,w = im_dri.shape[0],im_dri.shape[1]
            # print(f"h:{h}, w:{w}")
        if not os.path.exists(detection_path):
            print(f"{detection_path} is not exists !! PASS~~~")
            if return_type==1:
                return (None,None,None,None),None,None
            else:
                return (None,None,None)

        min_final,index = self.Get_Min_y_In_Drivable_Area(drivable_path)    
        VL = self.Find_Min_Y_Among_All_Vehicle_Bounding_Boxes_Ver2(min_final,detection_path,h,w)
        VL_Y,VL_X,VL_W,VL_H = VL
        # print(f"VL_Y:{VL_Y},VL_X:{VL_X},VL_W:{VL_W},VL_H:{VL_H}")
        dri_map = {"MainLane": 0, "AlterLane": 1, "BackGround":2}
        
        if os.path.exists(drivable_path):
            
            im_dri_cm = cv2.imread(drivable_path)
            im = cv2.imread(im_path)

            Lowest_H = 0
            Left_tmp_X = 0
            Right_tmp_X = 0
            find_left_tmp_x = False
            find_right_tmp_x = False
    
            Top_Y = 0
            temp_y = 0
            find_top_y = False
            DCA_W = 0
            DCA_H = 0
            # middle parameters
            main_lane_width_m = 0
            Search_M_line_H = 0
            main_lane_middle_width = 9999
            Final_Middle_Left_X = 0
            Final_Middle_Right_X = 0
            Search_Middle_line_H = 0

            ## Find the lowest X of Main lane drivable map
            for i in range(int(h)):
                find_left_tmp_x = False
                find_right_tmp_x = False
                for j in range(int(w)):

                    if int(im_dri[i][j][0]) == dri_map["MainLane"]:
                        if i>Lowest_H:
                            Lowest_H = i
                        if find_top_y==False:
                            Top_Y = i
                            find_top_y = True
                    if int(im_dri[i][j][0]) == dri_map["MainLane"] and find_left_tmp_x==False:
                        Left_tmp_X = j
                        find_left_tmp_x = True

                    if int(im_dri[i][j][0]) == dri_map["BackGround"] and find_right_tmp_x==False and find_left_tmp_x==True:
                        Right_tmp_X = j
                        find_right_tmp_x = True
                        temp_y = i
                
                # print(f"find_left_tmp_x:{find_left_tmp_x}")
                # print(f"find_right_tmp_x:{find_right_tmp_x}")
                tmp_main_lane_width = abs(Right_tmp_X - Left_tmp_X)


                '''
                Find the midium main lane drivable area width, 
                and not at the vanish point
                '''
                if tmp_main_lane_width>main_lane_width_m\
                    and find_left_tmp_x==True \
                    and find_right_tmp_x==True \
                    and abs(i-Top_Y)>h_min and abs(i-Top_Y) < (h_max):
                 
                    main_lane_width_m = tmp_main_lane_width
                    Final_Middle_Left_X = Left_tmp_X
                    Final_Middle_Right_X = Right_tmp_X
                    Search_M_line_H = i
                     
            # Middle bounding box
            if Final_Middle_Left_X==0 and Final_Middle_Right_X==0:
                Left_M_X = None
                Right_M_X = None
            else:
                Left_M_X = Final_Middle_Left_X
                Right_M_X = Final_Middle_Right_X
                # print(f"Left_M_X = {Left_M_X}")
                # print(f"Right_M_X = {Right_M_X}")
                # print(f"Search_M_line_H = {Search_M_line_H}")


                # print(f"line Y :{Search_line_H} Left_X:{Left_X}, Right_X:{Right_X} Middle_X:{Middle_X}")
            if Left_M_X is not None and Right_M_X is not None:
                # Middle bounding box
                DUA_X = int((Left_M_X + Right_M_X)/2.0)
                DUA_Y = int(Search_M_line_H + VL_Y /2.0)
                DUA_W = abs(Right_M_X - Left_M_X)
                DUA_H = abs(int(Search_M_line_H - VL_Y + 1))

                if self.show_im and return_type==1 and force_show_im:
                # if True:
                    # left X M
                    cv2.circle(im_dri_cm,(Left_M_X,Search_M_line_H), 10, (0, 255, 255), 3)
                    cv2.circle(im,(Left_M_X,Search_M_line_H), 10, (0, 255, 255), 3)
                    # right X M
                    cv2.circle(im_dri_cm,(Right_M_X,Search_M_line_H), 10, (255, 0, 255), 3)
                    cv2.circle(im,(Right_M_X,Search_M_line_H), 10, (255, 0, 255), 3)


                    # middle vertical line
                    # start_point = (Middle_X,0)
                    # end_point = (Middle_X,h)
                    # color = (255,127,0)
                    # thickness = 4
                    # cv2.line(im_dri_cm, start_point, end_point, color, thickness)
                    # cv2.line(im, start_point, end_point, color, thickness)

                    
                    if Left_M_X is not None:
                        cv2.rectangle(im_dri_cm, (Left_M_X, 0), (Right_M_X, Search_M_line_H), (0,127,127) , 3, cv2.LINE_AA)
                        cv2.rectangle(im, (Left_M_X, 0), (Right_M_X, Search_M_line_H), (0,127,127) , 3, cv2.LINE_AA)

                    cv2.imshow("drivable image",im_dri_cm)
                    cv2.imshow("image",im)
                    cv2.waitKey()
            else:
                DUA_X = None
                DUA_Y = None
                DUA_W = None
                DUA_H = None
                Search_M_line_H = None
        
        else:
            if return_type==1:
                return (None,None,None,None),None,None
            if return_type==2:
                return (None,None,None)
        
        # print(f"Middle_X:{Middle_X},Middle_Y:{Middle_Y},DCA_W:{DCA_W},DCA_H:{DCA_H}")
        if return_type==1:
            return (DUA_X,DUA_Y,DUA_W,DUA_H),h,w
        if return_type==2:
            return (Left_M_X,Right_M_X,Search_M_line_H,VL_Y),h,w

    def Get_DCA_XYWH(self,im_path,return_type=1):
        '''
        BDD100K Drivable map label :
        0: Main Lane
        1: Alter Lane
        2: BackGround
        '''
        
        drivable_path,drivable_mask_path,lane_path,detection_path = self.parse_path_ver2(im_path,type=self.data_type,detect_folder=DETECTTION_FOLDER)

        h = 0
        w = 0
        if os.path.exists(drivable_path):
            im_dri = cv2.imread(drivable_mask_path)
            h,w = im_dri.shape[0],im_dri.shape[1]
            # print(f"h:{h}, w:{w}")
        if not os.path.exists(detection_path):
            print(f"{detection_path} is not exists !! PASS~~~")
            return (None,None,None,None),None,None

        min_final,index = self.Get_Min_y_In_Drivable_Area(drivable_path)    
        min_final_2 = self.Find_Min_Y_Among_All_Vehicle_Bounding_Boxes(min_final,detection_path,h,w)

        dri_map = {"MainLane": 0, "AlterLane": 1, "BackGround":2}
        
        if os.path.exists(drivable_path):
            
            im_dri_cm = cv2.imread(drivable_path)
            im = cv2.imread(im_path)

            Lowest_H = 0
            Search_line_H = 0
            Left_tmp_X = 0
            Right_tmp_X = 0
            main_lane_width = 0
            find_left_tmp_x = False
            find_right_tmp_x = False
            Final_Left_X = 0
            Final_Right_X = 0
            ## Find the lowest X of Main lane drivable map
            for i in range(int(h)):
                find_left_tmp_x = False
                find_right_tmp_x = False
                for j in range(int(w)):

                    if int(im_dri[i][j][0]) == dri_map["MainLane"]:
                        if i>Lowest_H:
                            Lowest_H = i
                    if int(im_dri[i][j][0]) == dri_map["MainLane"] and find_left_tmp_x==False:
                        Left_tmp_X = j
                        find_left_tmp_x = True

                    if int(im_dri[i][j][0]) == dri_map["BackGround"] and find_right_tmp_x==False and find_left_tmp_x==True:
                        Right_tmp_X = j
                        find_right_tmp_x = True
                
                # print(f"find_left_tmp_x:{find_left_tmp_x}")
                # print(f"find_right_tmp_x:{find_right_tmp_x}")     
                '''
                find the largest width in main lane drivable area
                '''
                tmp_main_lane_width = abs(Right_tmp_X - Left_tmp_X)
                if tmp_main_lane_width>main_lane_width:
                    main_lane_width = tmp_main_lane_width
                    Final_Left_X = Left_tmp_X
                    Final_Right_X = Right_tmp_X
                    Search_line_H = i
                
                    

            # Search_line_H = int(Lowest_H - 80);

            Left_X = w
            update_left_x = False
            Right_X = 0
            update_right_x = False

            Left_X = Final_Left_X
            Right_X = Final_Right_X

            #for i in range(int(h*1.0/2.0),int(h),1):
            # for j in range(int(w)):
            #     if int(im_dri[Search_line_H][j][0]) == dri_map["MainLane"]:
            #         if j<Left_X:
            #             Left_X = j
            #             update_left_x=True
            
           
            # print(f"update_left_x:{update_left_x}")

            # for j in range(int(w)):
            #     if int(im_dri[Search_line_H][j][0]) == dri_map["MainLane"]:
            #         if j>Right_X:
            #             Right_X = j
            #             update_right_x=True

            Middle_X = int((Left_X + Right_X)/2.0)
            Middle_Y = int((min_final_2 + Search_line_H) / 2.0)
            DCA_W = abs(Right_X - Left_X)
            DCA_H = abs(int(min_final_2 - Search_line_H+1))
            # print(f"update_right_x:{update_right_x}")

            # print(f"line Y :{Search_line_H} Left_X:{Left_X}, Right_X:{Right_X} Middle_X:{Middle_X}")
            
        #     if self.show_im and return_type==1:
        #     # if True:
        #         start_point = (0,Search_line_H)
        #         end_point = (w,Search_line_H)
        #         color = (255,0,0)
        #         thickness = 4
        #         # search line
        #         cv2.line(im_dri_cm, start_point, end_point, color, thickness)
        #         cv2.line(im, start_point, end_point, color, thickness)
        #         # left X
        #         cv2.circle(im_dri_cm,(Left_X,Search_line_H), 10, (0, 255, 255), 3)
        #         cv2.circle(im,(Left_X,Search_line_H), 10, (0, 255, 255), 3)
        #         # right X
        #         cv2.circle(im_dri_cm,(Right_X,Search_line_H), 10, (255, 0, 255), 3)
        #         cv2.circle(im,(Right_X,Search_line_H), 10, (255, 0, 255), 3)

        #         # middle vertical line
        #         start_point = (Middle_X,0)
        #         end_point = (Middle_X,h)
        #         color = (255,127,0)
        #         thickness = 4
        #         cv2.line(im_dri_cm, start_point, end_point, color, thickness)
        #         cv2.line(im, start_point, end_point, color, thickness)

        #         # DCA Bounding Box
        #         cv2.rectangle(im_dri_cm, (Left_X, min_final_2), (Right_X, Search_line_H), (0,255,0) , 3, cv2.LINE_AA)
        #         cv2.rectangle(im, (Left_X, min_final_2), (Right_X, Search_line_H), (0,255,0) , 3, cv2.LINE_AA)
        #         cv2.imshow("drivable image",im_dri_cm)
        #         cv2.imshow("image",im)
        #         cv2.waitKey()
        if return_type == 1:
            return (Middle_X,Middle_Y,DCA_W,DCA_H),h,w
        elif return_type == 2:
            return (Left_X,Right_X,Search_line_H,min_final_2)
    

    def Get_Two_VPA_XYWH(self,im_path,return_type=2):
        '''
        func: Get VPA XYWH (Vanish Point Area)

        BDD100K Drivable map label :
        0: Main Lane
        1: Alter Lane
        2: BackGround

        Purpose : Get the bounding box that include below:
                    1. Sky
                    2. Vanish point
                    3. Drivable area of main lane
                and this bounding box information xywh for detection label (YOLO label.txt)
        input parameter : 
                    im_path : image directory path
        output :
                type 1:
                    (Middle_X,Middle_Y,DCA_W,DCA_H),h,w

                    Middle_X : bounding box center X
                    Middle_Y : bounding box center Y
                    DCA_W    : bounding box W
                    DCA_H    : bounding box H
                    h : image height
                    w : image width
                type 2:
                    (Left_X,Right_X,Search_line_H,VL_X,VL_Y),(Left_M_X,Right_M_X,Search_M_line_H)
                    Left_X (Left_M_X)               : VPA bounding box left x (M: Middle)
                    Right_X (Right_M_X)             : VPA bounding box right x (M: Middle)
                    Search_line_H (Search_M_line_H) : the y of the min width (width = Right_X - Left_X) (M: Middle)
                    VL_X          : min vehicle coordinate x
                    VL_Y          : mi coordinate y
        '''
        
        drivable_path,drivable_mask_path,lane_path,detection_path = self.parse_path_ver2(im_path,type=self.data_type,detect_folder=DETECTTION_FOLDER)

        h = 0
        w = 0
        if os.path.exists(drivable_path):
            im_dri = cv2.imread(drivable_mask_path)
            h,w = im_dri.shape[0],im_dri.shape[1]
            # print(f"h:{h}, w:{w}")
        if not os.path.exists(detection_path):
            print(f"{detection_path} is not exists !! PASS~~~")
            if return_type==1:
                return (None,None,None,None),None,None
            else:
                return None,None,None,None,None,None,None,None

        min_final,index = self.Get_Min_y_In_Drivable_Area(drivable_path)    
        VL = self.Find_Min_Y_Among_All_Vehicle_Bounding_Boxes_Ver2(min_final,detection_path,h,w)
        VL_Y,VL_X,VL_W,VL_H = VL
        # print(f"VL_Y:{VL_Y},VL_X:{VL_X},VL_W:{VL_W},VL_H:{VL_H}")
        dri_map = {"MainLane": 0, "AlterLane": 1, "BackGround":2}
        
        if os.path.exists(drivable_path):
            
            im_dri_cm = cv2.imread(drivable_path)
            im = cv2.imread(im_path)

            Lowest_H = 0
            Search_line_H = 0
            Left_tmp_X = 0
            Right_tmp_X = 0
            main_lane_width = 9999
            find_left_tmp_x = False
            find_right_tmp_x = False
            Final_Left_X = 0
            Final_Right_X = 0
            Top_Y = 0
            temp_y = 0
            find_top_y = False
            DCA_W = 0
            DCA_H = 0
            # middle parameters
            main_lane_width_m = 0
            Search_M_line_H = 0
            main_lane_middle_width = 9999
            Final_Middle_Left_X = 0
            Final_Middle_Right_X = 0
            Search_Middle_line_H = 0

            ## Find the lowest X of Main lane drivable map
            for i in range(int(h)):
                find_left_tmp_x = False
                find_right_tmp_x = False
                for j in range(int(w)):

                    if int(im_dri[i][j][0]) == dri_map["MainLane"]:
                        if i>Lowest_H:
                            Lowest_H = i
                        if find_top_y==False:
                            Top_Y = i
                            find_top_y = True
                    if int(im_dri[i][j][0]) == dri_map["MainLane"] and find_left_tmp_x==False:
                        Left_tmp_X = j
                        find_left_tmp_x = True

                    if int(im_dri[i][j][0]) == dri_map["BackGround"] and find_right_tmp_x==False and find_left_tmp_x==True:
                        Right_tmp_X = j
                        find_right_tmp_x = True
                        temp_y = i
                
                # print(f"find_left_tmp_x:{find_left_tmp_x}")
                # print(f"find_right_tmp_x:{find_right_tmp_x}")
                tmp_main_lane_width = abs(Right_tmp_X - Left_tmp_X)

                '''
                Find the minimun main lane drivable area width, 
                and not at the vanish point
                '''
                if tmp_main_lane_width<main_lane_width\
                    and find_left_tmp_x==True \
                    and find_right_tmp_x==True \
                    and tmp_main_lane_width>=50 \
                    and abs(i-Top_Y)<50 \
                    and abs(i-Top_Y)>20:
                 
                    # print(f"Top_Y:{Top_Y}")
                    # print(f"i:{i}, abs(i-Top_Y):{abs(i-Top_Y)}")
                    
                    main_lane_width = tmp_main_lane_width
                    Final_Left_X = Left_tmp_X
                    Final_Right_X = Right_tmp_X
                    Search_line_H = i

                '''
                Find the midium main lane drivable area width, 
                and not at the vanish point
                '''
                if tmp_main_lane_width>main_lane_width_m\
                    and find_left_tmp_x==True \
                    and find_right_tmp_x==True \
                    and tmp_main_lane_width>=200 \
                    and tmp_main_lane_width<=400 \
                    and abs(i-Top_Y)>50 and i < (h-150):
                 
                    main_lane_width_m = tmp_main_lane_width
                    Final_Middle_Left_X = Left_tmp_X
                    Final_Middle_Right_X = Right_tmp_X
                    Search_M_line_H = i
                     
            # Middle bounding box
            if Final_Middle_Left_X==0 and Final_Middle_Right_X==0:
                Left_M_X = None
                Right_M_X = None
            else:
                Left_M_X = Final_Middle_Left_X
                Right_M_X = Final_Middle_Right_X
                # print(f"Left_M_X = {Left_M_X}")
                # print(f"Right_M_X = {Right_M_X}")
                # print(f"Search_M_line_H = {Search_M_line_H}")

            # Top bounding box
            if Final_Left_X==0 and Final_Right_X==0:
                Left_X = None
                Right_X = None
            else:
                Left_X = Final_Left_X
                Right_X = Final_Right_X


            if Left_X is not None and Right_X is not None:
                # Top bounding box
                Middle_X = int((Left_X + Right_X)/2.0)
                Middle_Y = int((h) / 2.0)
                DCA_W = abs(Right_X - Left_X)
                DCA_H = abs(int(h-1))
                # print(f"update_right_x:{update_right_x}")

                # print(f"line Y :{Search_line_H} Left_X:{Left_X}, Right_X:{Right_X} Middle_X:{Middle_X}")
            if Left_M_X is not None and Right_M_X is not None:
                # Middle bounding box
                Middle_M_X = int((Left_M_X + Right_M_X)/2.0)
                Middle_M_Y = int(Search_M_line_H/2.0)
                VPA_M_W = abs(Right_M_X - Left_M_X)
                VPA_M_H = abs(int(Search_M_line_H))

                if self.show_im and return_type==1:
                # if True:
                    # Search_line_H = VL_Y
                    start_point = (0,Search_line_H)
                    end_point = (w,Search_line_H)
                    color = (255,0,0)
                    thickness = 4
                    # search line
                    cv2.line(im_dri_cm, start_point, end_point, color, thickness)
                    cv2.line(im, start_point, end_point, color, thickness)

                    # left X
                    cv2.circle(im_dri_cm,(Left_X,Search_line_H), 10, (0, 255, 255), 3)
                    cv2.circle(im,(Left_X,Search_line_H), 10, (0, 255, 255), 3)
                    # right X
                    cv2.circle(im_dri_cm,(Right_X,Search_line_H), 10, (255, 0, 255), 3)
                    cv2.circle(im,(Right_X,Search_line_H), 10, (255, 0, 255), 3)



                    # left X M
                    cv2.circle(im_dri_cm,(Left_M_X,Search_M_line_H), 10, (0, 255, 255), 3)
                    cv2.circle(im,(Left_M_X,Search_M_line_H), 10, (0, 255, 255), 3)
                    # right X M
                    cv2.circle(im_dri_cm,(Right_M_X,Search_M_line_H), 10, (255, 0, 255), 3)
                    cv2.circle(im,(Right_M_X,Search_M_line_H), 10, (255, 0, 255), 3)




                    # middle vertical line
                    start_point = (Middle_X,0)
                    end_point = (Middle_X,h)
                    color = (255,127,0)
                    thickness = 4
                    cv2.line(im_dri_cm, start_point, end_point, color, thickness)
                    cv2.line(im, start_point, end_point, color, thickness)

                    # VPA Bounding Box
                    cv2.rectangle(im_dri_cm, (Left_X, 0), (Right_X, h-1), (0,255,0) , 3, cv2.LINE_AA)
                    cv2.rectangle(im, (Left_X, 0), (Right_X, h-1), (0,255,0) , 3, cv2.LINE_AA)
                    
                    if Left_M_X is not None:
                        cv2.rectangle(im_dri_cm, (Left_M_X, 0), (Right_M_X, Search_M_line_H), (0,127,127) , 3, cv2.LINE_AA)
                        cv2.rectangle(im, (Left_M_X, 0), (Right_M_X, Search_M_line_H), (0,127,127) , 3, cv2.LINE_AA)

                    cv2.imshow("drivable image",im_dri_cm)
                    cv2.imshow("image",im)
                    cv2.waitKey()
            else:
                Middle_X = None
                Middle_Y = None
                DCA_W = None
                DCA_H = None
        
        # print(f"Middle_X:{Middle_X},Middle_Y:{Middle_Y},DCA_W:{DCA_W},DCA_H:{DCA_H}")
        if return_type==1:
            return (Middle_X,Middle_Y,DCA_W,DCA_H),h,w
        if return_type==2:
            return (Left_X,Right_X,Search_line_H,VL_X,VL_Y,Left_M_X,Right_M_X,Search_M_line_H)

    
    




