import glob
import os
import shutil
import cv2
from engine.dataset import BaseDataset

DETECTTION_FOLDER = "detection-DCA-VPA_ver3"

class MultiAreaTask(BaseDataset):

    def Get_Multi_Area_Tasks_Yolo_Txt_Labels(self):
        '''
            func: 
                Get Multiple ADAS Area (VLA,DCA,VPA,...,etc.) label.txt of yolo format

                
                PS: use enable/disable (True/False) to get tha label you want, 
                    for example : self.enable_vla = True is enable to generate the VLA label
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
                the label.txt with Multiple ADAS Area (VLA,DCA,VPA,...,etc.) bounding box
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
            xywh = (None,None,None,None)
            xywh_m = (None,None,None,None)
          
            detection_file = detection_path.split(os.sep)[-1]
            save_txt_path = os.path.join(self.save_txtdir,detection_file)
            if os.path.exists(save_txt_path):
                print("save_txt_path exists , PASS~~~!!")
                success = 1
                continue
                      
            VLA_xywh,DCA_xywh,DUA_xywh_upest,DUA_xywh_up,DUA_xywh_middle,DUA_xywh_down,Up,Down,im_h,im_w = self.Get_Multi_Area(im_path_list[i],return_types=1) 


            if Down[0] is not None and Down[1] is not None and Up[0] is not None and Up[1] is not None \
                and isinstance(Down[0],int)\
                and isinstance(Down[1],int)\
                and isinstance(Up[0],int)\
                and isinstance(Up[1],int):
                VP_x,VP_y,New_W,New_H = self.Get_VPA(im_path_list[i],Up,Down) # Up:(1,2,3,4,5) Down:(1,2,3,4) 
          
                x, y  = VP_x, VP_y
                VPA_xywh = (VP_x,VP_y,New_W,New_H)
            else:
                VPA_xywh = (None,None,None,None)

            success = self.Add_Multi_Area_Yolo_Txt_Label(VLA_xywh,DCA_xywh,VPA_xywh,DUA_xywh_upest,DUA_xywh_up,DUA_xywh_middle,DUA_xywh_down,detection_path,h,w,im_path_list[i])
    




    def Get_Multi_Area(self,im_path,return_types=1):
        '''
        func:  Get Multiple ADAS Area (VLA,DCA,VPA,...,etc.) XYWH

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
            drivable_path,drivable_mask_path,lane_path,detection_path = self.parse_path_ver2(im_path,type=self.data_type,detect_folder=self.det_folder)
            im_h = 0
            im_w = 0
            if os.path.exists(drivable_path):
                im_dri = cv2.imread(drivable_mask_path)
                im_h,im_w = im_dri.shape[0],im_dri.shape[1]
                # print(f"h:{h}, w:{w}")
            if not os.path.exists(detection_path):
                print(f"{detection_path} is not exists !! PASS~~~")
                if return_types==1:
                    return (None,None,None,None),(None,None,None,None),(None,None,None,None),(None,None,None,None),(None,None,None,None),\
                        (None,None,None),(None,None,None,None,None),None,None
                else:
                    return (None,None,None)
        else:
            if return_types==1:
                return (None,None,None,None),(None,None,None,None),(None,None,None,None),(None,None,None,None),(None,None,None,None),\
                        (None,None,None),(None,None,None,None,None),None,None
            else:
                return (None,None,None)
            
        if os.path.exists(drivable_path) and os.path.exists(im_path):
            im_dri_cm = cv2.imread(drivable_path)
            im = cv2.imread(im_path)
            # DUA_down,h,w = self.Get_DUA_XYWH(im_path,return_type = 2, w_min=200, w_max=500, h_min=80, h_max=140,force_show_im=False)
            # #(Left_M_X,Right_M_X,Search_M_line_H,VL_Y),h,w
            # DUA_middle,h,w = self.Get_DUA_XYWH(im_path,return_type = 2, w_min=50, w_max=200, h_min=40, h_max=80,force_show_im=False)
            # DUA_up,h,w = self.Get_DUA_XYWH(im_path,return_type = 2, w_min=50, w_max=200, h_min=20, h_max=40,force_show_im=False)
            VLA_xywh,DCA_xywh,DUA_upest,DUA_up,DUA_middle,DUA_down,Up,Down,h,w = self.Get_Multi_Area_XYWH(im_path,
                                                                                                            return_type=2,
                                                                                                            h_upperest=(8,15),
                                                                                                            h_upper=(20,40),
                                                                                                            h_middel=(40,80),
                                                                                                            h_down=(80,140),
                                                                                                            force_show_im=False)
            if DUA_down[0] is not None and DUA_down[3] is not None:
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

            if DUA_middle[0] is not None and DUA_middle[3] is not None:
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


            if DUA_up[0] is not None  and DUA_up[3] is not None:
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

            if DUA_upest[0] is not None  and DUA_upest[3] is not None:
                left_x = DUA_upest[0]
                right_x = DUA_upest[1]
                
                y_down = DUA_upest[2]
                VL_Y = DUA_upest[3]
                x = int((left_x + right_x) / 2.0)
                y = int(VL_Y)
                w = int(right_x - left_x)
                h = int(abs(int(y_down - VL_Y))*2.0)
                DUA_xywh_upest = (x,y,w,h)
            else:
                DUA_xywh_upest = (None,None,None,None)

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

                if DUA_upest[0] is not None:
                    left_x = DUA_upest[0]
                    right_x = DUA_upest[1]
                    
                    y_down = DUA_upest[2]
                    VL_Y = DUA_upest[3]
                    h = int(int(abs(y_down-VL_Y))*2.0)
                    p1 =(left_x,y_down)
                    p2 = (right_x,VL_Y-int(h/2.0))
                    cv2.rectangle(im_dri_cm, p1, p2, (0,120,255) , 3, cv2.LINE_AA)
                    cv2.rectangle(im, p1, p2, (0,120,255) , 3, cv2.LINE_AA)

                cv2.imshow("drivable image",im_dri_cm)
                cv2.imshow("image",im)
                cv2.waitKey()
        else:
            if return_types==1:
                return (None,None,None,None),(None,None,None,None),(None,None,None,None),(None,None,None,None),(None,None,None,None),\
                        (None,None,None),(None,None,None,None,None),None,None
            else:
                return (None,None,None)
            

        return VLA_xywh,DCA_xywh,DUA_xywh_upest,DUA_xywh_up,DUA_xywh_middle,DUA_xywh_down,Up,Down,im_h,im_w
        return NotImplemented



    
    def Add_Multi_Area_Yolo_Txt_Label(self,VLA_xywh,
                                        DCA_xywh,
                                        VPA_xywh,
                                        DUA_xywh_upest,
                                        DUA_xywh_up,
                                        DUA_xywh_middle,
                                        DUA_xywh_down,
                                        detection_path,h,w,im_path):
        success = 0
        im_w = w
        im_h = h
        # print(f"im_w={im_w},im_h ={im_h}")
        VLA_lxywh = None
        DCA_lxywh = None
        VPA_lxywh = None
        DUA_lxywh_upest = None
        DUA_lxywh_up = None
        DUA_lxywh_middle = None
        DUA_lxywh_down = None

        xywh_vla_not_None = True
        if VLA_xywh[0] is not None and VLA_xywh[1] is not None:
            xywh_vla_not_None = True
        else:
            xywh_vla_not_None = False

        xywh_dca_not_None = True
        if DCA_xywh[0] is not None and DCA_xywh[1] is not None:
            xywh_dca_not_None = True
        else:
            xywh_dca_not_None = False

        
        xywh_vpa_not_None = True
        if VPA_xywh[0] is not None and VPA_xywh[1] is not None:
            xywh_vpa_not_None = True
        else:
            xywh_vpa_not_None = False

        xywh_upest_not_None = True
        if DUA_xywh_upest[0] is not None and DUA_xywh_upest[1] is not None:
            xywh_upest_not_None = True
        else:
            xywh_upest_not_None = False
        
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


            if xywh_vla_not_None==True:
                # middle VPA bounding box
                # print(f"xywh_m[0] = {xywh_m[0]}, xywh_m[1]={xywh_m[1]}, xywh_m[2]={xywh_m[2]}. xywh_m[3]={xywh_m[3]}")
                # print(f"w={w}, h={h}")
                x_vla = float((int(float(VLA_xywh[0]/im_w)*1000000))/1000000)
                y_vla = float((int(float(VLA_xywh[1]/im_h)*1000000))/1000000)
                w_vla = float((int(float(VLA_xywh[2]/im_w)*1000000))/1000000)
                h_vla = float((int(float(VLA_xywh[3]/im_h)*1000000))/1000000)
                la_vla = self.vla_label
                # print(f"la = {la}")
                VLA_lxywh = str(la_vla) + " " \
                            + str(x_vla) + " " \
                            + str(y_vla) + " " \
                            + str(w_vla) + " " \
                            + str(h_vla)
                
            if xywh_dca_not_None==True:
                # middle VPA bounding box
                # print(f"xywh_m[0] = {xywh_m[0]}, xywh_m[1]={xywh_m[1]}, xywh_m[2]={xywh_m[2]}. xywh_m[3]={xywh_m[3]}")
                # print(f"w={w}, h={h}")
                x_dca = float((int(float(DCA_xywh[0]/im_w)*1000000))/1000000)
                y_dca = float((int(float(DCA_xywh[1]/im_h)*1000000))/1000000)
                w_dca = float((int(float(DCA_xywh[2]/im_w)*1000000))/1000000)
                h_dca = float((int(float(DCA_xywh[3]/im_h)*1000000))/1000000)
                la_dca = self.dca_label
                # print(f"la = {la}")
                DCA_lxywh = str(la_dca) + " " \
                            + str(x_dca) + " " \
                            + str(y_dca) + " " \
                            + str(w_dca) + " " \
                            + str(h_dca)
            
            if xywh_vpa_not_None==True:
                # middle VPA bounding box
                # print(f"xywh_m[0] = {xywh_m[0]}, xywh_m[1]={xywh_m[1]}, xywh_m[2]={xywh_m[2]}. xywh_m[3]={xywh_m[3]}")
                # print(f"w={w}, h={h}")
                x_vpa = float((int(float(VPA_xywh[0]/im_w)*1000000))/1000000)
                y_vpa = float((int(float(VPA_xywh[1]/im_h)*1000000))/1000000)
                w_vpa = float((int(float(VPA_xywh[2]/im_w)*1000000))/1000000)
                h_vpa = float((int(float(VPA_xywh[3]/im_h)*1000000))/1000000)
                la_vpa = self.vpa_label
                # print(f"la = {la}")
                VPA_lxywh = str(la_vpa) + " " \
                            + str(x_vpa) + " " \
                            + str(y_vpa) + " " \
                            + str(w_vpa) + " " \
                            + str(h_vpa)
            
            if xywh_upest_not_None==True:
                # middle VPA bounding box
                # print(f"xywh_m[0] = {xywh_m[0]}, xywh_m[1]={xywh_m[1]}, xywh_m[2]={xywh_m[2]}. xywh_m[3]={xywh_m[3]}")
                # print(f"w={w}, h={h}")
                x_upest = float((int(float(DUA_xywh_upest[0]/im_w)*1000000))/1000000)
                y_upest = float((int(float(DUA_xywh_upest[1]/im_h)*1000000))/1000000)
                w_upest = float((int(float(DUA_xywh_upest[2]/im_w)*1000000))/1000000)
                h_upest = float((int(float(DUA_xywh_upest[3]/im_h)*1000000))/1000000)
                la_upest = self.dua_upest_label
                # print(f"la = {la}")
                DUA_lxywh_upest = str(la_upest) + " " \
                            + str(x_upest) + " " \
                            + str(y_upest) + " " \
                            + str(w_upest) + " " \
                            + str(h_upest)

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

            if VLA_lxywh is not None and self.enable_vla is True:
                with open(save_label_path,'a') as f:
                    f.write(VLA_lxywh)
                    f.write("\n")

            if DCA_lxywh is not None and self.enable_dca is True:
                with open(save_label_path,'a') as f:
                    f.write(DCA_lxywh)
                    f.write("\n")
            
            if VPA_lxywh is not None and self.enable_vpa is True:
                with open(save_label_path,'a') as f:
                    f.write(VPA_lxywh)
                    f.write("\n")

            if DUA_lxywh_upest is not None and self.enable_duaupest is True:
                # Add VPA Middle label into Yolo label.txt
                with open(save_label_path,'a') as f:
                    f.write(DUA_lxywh_upest)
                    f.write("\n")

            if DUA_lxywh_up is not None and self.enable_duaup is True:
                # Add VPA Middle label into Yolo label.txt
                with open(save_label_path,'a') as f:
                    f.write(DUA_lxywh_up)
                    f.write("\n")

            if DUA_lxywh_middle is not None and self.enable_duamid is True:
                with open(save_label_path,'a') as f:
                    f.write(DUA_lxywh_middle)
                    f.write("\n")

            if DUA_lxywh_down is not None and self.enable_duadown is True:
                with open(save_label_path,'a') as f:
                    f.write(DUA_lxywh_down)
                    f.write("\n")
            
            

         
            success = 1
        else:
            success = 0
            print(f"detection_path:{detection_path} does not exists !! PASS~~~~~")
            return success

        return success

    def Get_Multi_Area_XYWH(self,
                            im_path,
                            return_type=1,
                            h_upperest=(0,20),
                            h_upper=(20,40), 
                            h_middel=(40,80), 
                            h_down=(80,140),
                            force_show_im=True):
        '''
        func: Get Multi Area (DCA,VLA,...,etc) XYWH 

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
        drivable_path,drivable_mask_path,lane_path,detection_path = self.parse_path_ver2(im_path,type=self.data_type,detect_folder=self.det_folder)
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

        if VL_Y is not None:
            VLA_X = int(w/2.0)
            VLA_Y = int(VL_Y)
            VLA_W = int(w)
            VLA_H = int(self.split_height)
        else:
            VLA_X = None
            VLA_Y = None
            VLA_W = None
            VLA_H = None

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
            main_lane_width_upest = 0
            main_lane_upperest_width = 9999
            Final_Upest_Left_X = 0
            Final_Upest_Right_X = 0
            Search_Upest_line_H = 0

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

            # initialize DCA parameters
            main_lane_width_DCA = 0
            Final_DCA_Left_X = 0
            Final_DCA_Right_X = 0
            Search_DCA_line_H = 0

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
                Find the upperest main lane drivable area width, 
                and not at the vanish point
                '''
                if tmp_main_lane_width>main_lane_width_up\
                    and find_left_tmp_x==True \
                    and find_right_tmp_x==True \
                    and abs(i-Top_Y)>h_upperest[0] and abs(i-Top_Y) < h_upperest[1]:
                    
                    main_lane_width_upest = tmp_main_lane_width
                    Final_Upest_Left_X = Left_tmp_X
                    Final_Upest_Right_X = Right_tmp_X
                    Search_Upest_line_H = i



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
                Find the Drivable Center Area width, 
                and not at the vanish point
                '''
                if tmp_main_lane_width>main_lane_width_DCA\
                    and find_left_tmp_x==True \
                    and find_right_tmp_x==True:
                    
                    main_lane_width_DCA = tmp_main_lane_width
                    Final_DCA_Left_X = Left_tmp_X
                    Final_DCA_Right_X = Right_tmp_X
                    Search_DCA_line_H = i

            '''
            Get Upperest bounding box "
            1. left x
            2. right x
            3. lower bound y 
            '''
            if Final_Upest_Left_X==0 and Final_Upest_Right_X==0:
                Left_Upest_X = None
                Right_Upest_X = None
            else:
                Left_Upest_X = Final_Upest_Left_X
                Right_Upest_X = Final_Upest_Right_X


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


            '''
            Get DCA bounding box
            1. left x
            2. right x
            3. lower bound y 
            '''
            if Final_DCA_Left_X==0 and Final_DCA_Right_X==0:
                Left_DCA_X = None
                Right_DCA_X = None
            else:
                Left_DCA_X = Final_DCA_Left_X
                Right_DCA_X = Final_DCA_Right_X

                # print(f"line Y :{Search_line_H} Left_X:{Left_X}, Right_X:{Right_X} Middle_X:{Middle_X}")

            '''
            Get Upperest DUA xywh
            '''
            if Left_Upest_X is not None and Right_Upest_X is not None \
                and VL_Y is not None:
                # Middle bounding box
                DUA_Upest_X = int((Left_Upest_X + Right_Upest_X)/2.0)
                DUA_Upest_Y = int(Search_Upest_line_H)
                DUA_Upest_W = abs(Right_Upest_X - Left_Upest_X)
                DUA_Upest_H = abs(int(Search_Upest_line_H - VL_Y))*2.0
            else:
                DUA_Upest_X = None
                DUA_Upest_Y = None
                DUA_Upest_W = None
                DUA_Upest_H = None
                Search_Upest_line_H = None

            '''
            Get Upper DUA xywh
            '''
            if Left_Up_X is not None and Right_Up_X is not None \
                and VL_Y is not None:
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
            if Left_Mid_X is not None and Right_Mid_X is not None \
                and VL_Y is not None:
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
            if Left_Down_X is not None and Right_Down_X is not None \
                and VL_Y is not None:
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

            '''
            Get DCA xywh
            '''
            if Left_DCA_X is not None and Right_DCA_X is not None \
                and VL_Y is not None:
                # Middle bounding box
                DCA_X = int((Left_DCA_X + Right_DCA_X)/2.0)
                DCA_Y = int((Search_DCA_line_H + VL_Y) /2.0)
                DCA_W = abs(Right_DCA_X - Left_DCA_X)
                DCA_H = abs(int(Search_DCA_line_H - VL_Y + 1))
            else:
                DCA_X = None
                DCA_Y = None
                DCA_W = None
                DCA_H = None
                Search_DCA_line_H = None
            
            
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
                return (None,None,None,None),(None,None,None,None),(None,None,None,None),(None,None,None,None),None,None
        
        # print(f"Middle_X:{Middle_X},Middle_Y:{Middle_Y},DCA_W:{DCA_W},DCA_H:{DCA_H}")
        if return_type==1:
            return (DUA_Up_X,DUA_Up_Y,DUA_Up_W,DUA_Up_H),(DUA_Mid_X,DUA_Mid_Y,DUA_Mid_W,DUA_Mid_H),(DUA_Down_X,DUA_Down_Y,DUA_Down_W,DUA_Down_H),h,w
        if return_type==2:
            DUA_upest = (Left_Upest_X,Right_Upest_X,Search_Upest_line_H,VL_Y)
            DUA_up = (Left_Up_X,Right_Up_X,Search_Up_line_H,VL_Y)
            DUA_mid = (Left_Mid_X,Right_Mid_X,Search_Mid_line_H,VL_Y)
            DUA_down = (Left_Down_X,Right_Down_X,Search_Down_line_H,VL_Y)
            VLA_xywh = (VLA_X,VLA_Y,VLA_W,VLA_H)
            DCA_xywh = (DCA_X,DCA_Y,DCA_W,DCA_H)
            Up = (Left_Up_X,Right_Up_X,Search_Up_line_H)
            # Up = (Left_Mid_X,Right_Mid_X,Search_Mid_line_H)
            Down = (Left_DCA_X,Right_DCA_X,Search_DCA_line_H,VL_X,VL_Y)
            # Down = (Left_Mid_X,Right_Mid_X,Search_Mid_line_H,VL_X,VL_Y)
            # Down = (Left_Down_X,Right_Down_X,Search_DCA_line_H,VL_X,VL_Y)
            # Using DUA up and DUA mid to get vanish point 2023-12-30
            return VLA_xywh,DCA_xywh,DUA_upest,DUA_up,DUA_mid,DUA_down,Up,Down,h,w