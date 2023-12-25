import glob
import os
import shutil
import cv2
from engine.dataset import BaseDataset


class VPA(BaseDataset):

   

    def Get_VPA_Yolo_Txt_Labels(self, version=1):
        '''
            func: 
                Get Vanish Point Area
            Purpose : 
                parsing the images in given directory, 
                find the Vanish Point Area (VPA: Vanish Point Area)
                and add bounding box information x,y,w,h 
                into label.txt of yolo format.
            input :
                self.im_dir : the image directory
                self.dataset_dir : the dataset directory
                self.save_dir : save crop image directory
            output:
                the label.txt with VPA (VPA: Vanish Point Area) bounding box
        '''
        im_path_list = glob.glob(os.path.join(self.im_dir,"*.jpg"))

        if self.data_num<len(im_path_list):
            final_wanted_img_count = self.data_num
        else:
            final_wanted_img_count = len(im_path_list)

        print(f"final_wanted_img_count = {final_wanted_img_count}")
        min_final_2 = None
        
        for i in range(final_wanted_img_count):
            drivable_path,drivable_mask_path,lane_path,detection_path = self.parse_path_ver2(im_path_list[i],type=self.data_type)
            print(f"{i}:{im_path_list[i]}")
            im = cv2.imread(im_path_list[i])
            h,w = im.shape[0],im.shape[1]
            # if version==1:
            #     xywh,h,w = self.Get_DCA_XYWH(im_path_list[i],return_type=1)
            # elif version==2:
            #     xywh,h,w = self.Get_VPA_XYWH(im_path_list[i],return_type=1)
            # elif version==3:
            detection_file = detection_path.split(os.sep)[-1]
            save_txt_path = os.path.join(self.save_txtdir,detection_file)
            if os.path.exists(save_txt_path):
                print("save_txt_path exists , PASS~~~!!")
                success = 1
                continue
            Down, Up= self.Get_DCA_XYWH(im_path_list[i],return_type=2) #(Left_X,Right_X,Search_line_H,min_final_2)
            # Up = self.Get_VPA_XYWH(im_path_list[i],return_type=2) 
            #(Left_X,Right_X,Search_line_H,min_final_2),(Final_Left_X_upper,Final_Right_X_upper,Search_line_H_upper)
            if Down[0] is not None and Down[1] is not None and Up[0] is not None and Up[1] is not None \
                and isinstance(Down[0],int)\
                and isinstance(Down[1],int)\
                and isinstance(Up[0],int)\
                and isinstance(Up[1],int):
                VP_x,VP_y,New_W,New_H = self.Get_VPA(im_path_list[i],Up,Down)
            # else:
            #     print("None value, return !!")
            #     continue
            # elif version==4:
            #     xywh,h,w = self.Get_VPA_XYWH_Ver2(im_path_list[i],return_type=1)
            # else:
            #     print("No this version , only support verson=0 : DCA include main lane drivable area, \
            #           version 1 : DCA include vanish point, sky and drivable area")
            #     return NotImplemented
            # if version == 1 or version == 2:
            #     x,y = xywh[0],xywh[1]
            # elif version == 3:
                x, y  = VP_x, VP_y
                xywh = (VP_x,VP_y,New_W,New_H)
            else:
                xywh = (None,None,None,None)
            
            success = self.Add_VPA_Yolo_Txt_Label(xywh,detection_path,h,w,im_path_list[i])

    

    def Add_VPA_Yolo_Txt_Label(self,xywh,detection_path,h,w,im_path):
        success = 0
        xywh_not_None = True
        DCA_lxywh = None
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
                la = self.vpa_label
                # print(f"la = {la}")
                DCA_lxywh = str(la) + " " \
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

            if DCA_lxywh is not None:
                # Add DCA label into Yolo label.txt
                with open(save_label_path,'a') as f:
                    f.write("\n")
                    f.write(DCA_lxywh)

            # print(f"{la}:{x}:{y}:{w}:{h}")
            success = 1
        else:
            success = 0
            print(f"detection_path:{detection_path} does not exists !! PASS~~~~~")
            return success

        return success

    def Get_DCA_XYWH(self,im_path,return_type=1):
        '''
        BDD100K Drivable map label :
        0: Main Lane
        1: Alter Lane
        2: BackGround
        '''
        
        drivable_path,drivable_mask_path,lane_path,detection_path = self.parse_path_ver2(im_path,type=self.data_type)

        h = 0
        w = 0
        if os.path.exists(drivable_path):
            im_dri = cv2.imread(drivable_mask_path)
            h,w = im_dri.shape[0],im_dri.shape[1]
            # print(f"h:{h}, w:{w}")
        if not os.path.exists(detection_path):
            print(f"{detection_path} is not exists !! PASS~~~")
            if return_type == 1:
                return (None,None,None,None),None,None
            elif return_type == 2:
                return (None,None,None,None),(None,None,None,None,None)

        min_final,index = self.Get_Min_y_In_Drivable_Area(drivable_path)    
        VL = self.Find_Min_Y_Among_All_Vehicle_Bounding_Boxes(min_final,detection_path,h,w,type=2) #min_final_2
        VL_Y,VL_X,VL_W,VL_H = VL
        min_final_2 = VL_Y
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
            Final_Left_X = None
            Final_Right_X = None
            Final_Left_X_upper = None
            Final_Right_X_upper = None
            Search_line_H_upper = None
            main_lane_width_upper = 9999
            find_top_y = False
            Top_Y = 0
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
                
                # print(f"find_left_tmp_x:{find_left_tmp_x}")
                # print(f"find_right_tmp_x:{find_right_tmp_x}")
                tmp_main_lane_width = abs(Right_tmp_X - Left_tmp_X)
                if tmp_main_lane_width>main_lane_width:
                    main_lane_width = tmp_main_lane_width
                    Final_Left_X = Left_tmp_X
                    Final_Right_X = Right_tmp_X
                    Search_line_H = i
                

                ## Find the Min Main Lane Width
                if tmp_main_lane_width<main_lane_width_upper\
                    and find_left_tmp_x==True \
                    and find_right_tmp_x==True \
                    and tmp_main_lane_width>=50 \
                    and abs(i-Top_Y)<50 \
                    and abs(i-Top_Y)>20:
                 
                    # print(f"Top_Y:{Top_Y}")
                    # print(f"i:{i}, abs(i-Top_Y):{abs(i-Top_Y)}")
                    
                    main_lane_width_upper = tmp_main_lane_width
                    Final_Left_X_upper = Left_tmp_X
                    Final_Right_X_upper = Right_tmp_X
                    Search_line_H_upper = i

                    

            # Search_line_H = int(Lowest_H - 80);

            # Left_X = w
            # update_left_x = False
            # Right_X = 0
            update_right_x = False
            if Final_Left_X is not None and Final_Right_X is not None:
                Left_X = Final_Left_X
                Right_X = Final_Right_X
            else:
                Left_X = None
                Right_X = None
            
            if Final_Left_X_upper is not None and Final_Right_X_upper is not None:
                Left_X_U = Final_Left_X_upper
                Right_X_U = Final_Right_X_upper
            else:
                Left_X_U = None
                Right_X_U = None

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
            if Left_X is not None:
                Middle_X = int((Left_X + Right_X)/2.0)
                Middle_Y = int((min_final_2 + Search_line_H) / 2.0)
                DCA_W = abs(Right_X - Left_X)
                DCA_H = abs(int(min_final_2 - Search_line_H+1))
            else:
                if return_type == 1:
                    return (None,None,None,None),None,None
                elif return_type == 2:
                    return (None,None,None,None),(None,None,None,None,None)
            # print(f"update_right_x:{update_right_x}")

            # print(f"line Y :{Search_line_H} Left_X:{Left_X}, Right_X:{Right_X} Middle_X:{Middle_X}")
            


            if self.show_im and return_type==1:
            # if True:
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

                # middle vertical line
                start_point = (Middle_X,0)
                end_point = (Middle_X,h)
                color = (255,127,0)
                thickness = 4
                cv2.line(im_dri_cm, start_point, end_point, color, thickness)
                cv2.line(im, start_point, end_point, color, thickness)

                # DCA Bounding Box
                cv2.rectangle(im_dri_cm, (Left_X, min_final_2), (Right_X, Search_line_H), (0,255,0) , 3, cv2.LINE_AA)
                cv2.rectangle(im, (Left_X, min_final_2), (Right_X, Search_line_H), (0,255,0) , 3, cv2.LINE_AA)
                cv2.imshow("drivable image",im_dri_cm)
                cv2.imshow("image",im)
                cv2.waitKey()
        else:
            return (None,None,None,None),(None,None,None,None,None) 
        if return_type == 1:
            return (Middle_X,Middle_Y,DCA_W,DCA_H),h,w
        elif return_type == 2:
            return (Left_X,Right_X,Search_line_H,min_final_2),(Final_Left_X_upper,Final_Right_X_upper,Search_line_H_upper,VL_X,VL_Y)
    

    def Get_VPA_XYWH(self,im_path,return_type=1):
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
                    (Left_X,Right_X,Search_line_H,VL_X,VL_Y)
                    Left_X        : VPA bounding box left x
                    Right_X       : VPA bounding box right x
                    Search_line_H : the y of the min width (width = Right_X - Left_X) 
                    VL_X          : min vehicle coordinate x
                    VL_Y          : mi coordinate y
        '''
        
        drivable_path,drivable_mask_path,lane_path,detection_path = self.parse_path_ver2(im_path,type=self.data_type)

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

                ## Find the Min Main Lane Width
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
                     

            # Search_line_H = int(Lowest_H - 80);

            # Left_X = w
            # update_left_x = False
            # Right_X = 0
            # update_right_x = False

            # Left_X = int(VL_X - (VL_W * 2.0)) if VL_X - (VL_W * 2.0)>0 else 0
            # Right_X = int(VL_X + (VL_W * 2.0)) if VL_X + (VL_W * 2.0)<w-1 else w-1
            if Final_Left_X==0 and Final_Right_X==0:
                Left_X = None
                Right_X = None
            else:
                Left_X = Final_Left_X
                Right_X = Final_Right_X


            if Left_X is not None and Right_X is not None:
                Middle_X = int((Left_X + Right_X)/2.0)
                Middle_Y = int((h) / 2.0)
                DCA_W = abs(Right_X - Left_X)
                DCA_H = abs(int(h-1))
                # print(f"update_right_x:{update_right_x}")

                # print(f"line Y :{Search_line_H} Left_X:{Left_X}, Right_X:{Right_X} Middle_X:{Middle_X}")
            
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

                    # middle vertical line
                    start_point = (Middle_X,0)
                    end_point = (Middle_X,h)
                    color = (255,127,0)
                    thickness = 4
                    cv2.line(im_dri_cm, start_point, end_point, color, thickness)
                    cv2.line(im, start_point, end_point, color, thickness)

                    # DCA Bounding Box
                    cv2.rectangle(im_dri_cm, (Left_X, 0), (Right_X, h-1), (0,255,0) , 3, cv2.LINE_AA)
                    cv2.rectangle(im, (Left_X, 0), (Right_X, h-1), (0,255,0) , 3, cv2.LINE_AA)
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
        elif return_type==2:
            return (Left_X,Right_X,Search_line_H,VL_X,VL_Y)

    
        
    





