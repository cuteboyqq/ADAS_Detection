import glob
import os
import shutil
import cv2

class BaseDataset:

    def __init__(self,args):

        ## Multi Area Task (MA) parameters
        self.enable_vla = args.enable_vla
        self.enable_dca = args.enable_dca
        self.enable_vpa = args.enable_vpa
        self.enable_duaup = args.enable_duaup
        self.enable_duamid = args.enable_duamid
        self.enable_duadown = args.enable_duadown
        self.enable_duaupest = args.enable_duaupest
        
        ## DUA upest include sky or not
        self.include_sky = args.include_sky

        ## data directory
        self.save_dir = args.save_dir
        self.im_dir = args.im_dir
        self.dataset_dir = args.data_dir
        
        ## Pasre detection folder
        self.det_folder = args.det_folder

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
        self.dua_up_label = args.dua_uplabel
        self.dua_mid_label = args.dua_middlelabel
        self.dua_down_label = args.dua_downlabel
        self.dla_left_label = args.dla_leftlabel
        self.dla_right_label = args.dla_rightlabel
        self.dua_upest_label = args.dua_upestlabel
        self.save_img = args.save_img

        ## parse image detail
        self.data_type = args.data_type
        self.data_num = args.data_num
        self.wanted_label_list = [1,2,3,4,5,6,7]
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
        min = 0
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
            # search_h = 0
            # while(search_h<drivable_h):
            #     for j in range(drivable_w):
            #         if drivable_img[search_h][j][0]!=0:
            #             find_small_y=True
            #             min = search_h
            #             break
            #     search_h+=1
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
            # if min==0:
            #     min=None
            # print(f"min = {min}, index={index}")
            # index = 99999
            return min,index
        
        
            
    def Find_Min_Y_Among_All_Vehicle_Bounding_Boxes(self,min,detection_path,img_h,img_w,type=1):
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
        if type == 1:
            return min
        elif type==2:
            return min,min_x,min_w,min_h
    

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
                la = int(line.split(" ")[0])
                x = int(float(line.split(" ")[1])*img_w)
                y = int(float(line.split(" ")[2])*img_h)
                w = int(float(line.split(" ")[3])*img_w)
                h = int(float(line.split(" ")[4])*img_h)
                #print(f"{la} {x} {y} {w} {h}")
                if min is not None:
                    if w*h < min_rea and int(la) in self.wanted_label_list and h<min:
                        # print(f"w*h={w*h},min_rea={min_rea},x:{x},y:{y}")
                        min_rea = w*h
                        find_min_area=True
                        min=y
                        min_x=x
                        min_w=w
                        min_h=h
                    # print(f"find_min_area :{find_min_area} ")
                else:
                    if w*h < min_rea and int(la) in self.wanted_label_list:
                        # print(f"w*h={w*h},min_rea={min_rea},x:{x},y:{y}")
                        min_rea = w*h
                        find_min_area=True
                        min=y
                        min_x=x
                        min_w=w
                        min_h=h
                # if min is not None:
                #     if int(la) in self.wanted_label_list and find_min_area:
                #         # print(f"y:{y} min:{min}")
                #         min=y
                #         min_x=x
                #         min_w=w
                #         min_h=h
                # else:
                #     if int(la) in self.wanted_label_list and find_min_area:
                #         # print(f"y:{y} min:{min}")
                #         min=y
                #         min_x=x
                #         min_w=w
                #         min_h=h
        # if min is None:
        #     min = int(img_h/2.0)
        return (min,min_x,min_w,min_h)
        #return min,min_x,min_w,min_h
   
    def Add_Yolo_Txt_Label(self,xywh,detection_path,h,w,im_path,add_label=14):
        '''
            function : 
                    Add_Yolo_Txt_Label
            Purpose :
                    1. Copy the original label.txt to new save txt directory
                    2. Add new label lxywh to label.txt of YOLO foramt for all tasks (VLA,DCA,VPA,DUA,...,etc.)
        '''
        success = 0
        xywh_not_None = True
        ADAS_lxywh = None
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
                la = add_label
                # print(f"la = {la}")
                ADAS_lxywh = str(la) + " " \
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

            if ADAS_lxywh is not None:
                # Add DCA label into Yolo label.txt
                with open(save_label_path,'a') as f:
                    f.write("\n")
                    f.write(ADAS_lxywh)

            # print(f"{la}:{x}:{y}:{w}:{h}")
            success = 1
        else:
            success = 0
            print(f"detection_path:{detection_path} does not exists !! PASS~~~~~")
            return success

        return success
    


    ##============== VLA (Vanish Point Area)=================================
    def Get_Vanish_Area(self):
        return NotImplemented

    def Add_Vanish_Line_Area_Yolo_Txt_Labels(self):
        return NotImplemented

    def Add_VLA_Yolo_Txt_Label(self,min_y,detection_path,h,w,im_path):
        return NotImplemented

    def split_Image(self,im_path,drivable_min_y):
        return NotImplemented
    
    ##============ DCA (Drivable Center Area)==================================
    def Get_DCA_Yolo_Txt_Labels(self, version=1):
        return NotImplemented

    def Add_DCA_Yolo_Txt_Label(self,xywh,detection_path,h,w,im_path):
        return NotImplemented

    def Get_DCA_XYWH(self,im_path,return_type=1):
        return NotImplemented

    ## =========== VPA (Vansih Point Area)==================================
    def Get_VPA_Yolo_Txt_Labels(self, version=1):
        return NotImplemented
    

    def Add_VPA_Yolo_Txt_Label(self,xywh,detection_path,h,w,im_path):
        return NotImplemented
    
    def Get_DCA_XYWH(self,im_path,return_type=1):
        return NotImplemented

    def Get_VPA_XYWH(self,im_path,return_type=1):
        return NotImplemented
    
    ## ===========LA (Line Area)==========================================
    def Get_Two_VPA_XYWH(self,im_path,return_type=1):
        return NotImplemented
    
    ## ============VPA & LA====================================================
    def Get_Vehicle_In_Middle_Image(self,detection_path,im,Left_X,Right_X,L_X2,R_X2,Th=200):
        im_h,im_w = im.shape[0],im.shape[1]
        middle_x = int(im_w/2.0)
        middle_y = int(im_h/2.0)

        Final_Vehicle_X = None
        Final_Vehicle_Y = None
        state = None
        Vehicle_Size = 80*80
        Final_W = None

        BB_middle_X = int((Left_X + Right_X)/2)

        with open(detection_path,'r') as f:
            lines = f.readlines()
            for line in lines:
                la = int(line.split(" ")[0])
                x = int(float(line.split(" ")[1])*im_w)
                y = int(float(line.split(" ")[2])*im_h)
                w = int(float(line.split(" ")[3])*im_w)
                h = int(float(line.split(" ")[4])*im_h)
                x1 = x - int(w/2.0)
                y1 = y - int(h/2.0)
                x2 = x + int(w/2.0)
                y2 = y + int(h/2.0)
                ratio = float(w/h) if h>0 else 9999
                is_square = True
                if ratio < 0.4 or ratio > 2.5:
                    is_square=False

                if w*h > Th*Th and abs(x-middle_x)<150 and la==2 and is_square==True: # w*h > 100*100 and x>Left_X and x<Right_X and la==2:
                    Final_Vehicle_X = x
                    Final_Vehicle_Y = y
                    Final_W = Th
                    print("Vehicle state 3")
                    state = 2
                elif w*h >Vehicle_Size and x>Left_X and x<Right_X and la==2 and x>=L_X2 and x<=R_X2 and is_square==True: #w*h >Vehicle_Size and x>Left_X and x<Right_X and la==2:
                    Final_Vehicle_X = x
                    Final_Vehicle_Y = y
                    Vehicle_Size = w*h
                    state = 1
                    Final_W = w
                    print("Vehicle state 1")
                elif w*h >Vehicle_Size and abs(x-middle_x)<150 and (BB_middle_X<im_w/5.0 or BB_middle_X>im_w*4/5) and la==2 and is_square==True:
                    Final_Vehicle_X = x
                    Final_Vehicle_Y = y
                    state = 1
                    Vehicle_Size = w*h
                    Final_W = w
               
                # elif w*h > Vehicle_Size and abs(x-middle_x)<200 and abs(y-middle_y) and la==2: # w*h > 100*100 and x>Left_X and x<Right_X and la==2:
                #     Final_Vehicle_X = x
                #     Final_Vehicle_Y = y
                #     Vehicle_Size = w*h
                #     print("Vehicle state 3")
                #     state = 3
        return Final_Vehicle_X,Final_Vehicle_Y,Final_W, state
    

    def Get_VPA(self,im_path,Up,Down, use_vehicle_info=True, force_show_im = True):
        # Vanish_line = Down[3]
        carhood = Down[2]
        drivable_path,drivable_mask_path,lane_path,detection_path = self.parse_path_ver2(im_path,type=self.data_type)
        im_dri_cm = cv2.imread(drivable_path)
        im = cv2.imread(im_path)
        h,w = im.shape[0],im.shape[1]
        L_X1,L_Y1 = Up[0],Up[2]
        L_X2,L_Y2 = Down[0],Down[2]

        p1,p2 = ( L_X1,L_Y1 ), (L_X2,L_Y2)
        # print(f"(L_X1,L_Y1) = ({L_X1},{L_Y1})")
        # print(f"(L_X2,L_Y2) = ({L_X2},{L_Y2})")
        R_X1,R_Y1 = Up[1],Up[2]
        R_X2,R_Y2 = Down[1],Down[2]

        p3,p4 = (R_X1,R_Y1),(R_X2,R_Y2)
        # print(f"(R_X1,R_Y1) = ({R_X1},{R_Y1})")
        # print(f"(R_X2,R_Y2) = ({R_X2},{R_Y2})")
        VP = self.Get_VP(p1,p2,p3,p4,im)
        
        VP_X,VP_Y = VP[0],VP[1]
        
        range = 100
        Left_X = None
        Right_X = None
        if VP_X is not None:
            print(f"VP_X={VP_X}, VP_Y={VP_Y}")
            Left_X = VP_X - range if VP_X -range>0 else 0
            Right_X = VP_X + range if VP_X + range < w-1 else w-1
        
        B_W = abs(int((R_X1 - L_X1)/2.0))
        #print(f"B_W:{B_W}")
        if Left_X is not None and Right_X is not None and use_vehicle_info:
            Vehicle_X,Vehicle_Y,Final_W,state = self.Get_Vehicle_In_Middle_Image(detection_path,im,Left_X,Right_X,L_X2,R_X2,Th=200)
        # Vehicle_X = None
        # Vehicle_Y = None
        # Final_W = None
        # state = 1
        if VP_X is not None:
            if VP_X<int(w/2.0)-350 or VP_X>int(w/2.0)+350: # why no use....???
                VP_X = None
                VP_Y = None
                print("VP is None")
            elif Vehicle_X is not None:
                if state==1:
                    range = int(Final_W/2.0) + int(Final_W/4.0)
                elif state==2:
                    range = int(Final_W/2.0) + int(Final_W/4.0)
                Left_X = Vehicle_X - range if Vehicle_X -range>0 else 0
                Right_X = Vehicle_X + range if Vehicle_X + range < w-1 else w-1
                VP_X = Vehicle_X
                VP_Y = Vehicle_Y
                Search_line_H = Vehicle_Y
                print("VP using vehicle point")
            else:
                # Left_X = L_X1 if L_X1>0 else 0
                # Right_X = R_X1 if R_X1 < w-1 else w-1
                range = int(abs(R_X1-L_X1) /2.0)
                Left_X = VP_X - range if VP_X -range>0 else 0
                Right_X = VP_X + range if VP_X + range < w-1 else w-1

                Search_line_H = VP_Y
                print("VP using line intersection")
        

        if self.show_im and VP_X is not None and force_show_im:
            # if True:
            color = (255,0,0)
            thickness = 4
            # search line
            # Vanish Point
            cv2.circle(im_dri_cm,(VP_X,VP_Y), 10, (0, 255, 255), 3)
            cv2.circle(im,(VP_X,VP_Y), 10, (0, 255, 255), 3)

        
            
            # Vehicle Point
            if Vehicle_X is not None:
                cv2.circle(im_dri_cm,(Vehicle_X,Vehicle_Y), 10, (0, 255, 255), 3)
                cv2.circle(im,(Vehicle_X,Vehicle_Y), 10, (0, 255, 255), 3)

            # Left p1
            cv2.circle(im_dri_cm,p1, 10, (0, 128, 255), 3)
            cv2.circle(im,p1, 10, (0, 128, 255), 3)

            # Left p2
            cv2.circle(im_dri_cm,p2, 10, (0, 128, 255), 3)
            cv2.circle(im,p2, 10, (0, 128, 255), 3)

            # Left p3
            cv2.circle(im_dri_cm,p3, 10, (0, 128, 255), 3)
            cv2.circle(im,p3, 10, (0, 128, 255), 3)

            # Left p4
            cv2.circle(im_dri_cm, p4, 10, (0, 128, 255), 3)
            cv2.circle(im, p4, 10, (0, 128, 255), 3)

            # Left line
            start_point = (VP_X,VP_Y)
            end_point = (L_X2,L_Y2)
            color = (255,0,127)
            thickness = 3
            cv2.line(im_dri_cm, start_point, end_point, color, thickness)
            cv2.line(im, start_point, end_point, color, thickness)

            # Right line
            start_point = (VP_X,VP_Y)
            end_point = (R_X2,R_Y2)
            color = (255,127,0)
            thickness = 3
            cv2.line(im_dri_cm, start_point, end_point, color, thickness)
            cv2.line(im, start_point, end_point, color, thickness)

            # left X
            cv2.circle(im_dri_cm,(Left_X,Search_line_H), 10, (0, 255, 255), 3)
            cv2.circle(im,(Left_X,Search_line_H), 10, (0, 255, 255), 3)
            # right X
            cv2.circle(im_dri_cm,(Right_X,Search_line_H), 10, (255, 0, 255), 3)
            cv2.circle(im,(Right_X,Search_line_H), 10, (255, 0, 255), 3)

            # middle vertical line
            start_point = (VP_X,0)
            end_point = (VP_X,h-1)
            color = (255,127,0)
            thickness = 4
            cv2.line(im_dri_cm, start_point, end_point, color, thickness)
            cv2.line(im, start_point, end_point, color, thickness)

            # VPA Bounding Box
            cv2.rectangle(im_dri_cm, (Left_X, 0), (Right_X, carhood), (0,255,0) , 3, cv2.LINE_AA)
            cv2.rectangle(im, (Left_X, 0), (Right_X, carhood), (0,255,0) , 3, cv2.LINE_AA)

            # if Up[2] is not None:
            #     cv2.rectangle(im_dri_cm, (Up[0], 0), (Up[1], Up[2]), (0,127,127) , 3, cv2.LINE_AA)
            #     cv2.rectangle(im, (Up[0], 0), (Up[1], Up[2]), (0,127,127) , 3, cv2.LINE_AA)

            cv2.imshow("drivable image",im_dri_cm)
            cv2.imshow("image",im)
            cv2.waitKey()

        if carhood is not None:
            Final_Y = int(carhood / 2.0)
            Final_W = range*2
            Final_H = carhood
        else:
            Final_Y = None
            Final_W = None
            Final_H = None
        return VP_X,Final_Y,Final_W,Final_H

    def Get_VP(self,p1,p2,p3,p4,cv_im):
        '''
        Purpose :
                Get the intersection point of two line, and it is named Vanish point
        y = a*x + b
        -->
            y = L_a * x + L_b
            y = R_a * x + R_b
        
        input :
                line 1 point : p1,p2
                line 2 point : p3,p4
        output : 
                Vanish point : (VL_X,VL_Y)
        '''
        if isinstance(p1[0],int) and isinstance(p2[0],int):
            # Get left line  y = L_a * x + L_b
            if p1[0]-p2[0] != 0 and p1[0] is not None and p1[1] is not None:
                L_a = float((p1[1]-p2[1])/(p1[0]-p2[0]))
                L_b = p1[1] - (L_a * p1[0])
            else:
                # L_a = float((p1[1]-p2[1])/(1.0))
                # L_b = p1[1] - (L_a * p1[0])
                return (None,None)
        else:
            return (None,None)
        if isinstance(p3[0],int) and isinstance(p4[0],int):
            # Get right line y = R_a * x + R_b
            if p3[0]-p4[0]!=0 and p3[0] is not None and p3[1] is not None:
                R_a = float((p3[1]-p4[1])/(p3[0]-p4[0]))
                R_b = p3[1] - (R_a * p3[0])
            else:
                # R_a = float((p3[1]-p4[1])/(1.0))
                # R_b = p3[1] - (R_a * p3[0])
                return (None,None)
        else:
            return (None,None)
        # Get the Vanish Point
        if (L_a - R_a) != 0:
            VP_X = int(float((R_b - L_b)/(L_a - R_a)))
            VP_Y = int(float(L_a * VP_X) + L_b)
        else:
            return (None,None)
            # h,w = cv_im.shape[0],cv_im.shape[1]
            # VP_X = int(w/2.0)
            # VP_Y = int(float(L_a * VP_X) + L_b)
        return (VP_X,VP_Y)
        return NotImplemented


    # 2023-12-18 updated Algorithm
    def Get_VPA_XYWH_Ver2(self,im_path,return_type=1): 
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

        Algorithm :
                    1. Search main lane drivable area from bottom to top, get the min y of Left X
                    2. Search main lane drivable area from bottom to top, get the min y of right X
                    3. Center X  = (Left X + Right X)/2.0
                    4. Get bounding box :
                        X = Center X
                        Y = Image H / 2.0
                        W = Right X - Left X
                        H = Image H
        input parameter : 
                    im_path : image directory path
        output :
                    (Middle_X,Middle_Y,DCA_W,DCA_H),h,w

                    Middle_X : bounding box center X
                    Middle_Y : bounding box center Y
                    DCA_W    : bounding box W
                    DCA_H    : bounding box H
                    h : image height
                    w : image width
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

           
            Search_line_H = 0
         
            Final_Left_X = 0
            Final_Right_X = 0
         
            DCA_W = 0
            DCA_H = 0
            ## -----------------Start Get VPA Algorithm--------------------------------------
            ## 1. Search main lane drivable area from bottom to top, get the left X with min y
            Get_First_Left_X = False
            First_Left_X = 0
            Get_Second_Left_X = False
            Second_Left_X = 0
            Left_Y = 0
            get_main_lane_point = False
            temp_X = 0
            left_temp_Y = h-1
            Get_Left_Boundary_X = False
            for i in range(h-1,1,-1): #(begin,end,step)
                Get_Left_Boundary_X = False
                for j in range(0,w-1,1): #(begin,end,step)
                    if im_dri[i][j][0] == dri_map["BackGround"] \
                        and  im_dri[i][j+1][0] == dri_map["MainLane"]\
                        and Get_Left_Boundary_X==False:
                        if i<=left_temp_Y:
                            left_temp_Y = i
                            Second_Left_X = j
                            Get_Left_Boundary_X=True
                

               
                            
            
            ## 2. Search main lane drivable area from bottom to top, get the Right X with min y
            Get_First_Right_X = False
            First_Right_X = w-1
            Get_Second_Right_X = False
            Second_Right_X = 0
            Right_Y = 0
            Get_middle_X = False
            middle_X = 0
            right_temp_Y = h-1
            Get_Right_Boundary_X = False
            for i in range(h-1,1,-1): #(begin,end,step)
                Get_Right_Boundary_X=False
                for j in range(w-1,0,-1): #(begin,end,step)
                    if im_dri[i][j][0] == dri_map["BackGround"] \
                        and  im_dri[i][j-1][0] == dri_map["MainLane"]\
                        and Get_Right_Boundary_X==False:
                        if i<=right_temp_Y:
                            right_temp_Y = i
                            Second_Right_X = j-1
                            Get_Right_Boundary_X=True
                

            ## 3. Center X  = (Left X + Right X)/2.0
            Left_X = Second_Left_X
            Right_X = Second_Right_X
            print(f"Left_X:{Left_X}")
            print(f"Right_X:{Right_X}")

            ## 4. Get bounding box
        
            Search_line_H = int((left_temp_Y+right_temp_Y)/2.0)

            # Left_X = w
            # update_left_x = False
            # Right_X = 0
            # update_right_x = False

            # Left_X = int(VL_X - (VL_W * 2.0)) if VL_X - (VL_W * 2.0)>0 else 0
            # Right_X = int(VL_X + (VL_W * 2.0)) if VL_X + (VL_W * 2.0)<w-1 else w-1
            # if Final_Left_X==0 and Final_Right_X==0:
            #     Left_X = None
            #     Right_X = None
            # else:
            #     Left_X = Final_Left_X
            #     Right_X = Final_Right_X


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
            return (Left_X,Right_X,Search_line_H)
