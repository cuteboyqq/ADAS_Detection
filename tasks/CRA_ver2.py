import glob
import os
import shutil
import cv2
from engine.dataset import BaseDataset
import numpy as np

class CRA_Ver2(BaseDataset):

    def Get_CRA_Yolo_Txt_Labels(self, version=1):
        '''
            func: 
                Get Cross Road Area
            Purpose : 
                parsing the images in given directory, 
                find the Cross road 
                and add bounding box information x,y,w,h 
                into label.txt of yolo format.
            input :
                self.im_dir : the image directory
                self.dataset_dir : the dataset directory
                self.save_dir : save crop image directory
            output:
                the label.txt with Cross Road Area (CRA) bounding box
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
         
            # x,y = xywh[0],xywh[1]
         
            # success = self.Add_CRA_Yolo_Txt_Label(xywh,detection_path,im_h,im_w,im_path_list[i])

    

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
     
        drivable_path,drivable_mask_path,lane_path,detection_path = self.parse_path_ver2(im_path,type=self.data_type,detect_folder=self.det_folder)
        h = 0
        w = 0
        if os.path.exists(im_path):
            im = cv2.imread(im_path)
            self.im = im
            self.im_h,self.im_w = im.shape[0],im.shape[1]
        else:
            print(f"{im_path} is not exists !! PASS~~~")
            return self.CRA_xywh_list,None,None
        #     # print(f"h:{h}, w:{w}")
        if not os.path.exists(detection_path):
            print(f"{detection_path} is not exists !! PASS~~~")
            return self.CRA_xywh_list,None,None

        xywh = (None,None,None,None)
        if os.path.exists(lane_path):
            im_lane = cv2.imread(lane_path)
            self.im_lane = im_lane
            self.im_lane_copy = np.zeros_like(im_lane)
            # Get CRA label map 
            self.get_particular_label_map(R=86,G=94,B=219) 
            # Find contour in CRA map
            self.get_contour(self.im_lane_copy)  
            print("Number of Contours found = " + str(len(self.contours)))
            # Get contour x1y1wh
            self.get_CRA_contour_x1y1wh_list()  
            if self.show_im and len(self.contours)>0:
                # Show contour Bounding Box image
                self.show_CRA_contour_images()
        else:
            im_h,im_w = None,None
        
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

        
        return self.CRA_xywh_list,self.im_h,self.im_w

    

    def get_CRA_contour_x1y1wh_list(self):
        '''
        Purpose : Get CRA contour bounding box x1,y1,w,h
                    aand save in the list
        '''
        #   Get CRA XYWH and draw bounding box
        for i in range(len(self.contours)):
            x1,y1,w,h = cv2.boundingRect(self.contours[i])
            contour_area = w*h
            
            if contour_area > self.contour_area_th:
                wh_ratio = float(w/h)
                if wh_ratio > 10:
                    self.CRA_horizontal_list.append([x1,y1,w,h])
                    self.horizontal_contour.append(self.contours[i])
                    color = (255,255,0)
                elif wh_ratio < 5:
                    self.CRA_vertical_list.append([x1,y1,w,h])
                    self.vertical_contour.append(self.contours[i])
                    color = (0,255,255)
                else:
                    self.CRA_unknown_list.append([x1,y1,w,h])
                    self.unknown_contour.append(self.contours[i])
                    color = (255,0,255)

                
                im = cv2.rectangle(self.im,(x1,y1),(x1+w,y1+h),color,1)
                im_lane = cv2.rectangle(self.im_lane,(x1,y1),(x1+w,y1+h),color,1)

                # YOLO format xywh
                x = x1 + int(w/2.0)
                y = y1 + int(h/2.0)
                self.CRA_xywh_list.append([x,y,w,h])
            
    
    def show_CRA_contour_images(self):
        '''
        Purpose : Show CRA contour image 
        '''
        # cv2.drawContours(im_lane, contours, -1, (255, 255, 0), 1)
        # cv2.drawContours(im, contours, -1, (255, 255, 0), 1)  
        if self.resize:
            im_lane = cv2.resize(self.im_lane, (640, 360), interpolation=cv2.INTER_AREA)
            im = cv2.resize(self.im, (640, 360), interpolation=cv2.INTER_AREA)
            edge = cv2.resize(self.edge, (640, 360), interpolation=cv2.INTER_AREA)
        else:
            im_lane = self.im_lane
            im = self.im
            edge = self.edge
        cv2.imshow('Contours_im_lane', im_lane)
        cv2.imshow('Contours', im)
        # cv2.imshow('Canny Edges After Contouring', edged) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 