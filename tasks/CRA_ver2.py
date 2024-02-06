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
            self.init()
            self.carhoody = self.Get_carhood_y(im_path_list[i])
            drivable_path,drivable_mask_path,lane_path,detection_path = self.parse_path_ver2(im_path_list[i],type=self.data_type,detect_folder=self.det_folder)
            print(f"{i}:{im_path_list[i]}")
            im = cv2.imread(im_path_list[i])
            h,w = im.shape[0],im.shape[1]  
            x1y1wh_list,im_h,im_w = self.Get_CRA_XYWH(im_path_list[i],return_type=1)
            # x,y = xywh[0],xywh[1]
            xywh_list = self.x1y1wh2xywh(x1y1wh_list)

            success = self.Add_CRA_Yolo_Txt_Label(xywh_list,detection_path,im_h,im_w,im_path_list[i])










    def x1y1wh2xywh(self, x1y1wh_list):
        xywh_list = []
        for i in range(len(x1y1wh_list)):
            x1 = x1y1wh_list[i][0]
            y1 = x1y1wh_list[i][1]
            w  = x1y1wh_list[i][2]
            h  = x1y1wh_list[i][3]
            x = x1 + int(w/2.0)
            y = y1 + int(h/2.0)
            xywh_list.append((x,y,w,h))
        
        return xywh_list


    def Add_CRA_Yolo_Txt_Label(self,xywh_list,detection_path,im_h,im_w,im_path):
        success = 0
        xywh_list_not_None = True
        CRA_lxywh_list = []
        if len(xywh_list)>0:
            xywh_list_not_None = True
        else:
            xywh_list_not_None = False
        # print(f"xywh[0]:{xywh[0]},xywh[1]:{xywh[1]},xywh[2]:{xywh[2]},xywh[3]:{xywh[3]},w:{w},h:{h}")
        if os.path.exists(detection_path):
            
            
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

            if xywh_list_not_None == True:
                for i in range(len(xywh_list)):
                    x = float((int(float(xywh_list[i][0]/im_w)*1000000))/1000000)
                    y = float((int(float(xywh_list[i][1]/im_h)*1000000))/1000000)
                    w = float((int(float(xywh_list[i][2]/im_w)*1000000))/1000000)
                    h = float((int(float(xywh_list[i][3]/im_h)*1000000))/1000000)

                    
                    la = self.cra_label
                    # print(f"la = {la}")
                    CRA_lxywh = str(la) + " " \
                                +str(x) + " " \
                                +str(y) + " " \
                                + str(w) + " " \
                                + str(h)
                    
                    if x<1 and y<1 and w<1 and h<1:
                        CRA_lxywh_list.append([CRA_lxywh])

            if len(CRA_lxywh_list)>0 :
                # Add DCA label into Yolo label.txt
                with open(save_label_path,'a') as f:
                    for i in range(len(CRA_lxywh_list)):
                        f.write(CRA_lxywh_list[i][0])
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
            self.CRA_mask = np.zeros_like(im_lane)
            # Get CRA label map 
            self.get_particular_label_map(R=86,G=94,B=219) 
            # Find contour in CRA map
            self.get_contour(self.im_lane_copy)  
            # Get contour x1y1wh
            self.get_CRA_contour_x1y1wh_list(draw_im=False)
            # Get merged x1y1wh
            # self.merge_overlapped_bounding_boxes()
            # Get CRA mask
            self.get_CRA_mask()
            self.get_contour(self.CRA_mask)  
            self.init()
            # Get contour x1y1wh
            self.get_CRA_contour_x1y1wh_list(draw_im=True)

            # Merge BB
            self.merge_bounding_boxes()

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
            # cv2.rectangle(im,p1,p2,(86,94,219),2,cv2.LINE_AA)
            # cv2.rectangle(im_lane,p1,p2,(86,94,219),2,cv2.LINE_AA)
            cv2.imshow("image",im)
            cv2.imshow("im_lane",im_lane)
            cv2.waitKey()

        
        return self.CRA_merged_x1y1wh_list,self.im_h,self.im_w

    
    def init(self):
        self.CRA_xywh_list = []
        self.CRA_horizontal_x1y1wh_list = []
        self.CRA_vertical_x1y1wh_list = []
        self.CRA_unknown_x1y1wh_list = []
        self.CRA_total_x1y1wh_list = []
        self.CRA_merged_x1y1wh_list = []

    def get_CRA_contour_x1y1wh_list(self,draw_im=False):
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
                    self.CRA_horizontal_x1y1wh_list.append([x1,y1,w,h])
                    self.horizontal_contour.append(self.contours[i])
                    color = (255,255,0)
                elif wh_ratio < 5:
                    self.CRA_vertical_x1y1wh_list.append([x1,y1,w,h])
                    self.vertical_contour.append(self.contours[i])
                    color = (0,255,255)
                else:
                    self.CRA_unknown_x1y1wh_list.append([x1,y1,w,h])
                    self.unknown_contour.append(self.contours[i])
                    color = (255,0,255)

                self.CRA_total_x1y1wh_list.append([x1,y1,w,h])

                # if draw_im:
                #     im = cv2.rectangle(self.im,(x1,y1),(x1+w,y1+h),color,self.rect_width)
                #     im_lane = cv2.rectangle(self.im_lane,(x1,y1),(x1+w,y1+h),color,self.rect_width)

                # YOLO format xywh
                x = x1 + int(w/2.0)
                y = y1 + int(h/2.0)
                self.CRA_xywh_list.append([x,y,w,h])


    def get_CRA_mask(self):
        for i in range(len(self.CRA_total_x1y1wh_list)):
            x1 = self.CRA_total_x1y1wh_list[i][0]
            y1 = self.CRA_total_x1y1wh_list[i][1]
            w = self.CRA_total_x1y1wh_list[i][2]
            h = self.CRA_total_x1y1wh_list[i][3]
           
            self.CRA_mask[y1:y1+h,x1:x1+w] = 255
        
        if self.erode_dilate:
            kernel = np.ones((3,3), np.uint8)
            self.CRA_mask = cv2.dilate(self.CRA_mask, kernel, iterations = self.dilate_iter)
            self.CRA_mask = cv2.erode(self.CRA_mask, kernel, iterations = self.erode_iter)
        
        if self.show_im and len(self.contours)>0:
            if self.resize:
                CRA_mask = cv2.resize(self.CRA_mask, (640, 360), interpolation=cv2.INTER_AREA)
                cv2.imshow('CRA mask', CRA_mask)
                cv2.moveWindow('CRA mask', 40,30)  # Move it to (40,30)
            else:
                cv2.imshow('CRA mask', self.CRA_mask)
                cv2.moveWindow('CRA mask', 40,30)  # Move it to (40,30)

            # cv2.waitKey(0)

    
    def show_CRA_contour_images(self):
        '''
        Purpose : Show CRA contour image 
        '''
        for i in range(len(self.CRA_merged_x1y1wh_list)):
            color = (255,255,0)
            x1 = self.CRA_merged_x1y1wh_list[i][0]
            y1 = self.CRA_merged_x1y1wh_list[i][1]
            w = self.CRA_merged_x1y1wh_list[i][2]
            h = self.CRA_merged_x1y1wh_list[i][3]
            im = cv2.rectangle(self.im,(x1,y1),(x1+w,y1+h),color,5)
            im_lane = cv2.rectangle(self.im_lane,(x1,y1),(x1+w,y1+h),color,5)


        # cv2.drawContours(im_lane, contours, -1, (255, 255, 0), 1)
        # cv2.drawContours(im, contours, -1, (255, 255, 0), 1)  
        if self.resize:
            im_lane = cv2.resize(self.im_lane, (640, 360), interpolation=cv2.INTER_AREA)
            im = cv2.resize(self.im, (640, 360), interpolation=cv2.INTER_AREA)
            # edge = cv2.resize(self.edge, (640, 360), interpolation=cv2.INTER_AREA)
        else:
            im_lane = self.im_lane
            im = self.im
            edge = self.edge
        cv2.imshow('Contours_im_lane', im_lane)
        cv2.imshow('Contours', im)

        cv2.moveWindow('Contours_im_lane', 750,30)  # Move it to (40,30)
        cv2.moveWindow('Contours', 40,550)  # Move it to (40,30)
        # cv2.imshow('Canny Edges After Contouring', edged) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 


    def merge_bounding_boxes(self):
        CRA_total_x1y1wh_enable_list = [1] *  len(self.CRA_total_x1y1wh_list)
        merged = False
        for i in range(len(self.CRA_total_x1y1wh_list)):
            merged = False
            x1y1wh_key = self.CRA_total_x1y1wh_list[i]
            x1_key = x1y1wh_key[0]
            y1_key = x1y1wh_key[1]
            w_key  = x1y1wh_key[2]
            h_key  = x1y1wh_key[3]
            x2_key = x1_key + w_key
            y2_key = y1_key + h_key
            x_c_key = x1_key + int(w_key/2.0)
            y_c_key = y1_key + int(h_key/2.0)
            key_ratio = float(w_key/h_key)
            
            if CRA_total_x1y1wh_enable_list[i] == 1 \
                and (w_key < int(self.im_w * 0.85) or y_c_key < int(self.im_h * 0.75)):
                for j in range(len(self.CRA_total_x1y1wh_list)):
                    if i!=j:
                        x1y1wh_query = self.CRA_total_x1y1wh_list[j]
                        x1_query = x1y1wh_query[0]
                        y1_query = x1y1wh_query[1]
                        w_query  = x1y1wh_query[2]
                        h_query  = x1y1wh_query[3]
                        x_c_query = x1_query + int(w_query/2.0)
                        y_c_query = y1_query + int(h_query/2.0)
                        x2_query = x1_query + w_query
                        y2_query = y1_query + h_query
                        query_ratio = float(w_query/h_query)

                        if  abs(x_c_key-x_c_query)<150 \
                            and abs(y2_key - y1_query) < 150 \
                            and x1y1wh_key!=x1y1wh_query \
                            and merged == False:
                            merged = True
                            print("Merged Case 1")
                        elif key_ratio>20 and query_ratio>10 \
                            and abs(y2_key - y1_query) < 250 \
                            and y2_key < y1_query \
                            and x1y1wh_key!=x1y1wh_query \
                            and merged==False:
                            merged = True
                            print("Merged Case 2")
                        if merged:
                            # merge bounding box
                            merge_x1 = x1_key if x1_key < x1_query else x1_query
                            merge_y1 = y1_key if y1_key < y1_query else y1_query

                            merge_x2 =  x2_key if x2_key > x2_query else x2_query
                            merge_y2 =  y2_key if y2_key > y2_query else y2_query

                            merge_w = merge_x2 - merge_x1
                            merge_h = merge_y2 - merge_y1

                            if merge_w > self.im_w * 0.90:
                                if self.carhoody !=0:
                                    merge_h = (self.carhoody - merge_y1 - 10)
                                else:
                                    merge_h = (self.im_h - merge_y1 - 10)
                            self.CRA_merged_x1y1wh_list.append([merge_x1,merge_y1,merge_w,merge_h])
                            CRA_total_x1y1wh_enable_list[i] = 0
                            CRA_total_x1y1wh_enable_list[j] = 0
            
            # if CRA_total_x1y1wh_enable_list[i] == 1:
            #     self.CRA_merged_x1y1wh_list.append([x1_key,y1_key,w_key,h_key])

        
        for i in range(len(self.CRA_total_x1y1wh_list)):
            x1y1wh_key = self.CRA_total_x1y1wh_list[i]
            x1_key = x1y1wh_key[0]
            y1_key = x1y1wh_key[1]
            w_key  = x1y1wh_key[2]
            h_key  = x1y1wh_key[3]
            x2_key = x1_key + w_key
            y2_key = y1_key + h_key
            x_c_key = x1_key + int(w_key/2.0)
            y_c_key = y1_key + int(h_key/2.0)

            if w_key > self.im_w * 0.90:
                if self.carhoody!=0:
                    h_key = (self.carhoody - h_key - 10)
                else:
                    h_key = (self.im_h - h_key - 10)

            key_ratio = float(w_key/h_key)
            if CRA_total_x1y1wh_enable_list[i] == 1 and key_ratio<30:
                self.CRA_merged_x1y1wh_list.append([x1_key,y1_key,w_key,h_key])
            elif CRA_total_x1y1wh_enable_list[i] == 1 and not (key_ratio>15 and y_c_key>self.im_h*0.70):
                self.CRA_merged_x1y1wh_list.append([x1_key,y1_key,w_key,h_key])



                    
