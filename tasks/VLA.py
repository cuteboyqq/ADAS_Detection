import glob
import os
import shutil
import cv2
from engine.dataset import BaseDataset



class VLA(BaseDataset):

    def Get_Vanish_Area(self):
        '''
        func: Get_Vanish_Area
        Purpose : 
            parsing the images in given image directory, 
            find the vanish line area and crop the vanish line area, 
            and crop others area
        input:
            self.im_dir : the image directory
            self.dataset_dir : the dataset directory
            self.save_dir : save crop image directory
            self.multi_crop : save multiple vanish area crop images
        output:
            the split images
        '''
        im_path_list = glob.glob(os.path.join(self.im_dir,"*.jpg"))

        if self.data_num<len(im_path_list):
            final_wanted_img_count = self.data_num
        else:
            final_wanted_img_count = len(im_path_list)

        # print(f"final_wanted_img_count = {final_wanted_img_count}")
        min_final_2 = None
        for i in range(final_wanted_img_count):
            # print(f"{i}:{im_path_list[i]}")
            drivable_path,lane_path,detection_path = self.parse_path(im_path_list[i],type=self.data_type)
            #print(f"drivable_path:{drivable_path}, \n lane_path:{lane_path}")
            if not os.path.exists(detection_path):
                print(f"detection_path not exist !! PASS:{detection_path}")
                continue
            
            img = cv2.imread(im_path_list[i])
            h,w = img.shape[0],img.shape[1]

            min_final,index = self.Get_Min_y_In_Drivable_Area(drivable_path)
            
            min_final_2 = self.Find_Min_Y_Among_All_Vehicle_Bounding_Boxes(min_final,detection_path,h,w)

            # if os.path.exists(lane_path):
            #     print("lane_path exists!")
            # self.split_Image(im_path_list[i],min_final_2,min_x,min_w,minh)
            self.split_Image(im_path_list[i],min_final_2)


        return min_final_2
    

    def Add_Vanish_Line_Area_Yolo_Txt_Labels(self):
        '''
        func: Get_Vanish_Area
        Purpose : 
            parsing the images in given image directory, 
            find the vanish line area and add bounding box information label x y w h 
            into label.txt of yolo format
        input:
            self.im_dir : the image directory
            self.save_dir : save crop image directory
            
        output:
            the label.txt with vanish line area bounding box
        '''
        im_path_list = glob.glob(os.path.join(self.im_dir,"*.jpg"))

        if self.data_num<len(im_path_list):
            final_wanted_img_count = self.data_num
        else:
            final_wanted_img_count = len(im_path_list)

        print(f"final_wanted_img_count = {final_wanted_img_count}")
        min_final_2 = None
        for i in range(final_wanted_img_count):
            print(f"{i}:{im_path_list[i]}")
            drivable_path,lane_path,detection_path = self.parse_path(im_path_list[i],type=self.data_type)
            #print(f"drivable_path:{drivable_path}, \n lane_path:{lane_path}")
            if not os.path.exists(detection_path):
                print(f"detection_path not exist !! PASS:{detection_path}")
                continue
            
            img = cv2.imread(im_path_list[i])
            h,w = img.shape[0],img.shape[1]

            min_final,index = self.Get_Min_y_In_Drivable_Area(drivable_path)
            # print(f"min_final = {min_final}")
            #input()
            min_final_2 = self.Find_Min_Y_Among_All_Vehicle_Bounding_Boxes(min_final,detection_path,h,w)

            # print(f"min_final_2 :{min_final_2}")
            
            success = self.Add_VLA_Yolo_Txt_Label(min_final_2,detection_path,h,w,im_path_list[i])
            
            #input()
        return min_final_2

    def Add_VLA_Yolo_Txt_Label(self,min_y,detection_path,h,w,im_path):
        success = 0
        # with open(detection_path,'r') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         print(line)
        txt_file = detection_path.split(os.sep)[-1]
        os.makedirs(self.save_txtdir,exist_ok=True)
        save_txt_path = os.path.join(self.save_txtdir,txt_file)

        ## Copy original label.txt to new directory
        if not os.path.exists(save_txt_path):
            shutil.copy(detection_path,save_txt_path)
            if self.save_img:
                shutil.copy(im_path,self.save_txtdir)
            print(f"Copy detection_path to :{save_txt_path} successful !!")
        else:
            print(f"File {save_txt_path} exists ~~~~~~~~~~~~~~,PASS!!")
            success = 1
            return success

        ## Add new VLA label to label.txt
        VLA_label = self.vla_label
        x = float(int((float(int(w/2.0)-1)/w)*1000000)/1000000)
        y = float(int(float(min_y/h)*1000000)/1000000)
        w = 1.0
        h = float(int(float(self.split_height / h)*1000000)/1000000)
        lxywh =  str(VLA_label) + " "\
                + str(x) + " "\
                + str(y) + " "\
                + str(w) + " "\
                 + str(h)
        with open(save_txt_path,'a') as f:
            # Add VLA(Vanish Line Area Bounding Box) l x y w h
            f.write("\n")
            f.write(lxywh)
        
        success = 1
        return success
    
    # def split_Image(self,im_path,drivable_min_y,min_x,min_w,minh):
    def split_Image(self,im_path,drivable_min_y):
        tag = 0
        retval = 0
        lower_bound_basic = 0
        upper_bound_basic = 0
        if os.path.exists(im_path):
            img_name = (im_path.split(os.sep)[-1]).split(".")[0]
            print(im_path)
            label=None
            y = 0 if drivable_min_y is None else int(drivable_min_y)
            print(f"y:{y}")
            img = cv2.imread(im_path)
            split_y = int(img.shape[0] / self.split_num)
            h,w = img.shape[0],img.shape[1]
            bound_list = []
            if not self.multi_crop:
                lower_bound_basic = y-int(split_y/2.0)
                upper_bound_basic = y+int(split_y/2.0)
                bound_list.append([lower_bound_basic,upper_bound_basic])
            else:
                lower_bound_basic = y-int(split_y/2.0)
                upper_bound_basic = y+int(split_y/2.0)
                bound_list.append([lower_bound_basic,upper_bound_basic])
                for i in range(int(self.multi_num/2.0)):
                    lower_bound = y-int(split_y/2.0)-(i+1)*int(self.shift_pixels)
                    upper_bound = y+int(split_y/2.0)-(i+1)*int(self.shift_pixels)
                    bound_list.append([lower_bound,upper_bound])
                for i in range(int(self.multi_num/2.0)):
                    lower_bound = y-int(split_y/2.0)+(i+1)*int(self.shift_pixels)
                    upper_bound = y+int(split_y/2.0)+(i+1)*int(self.shift_pixels)
                    bound_list.append([lower_bound,upper_bound])

            for i in range(len(bound_list)):
                lower_bound = bound_list[i][0]
                upper_bound = bound_list[i][1]
                print(f"split_y:{split_y}")
                if lower_bound>0:
                    split_vanish_crop_image = img[lower_bound:upper_bound,0:w-1]
                    
                    #input()
                    if self.save_imcrop:
                        label = 0
                        save_dir = os.path.join(self.save_dir,str(label))
                        os.makedirs(save_dir,exist_ok=True)
                        save_crop_im = img_name + "_" + str(tag) + ".jpg"
                        save_crop_im_path = os.path.join(save_dir,save_crop_im)
                        if not os.path.exists(save_crop_im_path):
                            cv2.imwrite(save_crop_im_path,split_vanish_crop_image)
                        else:
                            print(f"file {img_name}_0.jpg exists")
                        tag=tag+1
                    if self.show_imcrop:
                        if self.show_vanishline:
                            newImage = img.copy()
                            cv2.line(newImage, (0, drivable_min_y), (w-1, drivable_min_y), (0, 0, 255), 1)
                            #cv2.rectangle(newImage, start_point, end_point, color, thickness) 
                            split_vanish_crop_image = newImage[lower_bound:upper_bound,0:w-1]
                            cv2.imshow("split vanish area",split_vanish_crop_image)
                            cv2.waitKey(1)
                            input()
                        else:
                            cv2.imshow("split vanish area",split_vanish_crop_image)
                            cv2.waitKey(1)
                else:
                    print("y is too small~~")
                    retval = -1
                    return retval
                
            ## Generate others crop images
            if y>split_y:
                y_l = lower_bound_basic - split_y
                while(y_l>0 and y_l+split_y<h):
                    split_other_image = img[y_l:y_l+split_y,0:w-1]
                    # cv2.imshow("up others img",split_other_image)
                    # cv2.waitKey(100)
                    #input()
                    y_l=y_l-split_y
                    if self.save_imcrop:
                        label = 1
                        save_dir = os.path.join(self.save_dir,str(label))
                        os.makedirs(save_dir,exist_ok=True)
                        save_crop_im = img_name + "_" + str(tag) + ".jpg"
                        save_crop_im_path = os.path.join(save_dir,save_crop_im)
                        if not os.path.exists(save_crop_im_path):
                            cv2.imwrite(save_crop_im_path,split_other_image)
                        else:
                            print(f"file {img_name}_{tag}.jpg exists")
                        tag=tag+1
                y_t = upper_bound_basic + split_y
                while(y_t<(h-1) and y_t-split_y>0):
                    split_other_image = img[y_t-split_y:y_t,0:w-1]
                    # cv2.imshow("down others img",split_other_image)
                    # cv2.waitKey(100)
                    #input()
                    y_t = y_t + split_y
                    if self.save_imcrop:
                        label = 1
                        save_dir = os.path.join(self.save_dir,str(label))
                        os.makedirs(save_dir,exist_ok=True)
                        save_crop_im = img_name + "_" + str(tag) + ".jpg"
                        save_crop_im_path = os.path.join(save_dir,save_crop_im)
                        if not os.path.exists(save_crop_im_path):
                            cv2.imwrite(save_crop_im_path,split_other_image)
                        else:
                            print(f"file {img_name}_{tag}.jpg exists")
                        tag=tag+1
            
        return NotImplemented
    
    





