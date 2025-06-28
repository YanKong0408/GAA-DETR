import cv2
import numpy as np
import cv2
import os
import csv

csv_files=[]
raw_data_fold = 'c:\\Users\\kongyan\\Desktop\\dataset\\Fungus\\raw_gaze_data\\data_1'
img_fold = 'c:\\Users\\kongyan\\Desktop\\dataset\\Fungus\\all_raw\\data_1_cropped\\'
thre = 0.3 
GuassKernal=(119,119)

for file_name in os.listdir(raw_data_fold):
    if file_name.endswith('.csv'):
        csv_files.append(file_name)

for file_name in csv_files:
    file_path = os.path.join(raw_data_fold, file_name)
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        print(f"Reading file: {file_name}")
        for line in csv_file:
            line = line.strip()
            if line:
                imgName, _, gazeData, bboxs, _ = line.split(';')
                gazeData, bboxs=eval(gazeData), eval(bboxs)
                new_gazes=[]
                new_bboxs=[]
                for gaze in gazeData:
                    new_gazes.append([int(gaze[0][0]/gaze[1]),int(gaze[0][1]/gaze[1])])
                for bbox in bboxs:
                    new_bboxs.append((int(bbox[0]/bbox[4]),int(bbox[1]/bbox[4]),int(bbox[2]/bbox[4]),int(bbox[3]/bbox[4])))
                gazeData = new_gazes
                bboxs=new_bboxs
                img=cv2.imread(img_fold + imgName.split('\\')[-1])
                print(len(gazeData))
                image_shape = img.shape
                canvas = np.zeros((image_shape[0],image_shape[1]))

                # ===================== Modify this code to achieve better attention ================================== #
                for gaze in gazeData:
                    if gaze[0] < image_shape[0] and gaze[1] < image_shape[1]:                        
                        canvas[gaze[0]][gaze[1]] += np.exp(-canvas[gaze[0]][gaze[1]])
                g = cv2.GaussianBlur(canvas, GuassKernal, 0, 0)
                g = cv2.normalize(g, None, alpha=0, beta=1,
                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                g = g.T
                # g_expend=cv2.applyColorMap((g*255).astype(np.uint8), cv2.COLORMAP_JET)
                # heapmapimage = cv2.addWeighted(img,0.3,g_expend,0.7,0)
                # cv2.imshow('',heapmapimage)
                # cv2.waitKey(0)

                # remove attention with low value
                _, thresholded_map = cv2.threshold(g, 0.15, 1, cv2.THRESH_BINARY)
                g_expend=cv2.applyColorMap((g*thresholded_map*255).astype(np.uint8), cv2.COLORMAP_JET)
                heapmapimage = cv2.addWeighted(img,0.3,g_expend,0.7,0)
                cv2.imshow('',heapmapimage)
                cv2.waitKey(0)
                cv2.imwrite('C:\\Users\\kongyan\\Desktop\\PPTs\\Gaze-DETR+\\image\\'+ imgName.split('\\')[-1].replace('.jpg','_heatmap.jpg'), heapmapimage)

                # # remove attention with small size
                # thresholded_map = (thresholded_map * 255).astype(np.uint8)
                # num_labels, labels_im = cv2.connectedComponents(thresholded_map)
                # new_threshold_map = np.zeros_like(thresholded_map)
                # for label in range(1, num_labels): 
                #     mask = (labels_im == label).astype(np.uint8)
                #     area = np.sum(mask)
                #     if area >= 1000:
                #         new_threshold_map[mask > 0] = 255  
                # new_threshold_map = (new_threshold_map > 0).astype(np.uint8)

                # print(thresholded_map)
                # g_thresholded_map = thresholded_map * g
                # g_thresholded_map=cv2.applyColorMap((g_thresholded_map*255).astype(np.uint8), cv2.COLORMAP_JET)
                # heapmapimage = cv2.addWeighted(img,0.3,g_thresholded_map,0.7,0)

                # g_new = new_threshold_map * g
                # g_new = cv2.applyColorMap((new_threshold_map*255).astype(np.uint8), cv2.COLORMAP_JET)
                # new_heapmapimage = cv2.addWeighted(img,0.3,g_new,0.7,0)

                # i = np.hstack((heapmapimage, new_heapmapimage))
                # cv2.imshow('',i)
                # cv2.waitKey(0)
                heatmap = thresholded_map * g * 255
                heatmap = heatmap.astype(np.uint8)
                # ======================================================================================================= #
                                