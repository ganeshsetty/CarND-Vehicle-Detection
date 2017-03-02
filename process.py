# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 19:42:32 2017

@author: setty
"""
import numpy as np
import pickle
from utils import *
#from train import *
def init_global():
    global count,first_frame
    count = 0
    first_frame = 1
    


model_pickle = pickle.load( open( "svc_pickle_allsamples.p", "rb" ) )
svc = model_pickle['svc']
X_scaler = model_pickle['scaler']
parameters = model_pickle['parameters']
color_space = parameters['color_space']
orient = parameters['orient']
pix_per_cell = parameters['pix_per_cell']
cell_per_block = parameters['cell_per_block']
hog_channel = parameters['hog_channel']
spatial_size = parameters['spatial_size']
hist_bins = parameters['hist_bins']
spatial_feat = parameters['spatial_feat']
hist_feat = parameters['hist_feat']
hog_feat = parameters['hog_feat']

def process_detect(img):
    
    xstart = 0
    xstop = 1280
    ystart = 400
    ystop = 420
       
    heatmap_sum = np.zeros_like(img[:,:,0]).astype(np.float)
    for scale in (1.0,1.5,2.0):
        ystop = ystop + 75
        
        out_img,heatmap = find_cars(img,xstart, xstop, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, 
                                    cell_per_block, spatial_size, hist_bins)
        
        heatmap_sum = np.add(heatmap_sum,heatmap)
    
    # Apply threshold to help remove false positives
    heatmap = apply_threshold(heatmap_sum,2)
            
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heatmap, 0, 255)
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
            
    draw_img,bbox_list = draw_labeled_bboxes(np.copy(img), labels)
        
    return draw_img,heatmap,labels
    
def process_track(img):
    
    global heatmap_sum,draw_img_prevset,labels_prevset,bbox_list_prevset
    global first_frame,count
    xstart = 100
    xstop = 1280
    ystart = 400
    ystop = 420
    
    if first_frame == 1:
        heatmap_sum = np.zeros_like(img[:,:,0]).astype(np.float)
        for scale in (1.0,1.5,2.0):
            #xstart = xstart - 100
            ystop = ystop+75
            out_img,heatmap = find_cars(img,xstart, xstop, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, 
                                    cell_per_block, spatial_size, hist_bins)
        
            heatmap_sum = np.add(heatmap_sum,heatmap)
          
        # Apply threshold to help remove false positives
        heatmap = apply_threshold(heatmap_sum,2)
        
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heatmap, 0, 255)
    
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
            
        draw_img,bbox_list = draw_labeled_bboxes(np.copy(img), labels)
        
        draw_img_prevset = draw_img
        bbox_list_prevset = bbox_list
        labels_prevset = labels
        count = 0
               
        first_frame = 0
        count = count + 1
    
        return draw_img
     
    if count <= 3:
        #xstart = 400
        ystop = 400
        for scale in (1.0,1.5,2.0):
            #xstart = xstart -100
            ystop = ystop+100
            
            out_img,heatmap = find_cars(img,xstart, xstop, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, 
                                    cell_per_block, spatial_size, hist_bins)
            heatmap_sum = np.add(heatmap_sum , heatmap)
            
        count = count + 1
        draw_img = img
        
        for car_number in range(1, labels_prevset[1]+1):
            draw_img = cv2.rectangle(draw_img, bbox_list_prevset[car_number-1][0],
                                     bbox_list_prevset[car_number-1][1], (255,0,0), 6)
        
        return draw_img
    
    # Apply threshold to help remove false positives
    heatmap = apply_threshold(heatmap_sum,2)
        
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heatmap, 0, 255)
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
            
    draw_img, bbox_list = draw_labeled_bboxes(np.copy(img), labels)
           
    draw_img_prevset = draw_img
    bbox_list_prevset = bbox_list
    labels_prevset = labels
    
    heatmap_sum = np.zeros_like(img[:,:,0]).astype(np.float)
    count = 0
          
    return draw_img
   
    