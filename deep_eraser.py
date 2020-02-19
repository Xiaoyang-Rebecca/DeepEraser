# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 17:20:26 2019

@author: xli63
"""
import os
import sys
import gc

import random
import math
import re
import time
import numpy as np
import cv2
from skimage import io,img_as_ubyte
from skimage import segmentation,morphology
import warnings
warnings.filterwarnings("ignore")

import matplotlib
#     # Agg backend runs without a display
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Root directory of the project
mrcnn_root = os.path.abspath( r"../SegmentationPipeline_maui/mrcnn_Seg")
COCO_MODEL_PATH = os.path.join(mrcnn_root, "mask_rcnn_coco.h5")
print (os.path.abspath(COCO_MODEL_PATH))
sys.path.append(mrcnn_root)  # To find local version of the library
checkpoint_dir=r"/model_logs/places2"    

COCO_DIR = os.path.abspath("../cocoapi/PythonAPI/")
sys.path.append(COCO_DIR)	# To find local version of the library
import pycocotools as pycocotools

#%%
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    image_masked = image.copy()
    for c in range(3):
        image_masked[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] ,
                                  image[:, :, c])
    return image_masked
#%%
''' MRCNN: load coco model'''
def deep_rat_detect (image, tname='baseball bat'):
    # Directory to save logs and trained model

    # Import Mask RCNN
    from mrcnn import utils
    from mrcnn import visualize
    from mrcnn.visualize import display_images
    import mrcnn.model as modellib
    from mrcnn.model import log
    import tensorflow as tf
    from samples.coco import coco

    MODEL_DIR = os.path.join(mrcnn_root, "logs")
    
    # Local path to trained weights file
    # Download COCO trained weights from Releases if needed
    # changes for inferencing.
    
    config = coco.CocoConfig()
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    
    config = InferenceConfig()
    # config.display()
    #Load Model
    # Set weights file path
    config.NAME == "coco"
    weights_path = COCO_MODEL_PATH
    dataset = coco.CocoDataset()
    dataset.class_names=['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
                'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 
                'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 
                'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
                 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                  'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
                   'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    DEVICE = "/gpu:0 "  # /cpu:0 or /gpu:0    
    # Inspect the model in training or inference modes
    # values: 'inference' or 'training'
    
    # Create model in inference mode
    with tf.device(DEVICE):
        model_mrcnn = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)   
        # Load weights
        model_mrcnn.load_weights(weights_path, by_name=True)  
        # Run object detection
        results = model_mrcnn.detect([image], verbose=0)
    # tf.reset_default_graph()

    # del model_mrcnn
    gc.collect()
    
    # Display results 
    r = results[0]                                                      
    
    search_id = dataset.class_names.index(tname)
    if search_id in r['class_ids'] :
        
        obj_id_baseball_rat = np.where( r['class_ids'] == search_id)[0]  # 35 is baseball rat detection
        mask_br = r['masks'][:,:,obj_id_baseball_rat][:,:,0]
        #%%
        mask_br = morphology.binary_dilation (mask_br,morphology.disk(5))   
        mask = np.dstack([mask_br]*(image.shape[2]))  # true for mask,0 for background
            
        # visualize the detection result in border
        borders = segmentation.find_boundaries(mask_br)
        borders = morphology.binary_dilation (borders,morphology.disk(2))   # borden the borders  
        border_coords = np.where(borders)
        color = [255,255,0]  # yellow color
        borded_image = apply_mask(image, mask_br.copy(), color, alpha=0.5)  # appied mask
        borded_image[border_coords[0],border_coords[1] , :] = color    # applied border
        #%%
        
    else:
        mask = None
        borded_image = None


    return mask,borded_image
        
''' Smooth Erasor'''
def pixel_fill (image,mask):
    import tensorflow as tf2
    import neuralgym as ng
    from inpaint_model import InpaintCAModel

    if image.ndim > mask.ndim:
        mask = np.dstack([mask]*image.shape[2])
    assert image.shape == mask.shape
    model = InpaintCAModel()
    FLAGS = ng.Config('inpaint.yml')

    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    sess_config = tf2.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf2.Session(config=sess_config) as sess:
        input_image = tf2.constant(input_image, dtype=tf2.float32)
        output = model.build_server_graph(FLAGS, input_image)
        output = (output + 1.) * 127.5
        output = tf2.reverse(output, [-1])
        output = tf2.saturate_cast(output, tf2.uint8)
        # load pretrained model
        vars_list = tf2.get_collection(tf2.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        print ("checkpoint_dir = ",checkpoint_dir)
        for var in vars_list:
            vname = var.name
            from_name = vname
            if "inpaint_net" in var.name:    # or else is going to mix with mrcnn
                var_value = tf2.contrib.framework.load_variable(checkpoint_dir, from_name)
                assign_ops.append(tf2.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.'*10)
        result = sess.run(output)
    sess.close() 
    tf2.reset_default_graph()
    return result[0][:, :, ::-1]   

##%%
#image= io.imread(r"/project/roysam/xli63/exps/generative_inpainting/examples/astros/raw_img.png")
#
#model_mrcnn = load_coco_model (ROOT_DIR)  # only once 
#
# #%%
# image = io.imread(r"D:\Rebecca. Li\application\astros\raw_img.png")
# mask_br  = io.imread(r"D:\Rebecca. Li\application\astros\mask_pad.png")
#filled_image = pixel_fill (image,mask)
#%%
#print ("filled_image successed!")
#io.imsave("filled_image.png",filled_image)


def detect_and_color_splash(write_path , image_path=None, video_path=None,tname='baseball bat'):
    # model_mrcnn = load_coco_model ()  # only once 
    
    if os.path.exists(write_path) is False:
        os.mkdir(write_path)        
        
    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        file_name_prefix = os.path.basename(image_path).split(".")[0]

        # Read image
        image = io.imread(args.image)
        # Detect objects
        mask,borded_image = deep_rat_detect (image, tname)

        # pixel_fill
        if mask is not None:
            #%%
            image_masked = image.copy()
            image_masked= image*(~mask)
            #%%
            filled_image = pixel_fill (image_masked,mask*255)
            print ("\n"*10,"*"*10,"Output saved")
            # Save output
        else:
            print ("\n"*10,"*"*10,"No "+tname+" detected!")
            borded_image = image.copy()
            filled_image = image.copy()
        io.imsave(os.path.join( write_path,file_name_prefix+  "_borded.png"), borded_image)
        io.imsave(os.path.join( write_path,file_name_prefix+  "_erased.png"), filled_image)
        io.imsave(os.path.join( write_path,file_name_prefix+  "_masked.png"), image_masked)

            
    elif video_path:
        # Video capture
        print("Running on {}".format(video_path))
        file_name_prefix = os.path.basename(video_path).split(".")[0]
        vcapture = cv2.VideoCapture(video_path)
        
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)
        # Define codec and create video writer
        border_writer = cv2.VideoWriter(os.path.join( write_path,file_name_prefix+  "_borded.avi"),
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))
        erased_writer = cv2.VideoWriter(os.path.join( write_path,file_name_prefix+  "_erased.avi"),
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))
        count = 0
        success = True
        mask_dic = {}
        image_dic = {}
        while success:
            print("*"*20+ "frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success :
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                mask,borded_image = deep_rat_detect (image)
                # mask_ls .append(mask)
                # pixel_fill
               
                if mask is not None:
                    image_masked = image.copy()
                    image_masked[mask] = 0                    
                    # io.imsave(os.path.join( write_path,file_name_prefix+  "_mask-frame"+str(count)+".png"),  img_as_ubyte(mask))
                    print ("\n"*10,"*"*10,"Output saved")
                    # Save output
                else:
                    print ("\n"*10,"*"*10,"No "+tname+" detected!")
                    borded_image = image.copy()

                mask_dic[count] = mask
                image_dic[count] = image

                border_writer.write(borded_image[..., ::-1])
                # Add image to video writer
                count += 1
        border_writer.release()

        print ("-----------------------fill")
        for count in image_dic:
            # Detect objects
            # maskfname = file_name_prefix+  "_mask-frame"+str(count)+".png"
            # # mask_ls .append(mask)
            # # pixel_fill
            # print ("load " ,maskfname)
            image = image_dic[count]
            if mask_dic[count] is not None :                
                mask = mask_dic[count]
                image_masked= image*(~mask)
                filled_image = pixel_fill (image_masked,mask*255)
                # io.imsave(os.path.join( write_path,file_name_prefix+  "_filled-frame"+str(count)+".png"),  img_as_ubyte(filled_image))
                print ("\n"*10,"*"*10,"Output saved")
            else:
                print ("No " +tname+" detected!")
                filled_image = image.copy()                                       
            erased_writer.write(filled_image[..., ::-1])
            # Add image to video writer
        erased_writer.release()        
        
    print("Saved to ", write_path)

if __name__ == '__main__':
    import argparse,time
    t0=time.time()
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')

    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('-o',"--output_dir", required=True,
                        metavar="path or write the result",
                        help='Video to apply the color splash effect on')
    parser.add_argument('-t',"--tname", required=True,default = "baseball bat",
                        metavar="type name to erase",
                        help='type name to erase')
    args = parser.parse_args()
    
    detect_and_color_splash(write_path = args.output_dir,image_path=args.image, video_path=args.video,tname=args.tname)
    print ("Total time:", time.time() - t0)


