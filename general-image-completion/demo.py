from glob import glob
import os
import numpy as np
import cv2
from graph_mscoco import *

pen_size = 3
img_idx = 0
drawing = False
ix, iy = -1, -1
vis_size = 320
blank_size = 20

# Functions
def nothing(x):
    pass
  
def draw(event, x, y, flags, param):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(img, (ix, iy), (x, y), (0.9, 0.01, 0.9), pen_size)
            ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (ix, iy), (x, y), (0.9, 0.01, 0.9), pen_size)

def masking(img):
    mask = (np.array(img[:,:,0]) == 0.9) & (np.array(img[:,:,1]) == 0.01) & (np.array(img[:,:,2]) == 0.9)
    mask = np.dstack([mask,mask,mask]);
    return (True ^ mask) * np.array(img)


# Read data path
img_paths = []
img_paths.extend( sorted(glob(os.path.join('testimages/', '*.bmp'))) )
img_ori = cv2.imread( img_paths[img_idx]) / 255.
img = img_ori
empty = np.zeros((vis_size, vis_size, 3))
blank = np.zeros((vis_size, blank_size, 3)) + 1
text_region = np.zeros((blank_size, 2*vis_size + blank_size, 3)) + 1.

cv2.namedWindow("General Completion Demo", cv2.WINDOW_NORMAL) 
cv2.setMouseCallback('General Completion Demo', draw)

###-------------------------------------
# Only CPU Version is available
sess = tf.InteractiveSession()
# Pre-train paths
pretrained_model_path = 'model_mscoco'
# Check whehter is in the training stage
is_train = tf.placeholder( tf.bool )
# Input image 
images_tf = tf.placeholder( tf.float32, shape=[1, vis_size, vis_size, 3], name="images")
# Generate image
model = Model()
reconstruction_ori = model.build_reconstruction(images_tf, is_train)
# Set the number of checkpoints that you need to save
saver = tf.train.Saver(max_to_keep=100)
# Restore Model
saver.restore( sess, pretrained_model_path )
###-------------------------------------


# Fake button widgets
cv2.createTrackbar('Pen Size','General Completion Demo',1,10,nothing)

# Set font for put tex
font = cv2.FONT_ITALIC

# Initial reconstructed image as empty
recon_img = empty

# Mainloop
while (1):

    # Show window
    view = np.hstack((img, blank,recon_img[:,:,[2,1,0]]))
    window = np.vstack( (view, text_region) )
    cv2.imshow('General Completion Demo', window)
    
    # Show text (the position is mannuly selected)
    cv2.putText(text_region,'Original Image',(110,15), font, 0.4,(0,0,0),1)
    cv2.putText(text_region,'Completed Image',(130+vis_size,15), font, 0.4,(0,0,0),1)
    
    # Interactive keys
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == 99: # Convert (c)
        masked_input = masking(img)
        masked_input = masked_input[:,:,[2,1,0]]
        shape3d = np.array( masked_input ).shape
        model_input = np.array( masked_input ).reshape(1, shape3d[0], shape3d[1], shape3d[2])
        model_output = sess.run(reconstruction_ori,feed_dict={images_tf: model_input, is_train: False})
        recon_img = np.array(model_output)[0,:,:,:].astype(float)
        cv2.imwrite( os.path.join('results', img_paths[img_idx][21:35]), ((recon_img[:,:,[2,1,0]]) * 255) )
        cv2.imwrite( os.path.join('inputs', img_paths[img_idx][21:35]), ((img) * 255) )
    elif k == 114: # Reset (r)
        img_ori = cv2.imread( img_paths[img_idx]) / 255.
        img = img_ori
        recon_img = empty
    elif k == 110: # Next image (n)
        img_idx = (img_idx + 1) % len(img_paths)
        img_ori = cv2.imread( img_paths[img_idx]) / 255.
        img = img_ori 
        recon_img = empty

    # Adjust pen size
    pen_size = cv2.getTrackbarPos('Pen Size','General Completion Demo')


cv2.destroyAllWindows()
