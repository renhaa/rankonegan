import os
import sys
import itertools
import glob


import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import PIL
from PIL import Image

import cv2
import dlib
import imutils
from imutils import face_utils
import tensorflow as tf
#tf.keras.utils.get_file
from tensorflow.keras.utils import get_file
#from utils.ffhq_dataset.face_alignment import image_align

def unpack_bz2(src_path):
    import bz2
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path
    
LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
LANDMARKS_MODEL_PATH = unpack_bz2(get_file(
    'shape_predictor_68_face_landmarks.dat.bz2', LANDMARKS_MODEL_URL, cache_subdir='pretrained_models/'))
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(LANDMARKS_MODEL_PATH)


def extract_landmark(img, landmark_detector = "mediapipe", verbose = False):
    if landmark_detector == "dlib":
        landmark = get_boundingbox_and_landmarks(img)[1]
    elif landmark_detector == "mediapipe":
        landmark = get_mediapipe_landmarks(img, denormalize = True, _2D = True) 
    else:
        raise NotImplementedError("Use dlib / mediapipe")
    if landmark is None and verbose: 
        print("[Warning]: No face found!!")
    return landmark


def plot_dlib_lines(landmark, color = "b"):
    assert landmark.shape == (68,2)
    dlib_lines_idx =   [(0, 16),     # Jaw line
                    (17, 21),    # Left eyebrow
                    (22, 26),    # Right eyebrow
                    (27, 30),    # Nose bridge
                    (30, 35),    # Lower nose closed
                    (36, 41),    # Left eye closed
                    (42, 47),    # Right Eye closed
                    (48, 59),    # Outer lip closed
                    (60, 67) ]  # Inner lip closed
    dlib_lines_idx_closed = [0,0,0,0,1,1,1,1,1]
    def range1(start, end):
        return range(start, end+1)
    for closed, line_idx in zip(dlib_lines_idx_closed,dlib_lines_idx):
        points = np.array([landmark[i] for i in range1(*line_idx)])
        if closed:
            points = np.row_stack((points, points[0]))
        plt.plot(*points.T, c = color)

def get_mediapipe_landmarks(img, denormalize = True, _2D = True):
    #https://stackoverflow.com/questions/67141844/how-do-i-get-the-coordinates-of-face-mash-landmarks-in-mediapipe
    import mediapipe as mp
    import cv2
    mp_face_mesh = mp.solutions.face_mesh


    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


    landmarks = np.zeros((3,468))
    if results.multi_face_landmarks is None:
        return None

    for i,l in enumerate(results.multi_face_landmarks[0].landmark):
        landmarks[:,i] = np.array([l.x,l.y,l.z])
    if denormalize: 
        shape = img.shape 
        landmarks[0,:] = landmarks[0,:] * shape[1]
        landmarks[1,:] = landmarks[1,:] * shape[0] 
    if _2D: 
        landmarks = landmarks[:2]
    return np.array(landmarks.T, dtype = int)


def get_boundingbox_and_landmarks(im, plot = False, verbose = False ):

    if not type(im) is np.ndarray:
        im = np.array(im)
    rects = detector(im, 1)
    if len(rects) == 0 and verbose:
        print("[WARNING]", len(rects), " faces detected detected") 
        return None,None

    if len(rects) > 1 and verbose:
        print("[WARNING]", len(rects), " faces detected detected")
    assert not rects is None
    
    try:
        rect = rects[0]
    except:
        if verbose:
            print("[DLIB FAILED]")
        return None, None
        
  
    face_landmarks = get_landmarks(shape_predictor,im,rect)
    if plot:
        plot_rects(im, rects[0])
        plot_landmarks(im,face_landmarks)
        plt.axis("off")
    return rect, face_landmarks

def image_resize(image,
                 width = None,
                 height = None,
                 inter = cv2.INTER_AREA):
    image = np.array(image)
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    elif width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = inter)
    resized = PIL.Image.fromarray(resized)
    return resized

def add_borders(img):
        old_size = img.size
        if np.max(old_size) > 1024:
            idx = np.argmax(old_size)
            if idx == 1:
                img = image_resize(img,width = None, height = 1024)
            else:
                img = image_resize(img,width = 1024,height = None)
        new_size = (1024, 1024)
        new_img = Image.new("RGB", new_size)  
        new_img.paste(img, (int((new_size[0]-old_size[0])/2),int((new_size[1]-old_size[1])/2)))
        return new_img


def get_mask(im,rect,shape,use_grabcut = True, scale_mask = 1.5 ):

    shape = face_utils.shape_to_np(shape)

    # we extract the face
    vertices = cv2.convexHull(shape)
    mask = np.zeros(im.shape[:2],np.uint8)
    cv2.fillConvexPoly(mask, vertices, 1)
    if use_grabcut:
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (0,0,im.shape[1],im.shape[2])
        (x,y),radius = cv2.minEnclosingCircle(vertices)
        center = (int(x),int(y))
        radius = int(radius*scale_mask)
        mask = cv2.circle(mask,center,radius,cv2.GC_PR_FGD,-1)
        cv2.fillConvexPoly(mask, vertices, cv2.GC_FGD)
        cv2.grabCut(im,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
        mask = np.where((mask==2)|(mask==0),0,1)

    return mask.astype("uint8")


def plot_rects(im, rect):
    x1,y1 = rect.tl_corner().x,rect.tl_corner().y
    x2,y2 = rect.br_corner().x,rect.br_corner().y
    image = cv2.rectangle(im, (x1,y1), (x2,y2) , color = (0,0,255), thickness = 4)
   
    return image

# FACE_POINTS = list(range(17, 68))
# MOUTH_POINTS = list(range(48, 61))
# RIGHT_BROW_POINTS = list(range(17, 22))
# LEFT_BROW_POINTS = list(range(22, 27))
# RIGHT_EYE_POINTS = list(range(36, 42))
# LEFT_EYE_POINTS = list(range(42, 48))
# NOSE_POINTS = list(range(27, 35))
# JAW_POINTS = list(range(0, 17))
# CHIN_POINTS=list(range(6,11))

def get_landmarks(shape_predictor,img,rect):
    try:
        face_landmarks = np.array([np.array([item.x, item.y]) for item in shape_predictor(img, rect).parts()])
        return face_landmarks
    except:
        print("Exception in get_landmarks()!")

def plot_landmarks(im,face_landmarks):
    landmark_image = im.copy()
    for point in face_landmarks:
        cv2.circle(landmark_image,tuple(point),2,(0,255,0),-1)
    plt.imshow(landmark_image)

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def preprocess_img(img, save = False):
    if not img.mode == "RGB":
        img = img.convert("RGB")
    img = add_borders(img)
    rect, face_landmarks = get_boundingbox_and_landmarks(img, plot = False)
    if face_landmarks is None:
        print("[WARNING], skipping FFHQ alignment step file:", save)
    else:
        img = image_align(img, face_landmarks, save = False)
    if save:
        img.save(save)
        print("[INFO] Aligned image saved to:", save)
    return img