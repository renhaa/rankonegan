import numpy as np
import scipy.ndimage
import PIL.Image
import cv2

import PIL
from PIL import Image
from imutils import face_utils
from matplotlib import pyplot as plt

from tensorflow.keras.utils import get_file

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
# detector = dlib.get_frontal_face_detector()
# shape_predictor = dlib.shape_predictor(LANDMARKS_MODEL_PATH)

def image_align(img, face_landmarks, save = False, output_size=1024, transform_size=4096, enable_padding=True, x_scale=1, y_scale=1, em_scale=0.1, alpha=False):
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    lm = np.array(face_landmarks)
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    x *= x_scale
    y = np.flipud(x) * [-y_scale, y_scale]
    c = eye_avg + eye_to_mouth * em_scale
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2


    img = PIL.Image.fromarray(img)
    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = np.uint8(np.clip(np.rint(img), 0, 255))
        if alpha:
            mask = 1-np.clip(3.0 * mask, 0.0, 1.0)
            mask = np.uint8(np.clip(np.rint(mask*255), 0, 255))
            img = np.concatenate((img, mask), axis=2)
            img = PIL.Image.fromarray(img, 'RGBA')
        else:
            img = PIL.Image.fromarray(img, 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
    
    if save: 
        img.save(save, 'PNG')
    return np.array(img)
    
def get_boundingbox_and_landmarks(im, plot = False):
    import dlib
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(LANDMARKS_MODEL_PATH)

    if not type(im) is np.ndarray:
        im = np.array(im)
    rects = detector(im, 1)
    if len(rects) == 0:
        print("[WARNING]", len(rects), " faces detected detected") 
        return None,None

    if len(rects) > 1:
        print("[WARNING]", len(rects), " faces detected detected")

    rect = rects[0]
  
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

def get_landmarks(shape_predictor,img,rect):
    try:
        face_landmarks = [(item.x, item.y) for item in shape_predictor(img, rect).parts()]
        return face_landmarks
    except:
        print("Exception in get_landmarks()!")

def plot_landmarks(im,face_landmarks):
    landmark_image = im.copy()
    for point in face_landmarks:
        cv2.circle(landmark_image,point,10,(255,0,0),-1)
    plt.imshow(landmark_image)


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


