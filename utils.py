
import torch
import torchvision.transforms as T

import cv2
from PIL import Image
from PIL import Image, ImageDraw ,ImageFont

import matplotlib.pyplot as plt

from face_alignment import get_boundingbox_and_landmarks, image_align
from face_utils import extract_landmark
from face_utils import plot_dlib_lines


def get_webcam_frame():
    cap = cv2.VideoCapture(0)
    # Grab a single frame of video
    ret, frame = cap.read()
    frame = frame[:,:,::-1]
    cap.release()
    return frame
def get_qs_from_frame(frame,mlp,sg,cfg,plot=False):
    ## Alignment
    rect, face_landmarks = get_boundingbox_and_landmarks(frame, plot = False)
    aligned_img = image_align(frame, face_landmarks, save = False, output_size=sg.res)

    webcam_landmarks = extract_landmark(aligned_img, landmark_detector = cfg.landmark_type)

    webcam_landmarks_frame = extract_landmark(frame, landmark_detector = cfg.landmark_type)


    web_X = (torch.tensor(webcam_landmarks).T/sg.res).to(sg.device)
    q_web, losses = mlp.r1m.backwards(web_X, lr = 0.001, num_iters = 2000)
    l_hat = (mlp.r1m.r1mforward(q_web)*sg.res).detach().cpu()

    # plt.figure(figsize=(5,5))
    if plot:
        plt.figure(figsize=(40,10))
        plt.subplot(141)
        plt.imshow(frame)

        plt.axis("off")

        plt.subplot(142)
        plt.imshow(aligned_img)
        plt.axis("off")

        plt.subplot(143)
        plt.imshow(aligned_img)
        plt.scatter(*(webcam_landmarks.T))
        plot_dlib_lines(webcam_landmarks)
        plt.axis("off")


        plt.subplot(144)
        plt.imshow(aligned_img)
        plt.scatter(*(webcam_landmarks.T), label =  cfg.landmark_type)
        plt.scatter(*(l_hat), label = "r1m factorization" )
        plt.axis("off")
        plt.legend()
        plt.tight_layout()
        plt.show()
    return q_web, aligned_img


def add_margin(pil_img, top = 2, right = 2, bottom = 2, 
                    left = 2, color = (255,255,255)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    
    result.paste(pil_img, (left, top))
    return result


def tensor_to_pil(tensor_imgs):
    if type(tensor_imgs) == list:
        tensor_imgs = torch.cat(tensor_imgs)
    tensor_imgs = (tensor_imgs / 2 + 0.5).clamp(0, 1)
    to_pil = T.ToPILImage()
    pil_imgs = [to_pil(img) for img in tensor_imgs]    
    return pil_imgs

def image_grid(imgs, rows = 1, cols = None, 
                    size = None,
                   titles = None, 
                   top=20,
                   font_size = 20, 
                   text_pos = (0, 0), add_margin_size = None):
    if type(imgs) == list and type(imgs[0]) == torch.Tensor:
        imgs = torch.cat(imgs)
    if type(imgs) == torch.Tensor:
        imgs = tensor_to_pil(imgs)
        
    if not size is None:
        imgs = [img.resize((size,size)) for img in imgs]
    if cols is None:
        cols = len(imgs)
    assert len(imgs) >= rows*cols
    if not add_margin_size is None:
        imgs = [add_margin(img, top = add_margin_size,
                                right = add_margin_size,
                                bottom = add_margin_size, 
                                left = add_margin_size) for img in imgs]
        
    w, h = imgs[0].size
    delta = 0
    if len(imgs)> 1 and not imgs[1].size[1] == h:
        delta = h - imgs[1].size[1] #top
        h = imgs[1].size[1]
    if not titles is  None:
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 
                                    size = font_size, encoding="unic")
        h = top + h 
    grid = Image.new('RGB', size=(cols*w, rows*h+delta))    
    for i, img in enumerate(imgs):
        
        if not titles is  None:
            img = add_margin(img, top = top, bottom = 0,left=0)
            draw = ImageDraw.Draw(img)
            draw.text(text_pos, titles[i],(0,0,0), 
            font = font)
        if not delta == 0 and i > 0:
           grid.paste(img, box=(i%cols*w, i//cols*h+delta))
        else:
            grid.paste(img, box=(i%cols*w, i//cols*h))
        
    return grid    
