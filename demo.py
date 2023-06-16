import torch
from matplotlib import pyplot as plt

from PIL import Image, ImageDraw
import gradio as gr

from utils import tensor_to_pil, image_grid
from config import create_default_config
from main import load_experiment

def update(seed, yaw, pitch, b1,b2,method,idreg):
  
    w = r1gan.sg.sample(seed = seed)

    q_hat = r1gan.mlp(w).squeeze()
    q_target = q_hat.clone()

    img = r1gan.sg.synthesize(w)
    pil_img = tensor_to_pil(img)[0]

    q_target[4] += yaw
    q_target[3] += pitch
    q_target[6] += b1
    q_target[7] += b2

    if idreg:
        arcface = 0.12
    else: 
        arcface = 0
  
    if method == "Linear":
        method_ = "lin"
    elif  method == "Gradient":
        method_ = "grad"
    else:
        raise NotImplementedError

    w_edit = r1gan.edit(w, q_target = q_target, method=method_, 
                        arcface_lambda = arcface,
                        lr=0.7, num_iters = 30)
    img_edit = r1gan.sg.synthesize(w_edit)
    pil_img_edit = tensor_to_pil(img_edit)[0]

    pil_img_edit_annotated = pil_img_edit.copy()
   
    draw = ImageDraw.Draw(pil_img_edit_annotated)
    
    # target_shape = r1gan.mlp.r1m.r1mforward(q_target).detach().cpu()*r1gan.sg.res
    q_hat_edit = r1gan.mlp(w_edit).squeeze()
    predicted_shape = r1gan.mlp.r1m.r1mforward(q_hat_edit).detach().cpu()*r1gan.sg.res

    size = 5
    for point in predicted_shape.T:
        x, y = point
        draw.ellipse([x-size/2,y-size/2,x+size//2,y+size//2], fill="blue")
    edit_img_pil = image_grid([pil_img_edit, pil_img_edit_annotated], add_margin_size=5)

    return [pil_img, edit_img_pil]

if __name__ == "__main__":

    inputs = [
        gr.Slider(minimum=1, maximum=100, value=42, step=1, label="seed"),
        gr.Slider(minimum=-1.0, maximum=1.0, value=0.0, step=0.1, label="yaw"),
        gr.Slider(minimum=-1.0, maximum=1.0, value=0.0, step=0.1, label="pitch"),
        gr.Slider(minimum=-0.6, maximum=0.6, value=0.0, step=0.05, label="Basisshape 1"),
        gr.Slider(minimum=-0.6, maximum=0.6, value=0.0, step=0.05, label="Basisshape 1"),
        gr.Dropdown(choices=["Linear","Gradient"],value="Gradient", label="Method"),
        gr.Checkbox(label="Identity Regularization")
    ]

    cfg = create_default_config(latent_space = "w",
                                num_samples = 50000,
                                landmark_type = "mediapipe",
                                sgmodel = "sg2-256-ffhq")

    r1gan, results, data = load_experiment(cfg, redo_r1mfit=False,
                                                retrain_model = False)

    outputs = [gr.Image(label="Input Image", type="pil"),
               gr.Image(label="Edit Result", type="pil")]


    interface = gr.Interface(fn=update, inputs=inputs, outputs=outputs)

    interface.launch(share = True)