import os
import torch 
import easydict
import mediapipe as mp
from tqdm import tqdm 
import matplotlib.pyplot as plt

from models.StyleGANWrapper import StyleGAN
from face_utils import extract_landmark, plot_dlib_lines
from config import create_default_config, sgmodel_to_path

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def to_np_image(all_images):
    all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
    return all_images

def generate_data(sg, cfg, verbose = False):
    stylemix = "stylemix" in cfg.latent_space
    print("DEBUG dataloader l.23. Doing stylemix", stylemix)
    if stylemix:
        data = easydict.EasyDict(
                {"wps": torch.zeros((cfg.num_samples,sg.G.num_ws*512)), 
                "dlib": torch.zeros(cfg.num_samples,2,68),
                "mediapipe": torch.zeros(cfg.num_samples,2,468)})
    else:
        data = easydict.EasyDict(
               {"zs": torch.zeros((cfg.num_samples,512)), 
                "ws": torch.zeros((cfg.num_samples,512)), 
                "dlib": torch.zeros(cfg.num_samples,2,68),
                "mediapipe": torch.zeros(cfg.num_samples,2,468)})

    for i in tqdm(range(cfg.num_samples)):
        landmark_dlib, landmark_mediapipe = None, None
        while landmark_dlib is None or landmark_mediapipe is None:
            if stylemix:
                            #z = torch.randn([1, G.z_dim]).to(device=device)  # latent codes
                w = torch.cat([sg.G.mapping(torch.randn([1, sg.G.z_dim]).to(device=sg.device), 
                                None, truncation_psi=0.8, 
                                truncation_cutoff=8)[0,0,:].flatten().unsqueeze(0)
                                for _ in range(sg.G.num_ws)]).unsqueeze(0)
                img = sg.G.synthesis(w, noise_mode='const', force_fp32=True)
                img = to_np_image(img)
            else:
                z = sg.sample()  
                img = sg.synthesize(z, to_np=True)
            landmark_dlib = extract_landmark(img, landmark_detector = "dlib")
            if landmark_dlib is None and verbose: print("dlib landmark not found")
            landmark_mediapipe = extract_landmark(img, landmark_detector = "mediapipe")
            if landmark_mediapipe is None and verbose: print("mediapipe landmark not found")
        
        if stylemix:
            data.wps[i] = w.flatten().detach().cpu() 
        else:
            data.zs[i] = z.detach().cpu()
            data.ws[i] = sg.z_to_w(z,to_wp=False).detach().cpu()
        data.dlib[i] = torch.tensor(landmark_dlib.T/sg.res) 
        data.mediapipe[i] = torch.tensor(landmark_mediapipe.T/sg.res)  
    return data

def generate_data_humans(sg,cfg, verbose = False):
    stylemix = "stylemix" in cfg.latent_space
    if stylemix:
        data = easydict.EasyDict(
                {"wps": torch.zeros((cfg.num_samples,sg.G.num_ws*512)), 
                "mediapipe": torch.zeros(cfg.num_samples,2,33)})
    else:
        data = easydict.EasyDict(
                 {"zs": torch.zeros((cfg.num_samples,512)),
                "ws": torch.zeros((cfg.num_samples,512)), 
                "mediapipe": torch.zeros(cfg.num_samples,2,33)})
    for i in tqdm(range(cfg.num_samples)):
        X = None
        while X is None:
            if stylemix:
                            #z = torch.randn([1, G.z_dim]).to(device=device)  # latent codes
                w = torch.cat([sg.G.mapping(torch.randn([1, sg.G.z_dim]).to(device=sg.device), 
                                None, truncation_psi=0.8, 
                                truncation_cutoff=8)[0,0,:].flatten().unsqueeze(0)
                                for _ in range(sg.G.num_ws)]).unsqueeze(0)
                img = sg.G.synthesis(w, noise_mode='const', force_fp32=True)
                img = to_np_image(img)
            else:
                z = sg.sample()  
                img = sg.synthesize(z, to_np=True)
            X = detect_pose(img)
        if stylemix:
            data.wps[i] = w.flatten().detach().cpu() 
        else:
            data.zs[i] = z.detach().cpu()
            data.ws[i] = sg.z_to_w(z,to_wp=False).detach().cpu()
        data.mediapipe[i] = (X/sg.res).detach().cpu()  
    return data

def detect_pose(img):
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.2) as pose:
        image_height, image_width, _ = img.shape

        results = pose.process(img)
        if results.pose_landmarks is None:
            return None 
        X = torch.zeros((2,33))
        for i, l in  enumerate(results.pose_landmarks.landmark):
            X[:,i] = torch.tensor([l.x*image_width,l.y*image_height])
    return X

def annotate_pose_man(X):
    cons = [j for j in mp_pose.POSE_CONNECTIONS]
    con = cons[0]
    for con in cons:
        c = torch.cat([X[:,con[0]].unsqueeze(0),
                    X[:,con[1]].unsqueeze(0)])
        plt.plot(*c.T)

def load_data(cfg, force_rerun = False):
    
    sg = StyleGAN(cfg.sg_path, latentspace_type = "z",
            is_third_time_repo = cfg.is_third_time_repo)

    ## Generate data if needed
    if os.path.exists(cfg.data_path) and not force_rerun:
        data = torch.load(cfg.data_path)
        print("Loaded data from:", cfg.data_path)
    else: 
        if "humans" in cfg.sgmodel:
            data  = generate_data_humans(sg, cfg, verbose = False)
        else:
            data  = generate_data(sg, cfg, verbose = False)
        torch.save(data, cfg.data_path)
        print("Saved data to:", cfg.data_path)

    ## Select appropriate latents and landmarks
    landmarks = data[cfg.landmark_type]

    if "stylemix" in cfg.latent_space:
        latents = data.wps
    elif cfg.latent_space == "wp":
        latents = data.ws.repeat([1, sg.G.num_ws])

    elif cfg.latent_space == "s":
        if "ss" in data.keys():
            latents = data.ss
        else:
            print("Generating stylespace latents []")
            latents = data.ws.repeat([1, sg.G.num_ws])
            latents = torch.cat([sg.w2s(w.to(sg.device)).detach().cpu().unsqueeze(0) for w in tqdm(latents)],axis = 0)
            data.ss = latents
            torch.save(data, cfg.data_path)
    else:
        latents = data[cfg.latent_space + "s"]

    return latents, landmarks

