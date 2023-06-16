import os
from gan_manifold import load_transformation_matrix
import gdown
import wget

from models.StyleGANWrapper import StyleGAN
from rankonemlp import  train_or_load_mlp
from dataloader import load_data
from manipulator import RankOneEditor
from factorization import fit_r1m_model


output_directory = "pretrained_models/"


def load_experiment(cfg, redo_data = False, redo_r1mfit = False, retrain_model = False, redohessian =False):

    
    assure_sgmodel(cfg.sg_path)
    assure_arcface()
    latents, landmarks = load_data(cfg, force_rerun = redo_data)
    


    results = fit_r1m_model(landmarks, cfg,
                                    plot = False, 
                                    force_rerun = redo_r1mfit,
                                    verbose = False)
  
  
    if cfg.use_lpips_transform:
        U = load_transformation_matrix(results,cfg,force_rerun=redohessian)
        U = U[:,:cfg.num_lpips_hession_eigvecs]
        latents = latents @ U.to(latents.device)    
    else:
        U = None

    mlp = train_or_load_mlp(latents, landmarks, results, cfg,
                            force_rerun = retrain_model)

  
    ls = "wp" if "wp" in cfg.latent_space else cfg.latent_space  
    sg = StyleGAN(cfg.sg_path,transformation_matrix=U, latentspace_type = ls)
    r1gan = RankOneEditor(mlp, sg, cfg)

    return r1gan, results, (latents, landmarks)


def assure_arcface(path = "pretrained_models/model_ir_se50.pth"):
    if not  os.path.exists(path):
        id = "1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn"
        gdown.download(id = id, output = path, quiet=False, fuzzy=True)

def assure_sgmodel(path):
 
    if os.path.exists(path):
        print("Loading", path)
        return 

    sg2_256_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl"
    sg2_1024_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl"
    sg3_256_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-256x256.pkl"
    sg3_1024_url ="https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl"


    model = path.split("/")[1]

    url = { 
        "stylegan2-ffhq-256x256.pkl":sg2_256_url,
        "stylegan2-ffhq-1024x1024.pkl":sg2_1024_url,
        "stylegan3-r-ffhqu-256x256.pkl": sg3_256_url,
        "stylegan3-r-ffhq-1024x1024.pkl":sg3_1024_url,
        }[model]
    print("Downloading", model)
    wget.download(url, out=output_directory)

