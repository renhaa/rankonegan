import torch
from criteria.id_loss import IDLoss
from criteria.lpips.lpips import LPIPS
from functools import partial
from matplotlib import pyplot as plt
from torchvision import transforms as T
from tqdm import tqdm

from config import cfgs, create_default_config
from models.StyleGANWrapper import StyleGAN, show_torch_img

class GANmetric:
    def __init__(self, sg, metric = "lpips", 
                 ):

        self.device = sg.device
        self.metric = metric
        self.sg = sg
        self.set_metric_func(metric = metric)
        self.transform = T.Resize((256,256))
        # "cuda" if torch.cuda.is_available() else  "cpu"
    
    def set_metric_func(self, metric = "lpips"):    
        if metric == "lpips":
            self.metric_func =  LPIPS(net_type='vgg').to(self.device).eval()
        elif metric == "arcface":
            self.metric_func = IDLoss().to(self.device).eval()

    def distance_func(self,w, w0):
        img0 = self.transform(self.sg.synthesize(w0))
        img = self.transform(self.sg.synthesize(w))
        return  self.metric_func(img0, img) #**2 ## squared distance func

    def jacobian(self, w0):    
        dfunc = partial(self.distance_func, w0 = w0)
        H = torch.autograd.functional.jacobian(dfunc, w0)
        return H

    def hessian(self, w0):    
        dfunc = partial(self.distance_func, w0 = w0)
        H = torch.autograd.functional.hessian(dfunc, w0)
        return H

    def get_tranformation_matrix(self,num_samples = 1000):
        w_mean = self.sg.get_mean_latent(num_samples = num_samples)
        H = self.hessian(w_mean)
        U,s,Vh = torch.svd(H)
        return U, s

def load_transformation_matrix(results,cfg, 
                               force_rerun = False):
    print(["INFO Using LIPS hessian"])
    if "U" in results.keys() and not force_rerun:
        return results.U
    else:
        print("\n---------------------------------------------\n")
        print("Now calculating global Hessians for", cfg.sgmodel)
        latent_transforms = {}
 
        sg = StyleGAN(cfg.sg_path, transformation_matrix=None ,latentspace_type = cfg.latent_space)
        gan_metric = GANmetric(sg)
        U, s = gan_metric.get_tranformation_matrix(
                        num_samples = 1000,
                        )
        U = U.detach().cpu()
        latent_transforms[cfg.sgmodel+"-"+cfg.latent_space] = U
        results.U = U
        results.lpips_H_s = s
        
        torch.save(results,cfg.results_path)
        # torch.save(latent_transforms, out_path)
        print("Hessians saved to", cfg.results_path)
        print("\n---------------------------------------------\n")
        
        return U 
    