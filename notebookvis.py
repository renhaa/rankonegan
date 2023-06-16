
import torch 
import roma
import easydict
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from matplotlib import pyplot as plt
from criteria.lpips.lpips import LPIPS
from criteria.id_loss import IDLoss

# 2DO refactor these
from dataloader import annotate_pose_man
from face_utils import plot_dlib_lines

def vis_basis_shapes(B, cfg, D = None, figsize = (10, 4)):
    if D is None: D = torch.eye(3)
    D = D.to(B.device)
    
    def plot_basisshape(alpha_val, x,y,z):
        
        rotvec = torch.tensor([x,y,z]).to(B.device)
        
        R = roma.rotvec_to_rotmat(rotvec)
    
        plt.figure(figsize=figsize)
        for i in range(10):
            
            shape  =  (R @ D @  ( B[0] + alpha_val * B[i+1]))[:2].detach().cpu()
            plt.subplot(2,5,i+1)
            plt.scatter(*shape,s=5)
            if "humans" in cfg.sgmodel:
                annotate_pose_man(shape)
                plt.xlim((-0.8,0.8))
                plt.ylim((-0.4,0.4))
            elif "dlib" in cfg.landmark_type:
                plot_dlib_lines(shape.T.numpy())
            plt.gca().invert_yaxis()
            plt.axis("off")
            if False: 
                plt.title(f"#{i} (idx {i+6})")
        plt.show()

    interact(plot_basisshape, 
            alpha_val = widgets.FloatSlider(value=-0.1,
                                                min=-0.5,
                                                max=0.5,
                                                step=0.05), 
                    x = widgets.FloatSlider(value=0.2,
                                                min=-1.,
                                                max=1.,
                                                step=0.05),
                    y = widgets.FloatSlider(value=0.,
                                                min=-1.,
                                                max=1.,
                                                step=0.05),
                    z = widgets.FloatSlider(value=0.,
                                                min=-1.,
                                                max=1.,
                                                step=0.1)
    );
    plt.show()


def start_widget(r1gan,cfg, w0 = None):

    widget = easydict.EasyDict()

    widget.seed = widgets.FloatSlider(value=42,
                            min=0,
                            max=100,
                            step=1) 

    widget.idx_basis_shape = widgets.FloatSlider(value=0,
                            min=0,
                            max=cfg.factorization.K,
                            step=1) 
    widget.strength_basis_shape = widgets.FloatSlider(value=0,
                            min=-0.6,
                            max=0.6,
                            step=.05)

    widget.rot_x = widgets.FloatSlider(value=0,
                            min=-1,
                            max=1,
                            step=.05)

    widget.rot_y = widgets.FloatSlider(value=0,
                            min=-1,
                            max=1,
                            step=.05)

    widget.rot_z = widgets.FloatSlider(value=0,
                            min=-1,
                            max=1,
                            step=.05)
    
    widget.arcface = widgets.FloatSlider(value=0,
                            min=0,
                            max=2,
                            step=.05,
                            description='Identity regularization'
                            )
    widget.lr = widgets.FloatSlider(value=1,
                            min=0.1,
                            max=10,
                            step=.1)

    widget.num_iters = widgets.FloatSlider(value=30,
                            min=1,
                            max=1000,
                            step=1)
    widget.method = widgets.Dropdown(
        options=['lin', 'grad',"hijack"],
        value='grad',
        description='Method',
    )
    def vis(seed, idx_basis_shape, strength_basis_shape,
                rot_x, rot_y, rot_z, method, arcface,lr,num_iters):

        if w0 is None:
            w = r1gan.sg.sample(seed = seed)
        else: 
            w = w0
        # edit q
                # edit q
        q_hat = r1gan.mlp(w).squeeze()
        q_target = q_hat.clone()
        q_target[3] += rot_x
        q_target[4] += rot_y
        q_target[5] += rot_z
        q_target[6 + int(idx_basis_shape)] += strength_basis_shape
        
        # Edited latent and landmarks
        w_edit = r1gan.edit(w, q_target = q_target, method=method, 
                            arcface_lambda = arcface,
                            lr=lr, num_iters = int(num_iters))
        target_shape = r1gan.mlp.r1m.r1mforward(q_target).detach().cpu()*r1gan.sg.res

        q_hat_edit = r1gan.mlp(w_edit).squeeze()

        predicted_shape = r1gan.mlp.r1m.r1mforward(q_hat_edit).detach().cpu()*r1gan.sg.res

        
        #Make figure
        height = 5
        if "humans" in cfg.sgmodel:
            height = 10
        

        plt.figure(figsize=(15,height))
        plt.subplot(1,3,1)
        r1gan.sg.show(w)
        plt.subplot(1,3,2)
        r1gan.sg.show(w_edit)

        plt.subplot(1,3,3)
        r1gan.sg.show(w_edit)
        plt.scatter(*target_shape, label = "target")
        if "humans" in cfg.sgmodel:
            annotate_pose_man(target_shape)
        elif "dlib" in cfg.landmark_type:
            plot_dlib_lines(target_shape.T.numpy())

        plt.scatter(*predicted_shape, label = "predicted")

        plt.legend()

    interact(vis,**widget)