import torch 
import numpy as np
import matplotlib.pyplot as plt

import easydict
from tqdm import tqdm 

from matplotlib import pyplot as plt 
from criteria.lpips.lpips import LPIPS
from criteria.id_loss import IDLoss

from notebookvis import vis_basis_shapes
from notebookvis import start_widget

def edit_q(q, edit):
    q_ = q.clone()
    i, s = edit
    q_[i]+=s
    return q_


class RankOneEditor:
    def __init__(self, mlp, sg, cfg, device = None):
         
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        #else: self.device = device
        
        self.lpips = None
        
        self.arcface = IDLoss().to(self.device)
        self.mlp = mlp
        self.sg = sg
        self.cfg = cfg

    def edit(self, w0, edit = None, q_target = None, method = "baseline",
             plot_latent = False, plot_scatter = False, **kwargs):
        assert not (edit is None and q_target is None)
        if q_target is None:
            idx, strength  = edit
            q_target = self.mlp(w0).squeeze()
            q_target[idx] += strength
        if method == "lin":
            w_edit = self.lin_edit(w0, q_target, **kwargs)
        if method == "alt":
            w_edit = self.alt_edit(w0, q_target)
        if method == "baseline":
            w_edit = self.lin_edit_baseline(w0,idx,strength, numsteps=1)  
        if method == "grad":
            w_edit = self.gradedit(w0, q_target,
                                   **kwargs
                                )   
        if plot_latent:   
            self.sg.show(w_edit)
        if plot_scatter:
            q = self.mlp(w_edit).squeeze()
            L = self.mlp.r1m.r1mforward(q).detach().cpu()
            plt.scatter(*(L*self.sg.res), label = "prediction")
            L = self.mlp.r1m.r1mforward(q_target).detach().cpu()
            plt.scatter(*(L*self.sg.res), label = "taget")
            plt.legend()

            if not plot_latent:
                plt.axis("off")
                plt.tight_layout()
                _, _, h, w = self.sg.synthesize(w_edit).shape
                plt.ylim((0,h))
                plt.xlim((0,w))
                plt.gca().invert_yaxis()

        return w_edit


    def lin_edit_baseline(self, w0,i,a, num_iters=1, **kwargs):
        w_edit = w0.clone()
        for _ in range(num_iters): 
            J = torch.autograd.functional.jacobian(self.mlp, w_edit).squeeze()
            J_inv = torch.pinverse(J)
            w_edit = w_edit + a*J_inv[:,int(i)]/num_iters
        return w_edit

    def lin_edit(self, w0, q_target, num_iters=1, **kwargs):
        w_edit = w0.clone()
        for _ in range(num_iters): 
            q_hat = self.mlp(w_edit).squeeze()
            J = torch.autograd.functional.jacobian(self.mlp, w_edit).squeeze()
            J_inv = torch.pinverse(J)
            w_edit = w_edit + J_inv @ (q_target - q_hat) / num_iters
        return w_edit

    def alt_edit(self, w0, q_target, num_iters = 100):
        w_edit = w0.clone()
        for _ in range(num_iters):
            q_hat = self.mlp(w_edit).squeeze()
            J = torch.autograd.functional.jacobian(self.mlp, w_edit).squeeze()
     
            w_edit = w_edit + (q_target - q_hat) @ J  
        return w_edit


    def gradedit(self, w0, q_target,
                    lr = 1,
                    prog_bar = True,
                    num_iters = 100, 
                    q_lambda = 0,
                    l_lambda = 1,
                    lpips_lambda = 0,
                    arcface_lambda = 0,
                    optimizer_type = torch.optim.SGD
                    ):
        if not lpips_lambda == 0:
            self.lpips = LPIPS(net_type = "vgg").to(self.device)
        
        self.history = easydict.EasyDict({"qloss": [], "lloss": [], 
                                          "lpips": [], "arcface": []})

        w1 = w0.detach().clone()
        w1.requires_grad = True 
        img0 = self.sg.synthesize(w0)
        L_target =  self.mlp.r1m.r1mforward(q_target)

        def loss_fn():
            q_hat = self.mlp(w1)[0]
            L_hat = self.mlp.r1m.r1mforward(q_hat)
            
            if l_lambda:
                lloss = l_lambda * torch.norm(L_hat - L_target) 
                self.history.lloss.append(lloss.detach().cpu())
                loss = lloss

            if q_lambda:
                qloss = q_lambda * torch.norm(q_hat - q_target) 
                self.history.qloss.append(qloss.detach().cpu())
                loss += qloss


            ### Regularization with LPIPS and/or Arcface    
            if lpips_lambda or arcface_lambda:
                cur_img = self.sg.synthesize(w1)

            if lpips_lambda:
                lpips_loss = lpips_lambda * self.lpips(cur_img, img0)
                self.history.lpips.append(lpips_loss.detach().cpu())
                loss += lpips_loss

            if arcface_lambda:
                arcface_loss = arcface_lambda * self.arcface(cur_img, img0)
                self.history.arcface.append(arcface_loss.detach().cpu())
                loss += arcface_loss
            
            return loss
        

        optimizer = optimizer_type([w1], lr=lr)
        op = range(num_iters)
        if prog_bar: op = tqdm(op) 
        for i in op:
            optimizer.zero_grad()    
            loss = loss_fn()
            loss.backward(retain_graph=True)
            optimizer.step()
        return w1.detach()


    def notebook_vis_basis_shapes(self):
        vis_basis_shapes(self.mlp.r1m.B, self.cfg, D = None, figsize = (8, 4))

    def notebook_start_widget(self, w0 = None):
        start_widget(self,self.cfg, w0 = w0)

    def show_fac_results(self, results):
        for k in results.fac_history.keys():
            plt.plot(np.append(results.training_err,
                    np.array(results.fac_history[k])), label = k)
            plt.plot(results.training_err, label = "als")
        plt.legend() 
        plt.show()      








