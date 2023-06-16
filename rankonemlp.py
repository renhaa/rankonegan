import easydict
import functorch

from tqdm import tqdm 

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from factorization import get_q_stats

class DeepLearnBase(torch.nn.Module):

    def __init__(self, device = None):
        super().__init__()
        
        self.criterion = nn.MSELoss()
        self.loss_hist = easydict.EasyDict({"train_qloss": [],"valid_qloss": [],
                                            "train_lloss": [],"valid_lloss": []})
       
        if device == None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = None

    def training_loss(self, batch):
        self.train()
        x, y, l = batch
        x, y, l = x.to(self.device), y.to(self.device), l.to(self.device)
        
        y_hat = self.forward(x)
        l_hat = self.r1m.batched_r1mforward(y_hat).flatten(1)
        lloss  = self.criterion(l_hat, l)
        self.loss_hist.train_lloss.append(lloss.detach().cpu().numpy())

        return  lloss

    def validation_loss(self, valid_data):
        self.eval()
        with torch.no_grad():
            x, y, l = valid_data.tensors
            x, y, l = x.to(self.device), y.to(self.device), l.to(self.device)
            y_hat = self.forward(x)
            l_hat = self.r1m.batched_r1mforward(y_hat).flatten(1)
            lloss  = self.criterion(l_hat, l)
            self.loss_hist.valid_lloss.append(lloss.detach().cpu().numpy())

        return lloss

    def fit(self, latents, 
                  landmarks,
                  lr = 0.01, 
                  weight_decay = 0.1, 
                  max_iters = 50, 
                  batch_size = 1024, 
                  optim_strategy ="adam",
                  **kwargs):
        
        print("lr", lr,"batch",batch_size, "weight_decay", weight_decay)
  
        
        X_train, X_test, y_train, y_test, l_train, l_test = train_test_split(latents, torch.zeros_like(landmarks.flatten(1)), landmarks.flatten(1), test_size=0.2, random_state=42)
        train_data = TensorDataset(X_train, y_train, l_train)
        train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 8)
        val_data = TensorDataset(X_test, y_test, l_test)
        
        print(len(y_train), "Training points", len(y_test), "Test points")
    
        self.optim_strategy = optim_strategy
        if self.optim_strategy == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                            lr = lr, 
                                            weight_decay = weight_decay)
        elif self.optim_strategy == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                            lr = lr, 
                                            weight_decay = weight_decay)


        ## Train Loop
        
        for _ in tqdm(range(max_iters)):
            
            for batch in train_loader:
                self.optimizer.zero_grad() 
                
                loss = self.training_loss(batch)
                ## Update step
                loss.backward(retain_graph=True)
                self.optimizer.step()
                ## Log Validation loss and acc
                self.validation_loss(val_data)
                
    def print_nr_params(self):
        print("Nr of parameters", sum(p.numel() for p in self.parameters()))

    def plot_loss(self):
        for k in self.loss_hist:
            if "lloss" in k:
                loss = self.loss_hist[k]
                plt.plot(loss,label = k)
                print("Final", k, loss[-1])
        plt.legend()
                
class RankOneMLP(DeepLearnBase):

    def __init__(self, w_dim, q_dim, 
                r1m,num_layers = 2, 
                layer_width = 512, 
                stats = None, 
                device = None,
                normalize_w_std = False,
                activation = "relu"):

        if activation == "relu":
            activ = nn.ReLU
        elif activation == "elu":
            activ = nn.ELU
        super().__init__(device = device)
        layers = [nn.Linear(w_dim, layer_width), activ()]
        for _ in range(num_layers):
            layers += [nn.Linear(layer_width, layer_width),
                 
                        activ()]
        layers += [nn.Linear(layer_width, q_dim)]
        self.device = device
        self.normalize_w_std = normalize_w_std
        self.model = nn.Sequential(*layers)
        self.model.to(self.device)
        self.print_nr_params()
        self.r1m = r1m
        self.qsize = q_dim

        ## Feature output normalization
        if stats is None:
            self.stats = easydict.EasyDict({
                "mean": torch.zeros(q_dim).unsqueeze(0).to(self.device),
                "std": torch.ones(q_dim).unsqueeze(0).to(self.device),
            })
        else: 
            self.stats = easydict.EasyDict()
            self.stats.mean = stats.mean.clone().to(self.device).unsqueeze(0)
            self.stats.std = stats.std.clone().to(self.device).unsqueeze(0)
    
    def forward(self, x):
        q = self.model(x.to(self.device))
        if self.normalize_w_std:
            return q*self.stats.std + self.stats.mean
        else:
            return q + self.stats.mean

    def full_forward(self, x):
        return self.r1m.batched_r1mforward(self.forward(x))
        
def euler_to_rotmat(theta):
    device = theta.device
    def Rx(t):
        return torch.cat([
            torch.cat([torch.ones(1,device = device) , torch.zeros(1,device = device), torch.zeros(1,device = device)], axis = 0).unsqueeze(0),
            torch.cat([torch.zeros(1,device = device), torch.cos(t)  ,-torch.sin(t)], axis = 0).unsqueeze(0),
            torch.cat([torch.zeros(1,device = device), torch.sin(t)  , torch.cos(t)], axis = 0).unsqueeze(0)], axis = 0)

    def Ry(t):
        return torch.cat([
            torch.cat([torch.cos(t)  , torch.zeros(1,device = device) , torch.sin(t)], axis = 0).unsqueeze(0),
            torch.cat([torch.zeros(1,device = device), torch.ones(1,device = device)  ,torch.zeros(1,device = device)], axis = 0).unsqueeze(0),
            torch.cat([-torch.sin(t) , torch.zeros(1,device = device) , torch.cos(t)], axis = 0).unsqueeze(0)], axis = 0)

    def Rz(t):
        return torch.cat([
            torch.cat([torch.cos(t) , -torch.sin(t), torch.zeros(1,device = device), ], axis = 0).unsqueeze(0),
            torch.cat([torch.sin(t), torch.cos(t)  ,torch.zeros(1,device = device)], axis = 0).unsqueeze(0),
            torch.cat([torch.zeros(1,device = device), torch.zeros(1,device = device) , torch.ones(1,device = device)], axis = 0).unsqueeze(0)],axis = 0)

    return  Rz(theta[2].unsqueeze(0)) @ Ry(theta[1].unsqueeze(0)) @ Rx(theta[0].unsqueeze(0)) 
    
class RankOneModel:
    def __init__(self, B, opt = None, D = None, device = None):

        if device is None:
            self.device = "cuda"
        else:
            self.device = device
        if opt is None:
            self.opt = easydict.EasyDict()
            K = B.shape[0]-1
            self.opt.parameter_lengths = [3,3,K,2]
            self.opt.parameter_names = ["k","rotvec","alpha","mean"]
        if D is None:
            D = torch.eye(3).to(self.device)

        self.B = B.to(self.device)
        self.D = D.to(self.device) # calibration
        self.L = B.shape[-1]
        self.batched_r1mforward = functorch.vmap(self.r1mforward, in_dims=0, out_dims=0, randomness='error')
        self.qsize = sum(self.opt.parameter_lengths)

    def r1mforward(self, q):
        "Return landmark representation from the q parameterization"
        q = q.to(self.device)
        # Split into camera, rotation and shape parameters
        p = easydict.EasyDict()
        curr_idx = 0
        for plen,pname in zip(self.opt.parameter_lengths,
                              self.opt.parameter_names):
            p[pname] = q[curr_idx:curr_idx+plen]
            curr_idx += plen
            
        # Rank one model equation 
        K = torch.cat([p.k[:2], torch.zeros((1)).to(self.device), p.k[-1:]],axis = 0).reshape(2,2)
        R = euler_to_rotmat(p.rotvec)
        I = torch.eye(3)[:2].to(self.device) 
        shape = self.B[0] + sum([a*self.B[i+1] for i,a in enumerate(p.alpha)])
        offset = p.mean.repeat((self.L)).reshape(self.L,2).T
        return K @ I @ R @ shape + offset
        
    def backwards(self, landmarks, lr = 0.001, num_iters = 1000):
        q = torch.zeros(sum(self.opt.parameter_lengths),requires_grad=True)

        optimizer = torch.optim.Adam([q],lr = lr)
        losses = []
        for _ in tqdm(range(num_iters)):
                optimizer.zero_grad() 
                
                loss = torch.norm(self.r1mforward(q) - landmarks)
                
                losses.append(loss.detach().cpu().numpy())
                loss.backward(retain_graph=True)
                optimizer.step()
        return q, losses


def train_or_load_mlp(latents, landmarks, results, cfg, force_rerun = False):
        
    B = results.B
    if cfg.factorization.do_pca_transform:
        print("[INFO] Doing PCA linear combination of basis shapes")
        U, _, _ = torch.pca_lowrank(results.alphas.T, 
          q=cfg.factorization.K-1, 
          center=True, niter=5)
        B = torch.cat([B[0].unsqueeze(0)]+[sum([coeff*B[i+1] 
        for i,coeff in enumerate(u)]).unsqueeze(0) 
        for u in U.T])
    
    
    r1m = RankOneModel(B, D = results.D_cal, opt = None, device = None)

    dims = (latents.shape[1],r1m.qsize)

    if cfg.mlp.output_normalization:
        if cfg.factorization.do_pca_transform: 
            alphas_pca = results.alphas @ U
            _, stats = get_q_stats(results.M0_cal, alphas_pca, results.means)
        else: 
            stats = results.stats
    else: 
        stats = None

    mlp = RankOneMLP(dims[0],dims[1], r1m, 
                    num_layers = cfg.mlp.num_layers,
                    layer_width = cfg.mlp.layer_width,
                    stats=stats, device = "cuda")

    if not "mlp_ckpt" in results.keys() or force_rerun:
        mlp.fit(latents, landmarks, **cfg.mlp) 
        results.mlp_ckpt = mlp.state_dict()
        results.mlp_hist = mlp.loss_hist
        torch.save(results, cfg.results_path)
        print("saved to", cfg.results_path)
    else:
        mlp.load_state_dict(results.mlp_ckpt)
        mlp.loss_hist = results.mlp_hist

        mlp.stats = easydict.EasyDict()
        mlp.stats.mean = stats.mean.clone().to(mlp.device).unsqueeze(0)
        mlp.stats.std = stats.std.clone().to(mlp.device).unsqueeze(0)

    return mlp 