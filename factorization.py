import os
import torch 
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm 
import easydict
import roma # RoMa: A lightweight library to deal with 3D rotations in PyTorch.
import functorch
import itertools


def normalize_landmark(landmark):  # [2, L ]
    mean = landmark.mean(axis = 1) # mean 2d point
    mean_centered_landmark = (landmark.T - mean).T 
    ## We do no scaling since this is absorbed into camera parameters.
    scale = 1
    normalized_landmark = mean_centered_landmark / scale
    return mean, scale, normalized_landmark 

def preprocess(landmarks):
    means = []
    scalings = []
    normalized_landmarks = []
    for landmark in landmarks:
        mean, scale, normalized_landmark  =  normalize_landmark(landmark)
        means.append(mean)
        scalings.append(scale)
        normalized_landmarks.append(normalized_landmark)
    normalized_landmarks = torch.cat([l.unsqueeze(0) for l in normalized_landmarks], axis = 0)        
    return normalized_landmarks, means, scalings

def nonrigid_factorization(W):
    """
    Given measurement matrix W in [2N,L]
    
    returns the factorization W = M0B0 + dMdB
    """
    # Rigid part
    U,S0,V = torch.linalg.svd(W, full_matrices=False) #
    M0 = U[:,:3] @ torch.diag(S0[:3])         
    B0 = V[:3,:]

    # Non rigid part
    U,dS, dB = torch.linalg.svd(W - M0@B0, full_matrices=False) #
    dM = U @ torch.diag(dS)
    # dB = dB[:K,:]
    # dM = dM[:,:K] 
    return M0, B0, S0, dM, dB, dS

def recover_rankone_parameters_als(M0, B0,dB, W,num_iters_als = 3, K = 12,**kwargs):

    print(f"[INFO] Now doing ALS algorithm for {num_iters_als} iters.\n ")
    def l2loss_fn(alphas,D):
        """
        Calculates  ||W_hat - W||^2 given alpha and D.
        """
        M = torch.mul(torch.kron(alphas,torch.ones(2,3)),torch.kron(torch.ones(K), M0))
        B = torch.diag(D.flatten()) @ torch.kron(torch.eye(K), torch.ones(3,1)) @ dB
        
        W_hat = M0 @ B0 + M @ B   
        return torch.norm(W_hat - W)
    
    N = int(M0.shape[0]/2)


    ## Initialize variables
    dB = dB[:K,:]
  

    D = torch.ones((K,3)) / torch.sqrt(torch.tensor([3], dtype = torch.float32)) 
    alphas = torch.randn((N,K))
    dW = W - M0@B0 # dM @ dB
    dW_is = [dW[2*i:2*i+2,:] for i in range(N)] 
    M_is = [M0[2*i:2*i+2,:] for i in range(N)]

    training_err = []
    for _ in tqdm(range(num_iters_als)):
        ## Update alpha
        for i,k in itertools.product(range(N),range(K)):
            B_ik = M_is[i] @ torch.outer( D[k],  dB[k])
            alphas[i,k] = ( dW_is[i].flatten() @ B_ik.flatten() )  / ( B_ik.flatten() @ B_ik.flatten() ) 
        training_err.append( l2loss_fn(alphas,D).detach().cpu().numpy())
       
        ## Update D
        for k in range(K):
            aM = torch.zeros((2* N,3))#.type( dtype)
            for i in range( N): 
                aM[2*i:2*i+2,:] =  alphas[i,k]* M_is[i]
            D[k] = torch.pinverse(aM) @ dW @  dB[k]
            D[k] =  D[k,:]/np.sqrt(( D[k,:] @  D[k,:]))

        training_err.append( l2loss_fn(alphas,D).detach().cpu().numpy())
            

        del aM
        del B_ik
    
    basis_shapes = [torch.outer(D[k], dB[k]) for k in range(K)]
    B = torch.cat([B0.unsqueeze(0), torch.cat([b.unsqueeze(0) for b in basis_shapes],axis = 0)])

    return B, alphas, D, np.array(training_err)

def recover_rankone_parameters_grad(M0, B0,dB, W,
                                alphas = None,
                                D = None,
                                optim = None,
                                lr = 0.01,                                     
                                num_iters_grad = 10,
                                prog_bar = True,
                                reg_strength = 5,
                                K=12,  **kwargs):
    print(f"[Info] Now doing grad for {num_iters_grad} iters")
    if optim is None: optim = torch.optim.Adam

    # nr datapoints and basisshapes
    dB = dB[:K,:]
    #dM = dM[:,:K] 
    N = int(M0.shape[0]/2)
    K = dB.shape[0]

    history = easydict.EasyDict({"l2_hist":[], "reg_hist":[]})

    def l2loss_fn(alphas,D,M0, B0,dB, W): # alphas,D,M0, B0,dB, W
        """
        Calculates  ||W_hat - W||^2 given alpha and D.
        """
        M = torch.mul(torch.kron(alphas,torch.ones(2,3)),torch.kron(torch.ones(K), M0))
        B = torch.diag(D.flatten()) @ torch.kron(torch.eye(K), torch.ones(3,1)) @ dB
        W_hat = M0 @ B0 + M @ B   
        return torch.norm(W_hat - W)

    def loss_fn(alphas,D,M0, B0,dB, W):
        l2loss = l2loss_fn(alphas,D,M0, B0,dB, W) #alphas,D,M0, B0,dB, W
        history.l2_hist.append(l2loss.detach().cpu().numpy())

        reg = reg_strength*sum([(torch.sqrt(d@d) - 1)**2 for d in D])
        history.reg_hist.append(reg.detach().cpu().numpy())

        loss = l2loss + reg 
        return loss

    # Init parameters
    if D is None: D = torch.ones((K,3)) / torch.sqrt(torch.tensor([3], dtype = torch.float32)).clone()
    if alphas is None: alphas = torch.randn((N,K)).clone()

    ## Main optim loop
    D.requires_grad = True
    alphas.requires_grad = True
    optimizer = optim([D, alphas], lr=lr)
    op = range(num_iters_grad)
    if prog_bar: op = tqdm(op) 
    for i in op:
        optimizer.zero_grad()    
        loss = loss_fn(alphas,D,M0, B0,dB, W)
        loss.backward(retain_graph=True)
        optimizer.step()
    D.requires_grad = False
    alphas.requires_grad = False

    #Define basis shapes
    basis_shapes = [torch.outer(D[k], dB[k]) for k in range(K)]
    B = torch.cat([B0.unsqueeze(0), torch.cat([b.unsqueeze(0) for b in basis_shapes],axis = 0)])
    
    
    history.l2_hist = np.array(history.l2_hist)
    history.reg_hist = np.array(history.reg_hist)


    return B, alphas, D, history

def autocalibrate(Ms):
    import scipy
    from scipy import optimize

    def selfcal(z):
        ## Para-perspective camera assumption
        F = []
        Z = torch.Tensor(
            [[z[0], 0, 0],
            [z[1], z[2], 0],
            [z[3], z[4], 1]])
        X = Z @ Z.T
        #print("DEBUG rankonemodel l 459,", Ms[0][0].shape, "should be matrix")
        for i in range((len(Ms) - 1)):
            f1 = ((Ms[i][0] @ X @ Ms[i][0].T)  / (Ms[i][1] @ X @ Ms[i][1].T)  
                - (Ms[i+1][0] @ X @ Ms[i+1][0].T)  / (Ms[i+1][1] @ X @ Ms[i+1][1].T))**2
            f2 = ((Ms[i][0] @ X @ Ms[i][1].T)  / (Ms[i][1] @ X @ Ms[i][1].T)  
                - (Ms[i+1][0] @ X @ Ms[i+1][1].T)  / (Ms[i+1][1] @ X @ Ms[i+1][1].T))**2  
            F.append(f1)
            F.append(f2)
        F = torch.Tensor(F)
        return F
        
    z0 = [1,0,1,0,0] # what is z0 ?
    P = [1, 0, 0,
        0, 1, 0]
    F = selfcal(z0)

    results = scipy.optimize.least_squares(selfcal, z0, method='lm')

    z = results["x"]

    Z = torch.Tensor(
        [[z[0], 0, 0],
        [z[1], z[2], 0],
        [z[3], z[4], 1]])
    reference_view = Ms[0] 
    Q, R = torch.linalg.qr((reference_view @ Z).T, mode = "complete")
    D = Z @ Q
    return D

def do_calibration(M0,B, cfg, D_cal = None):
    print(f"\n[INFO] doing calibration: {cfg.calibration}")

    N = int(M0.shape[0]/2)
    M_is = [M0[2*i:2*i+2,:] for i in range(N)]
    if D_cal is None:
        if cfg.calibration == "quan":
            print("Doing Quan Calibration")
            D_cal = autocalibrate(M_is)
            print("Calibration  Done")
        elif cfg.calibration == "x45":
            theta = torch.tensor([np.pi/4,0.,0])
            D_cal = euler_to_rotmat(theta)
        elif cfg.calibration == "z45":
            print("rotating pi/2 around z-axis")
    
            theta = torch.tensor([0.,0.,np.pi/2])
            D_cal = euler_to_rotmat(theta)

        elif cfg.calibration == "meanrot":
            print(f"\n[Calibration] Doing {cfg.calibration}")
            M0s = torch.cat([M.unsqueeze(0) for M in M_is])
            Ks, Rs = factorize_projection_matrix_batched(M0s)
            Rs3x3 = complete_rotmat_batch(Rs)
            Rmean = Rs3x3.mean(0)
            U,S,Vh = torch.linalg.svd(Rmean, full_matrices=False) #
            D_cal = U@Vh
            D_cal = D_cal.T
        else:   
            D_cal = torch.eye(3)

    B_cal = torch.cat([(torch.linalg.inv(D_cal) @ b).unsqueeze(0) for b in B])
    M0_cal = torch.cat([(M @ D_cal).unsqueeze(0) for M in M_is ])
    return M0_cal, B_cal, D_cal

def get_q_stats(M0s, alphas, means):
    print("[INFO] Calculating mean and std")
    Ks, Rs = factorize_projection_matrix_batched(M0s)
    Rs3x3 = complete_rotmat_batch(Rs)

    ## [2DO] figure out how to vectorize this function
    thetas = torch.cat([recover_euler_angles(R).unsqueeze(0) 
                        for R in Rs3x3])
    rotvecs = roma.rotmat_to_rotvec(Rs3x3)
    Qs = roma.rotmat_to_unitquat(Rs3x3)
    ks = Ks.reshape(Ks.shape[0],4)[:,[0,1,3]]
    means = torch.cat([m.unsqueeze(0) for m in means],axis = 0)
    parameters = [ks, thetas, alphas, means]
    

    qs = torch.cat([p for p in parameters],axis = 1)
    return qs, easydict.EasyDict({"mean":qs.mean(0),
                              "std":qs.std(0)})

def fit_r1m_model(landmarks,cfg,
                    plot = True, 
                    force_rerun = False,
                    verbose = True):
    if verbose: print("\n [INFO] Fitting rank one model")
    if os.path.exists(cfg.results_path) and not force_rerun:
        results = torch.load(cfg.results_path)
        print("Loading",cfg.results_path)
    else:
    
        normalized_landmarks, means, scalings = preprocess(landmarks)
        N,_, L = landmarks.shape
        
        if N > cfg.max_num_datapoints:
            print("fittin r1m on", cfg.max_num_datapoints, "out of ", N, "in the dataset")
            N = cfg.max_num_datapoints
            normalized_landmarks, means = normalized_landmarks[:N], means[:N]
            
        W = normalized_landmarks.reshape(2*N,L)

        M0, B0, S0, dM, dB, dS = nonrigid_factorization(W)
        
        K = cfg.factorization.K
        dB = dB[:K,:]
        dM = dM[:,:K] 
        W_fac = M0@B0 + dM@ dB


        ##2 step recovery of basis shapes
        B_als, alphas_als, D_als, training_err = recover_rankone_parameters_als(M0, B0,dB, W_fac,**cfg.factorization)

        B, alphas, D, fac_history = recover_rankone_parameters_grad(M0, B0,dB, W_fac,
                            D = D_als, alphas = alphas_als,
                             optim = torch.optim.SGD,  **cfg.factorization)

        M0_cal, B_cal, D_cal = do_calibration(M0,B, cfg, D_cal = None)
        
        qs, stats = get_q_stats(M0_cal, alphas, means)
               
        results = easydict.EasyDict(
            {"B":B_cal,"D":D, 
            "alphas":alphas, 
            "means": means,
            "fac_history": fac_history, 
            "training_err": training_err,
            "D_cal": D_cal,
            "stats": stats,
            "M0_cal": M0_cal,
            "qs": qs}
        )
        
        torch.save(results, cfg.results_path)

        if verbose: print("Saved to",  cfg.results_path)
    
    if plot:
        for k in results.fac_history.keys():
            plt.plot(np.append(results.training_err,
                    np.array(results.fac_history[k])), label = k)
            plt.plot(results.training_err, label = "als")
        plt.legend() 
        plt.show()      
    return results 

def factorize_projection_matrix(M):
    """
    Recovers camera K and rotation matrix R given 2x3 projection matrix. 
    """
    assert M.shape == (2,3)
    q,r = torch.linalg.qr(torch.flipud(M).T)
    K = torch.fliplr(torch.flipud(r.T))
    Q = torch.flipud(q.T)
    s = torch.diag(torch.sign(torch.diag(K))) # Is this the cause of the sign ambiguity ?
    K =  K @ s # camera matrix
    R = s @ Q  # roation 
    return K, R

    
def recover_euler_angles(R):
    """
    Given a 3x2 projection matrix, recovers the euler angles.
    """
    r3 = torch.cross(R[0],R[1])
    tx = torch.tensor([torch.atan2(r3[1],r3[2])])
    ty = torch.tensor([torch.atan2(-r3[0], torch.sqrt(r3[1]**2 + r3[2]**2))])
    tz = torch.tensor([torch.atan2(R[1][0],R[0][0]) ])
    return torch.tensor([tx,ty,tz])

def complete_rotmat(Rp):
    r3 = torch.cross(Rp[0],Rp[1])
    return torch.cat([Rp,r3.unsqueeze(0)])

complete_rotmat_batch = functorch.vmap(complete_rotmat)
recover_euler_angles_bathced = functorch.vmap(recover_euler_angles, in_dims=0, out_dims=0, randomness='error')
factorize_projection_matrix_batched = functorch.vmap(factorize_projection_matrix, in_dims=0, out_dims=0, randomness='error')
