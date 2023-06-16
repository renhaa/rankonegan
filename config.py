import easydict
import os 

os.makedirs("data/", exist_ok=True)
os.makedirs("results/", exist_ok=True)

sgmodel_to_path = {
"sg2-256-ffhq":"pretrained_models/stylegan2-ffhq-256x256.pkl", 
"sg2-humans":"pretrained_models/stylegan_human_v2_512.pkl",    
"sg3-256-ffhqu":"pretrained_models/stylegan3-r-ffhqu-256x256.pkl",
"sg2-1024-ffhq":"pretrained_models/stylegan2-ffhq-1024x1024.pkl", 
# "sg3-1024-ffhq-3rd":"pretrained_models/sg3-r-ffhq-1024.pt", 
"sg3-1024-ffhq":"pretrained_models/stylegan3-r-ffhq-1024x1024.pkl"
}


def create_default_config(latent_space = "w",
                          num_samples = 10000,
                          landmark_type = "mediapipe",
                          sgmodel = "sg2-256-ffhq", 
                          calibration = "meanrot",
                          do_pca_transform = False,
                          K = 12,
                          num_layers = 3,
                          layer_width = 512,
                          mlp_batchsize = 256,
                          output_normalization = True,
                          use_lpips_transform = False,
                          num_lpips_hession_eigvecs = 256,
                          mlp_lr = 2e-5,
                          mlp_weight_decay = 0,
                          mlp_max_iters = 50,
                          normalize_w_std = False
                          ):

    cfg = easydict.EasyDict()

    cfg.sgmodel = sgmodel
    cfg.sg_path = sgmodel_to_path[sgmodel]
    cfg.num_samples = num_samples
    cfg.latent_space = latent_space
    cfg.landmark_type = landmark_type

        
    cfg.data_label = f"{sgmodel}-{int(num_samples/1e3)}K" 
    if "stylemix" in cfg.latent_space:
        cfg.data_label += "-stylemix" 
    cfg.data_path = f"data/{cfg.data_label}.pt"

    cfg.calibration = calibration

    cfg.training_label = f"{latent_space}-{landmark_type}"
    cfg.results_path = f"results/{cfg.data_label}-{cfg.training_label}.pt"
    
    ## SG Model is from S space enabled third time repo
    cfg.is_third_time_repo = "3rd" in sgmodel or "sg3" in sgmodel
    cfg.use_lpips_transform = use_lpips_transform
    cfg.num_lpips_hession_eigvecs = num_lpips_hession_eigvecs
    ## Factorization parameters
    cfg.factorization = easydict.EasyDict()
    cfg.max_num_datapoints = 50000
    cfg.factorization.K = K # Number of non-rigid basis shapes
    cfg.factorization.lr = 0.001 
    cfg.factorization.num_iters_als = 10
    cfg.factorization.num_iters_grad = 0
    cfg.factorization.reg_strength = 5
    cfg.factorization.do_pca_transform = do_pca_transform
    # cfg.factorization.max_data_used = 1e4
        
    ## MLP Setting
    cfg.mlp = easydict.EasyDict()
    cfg.normalize_w_std = normalize_w_std
    cfg.mlp.num_layers = num_layers
    cfg.mlp.layer_width = layer_width
    cfg.mlp.max_iters = mlp_max_iters
    cfg.mlp.lr = mlp_lr
    cfg.mlp.batch_size = mlp_batchsize  # 1024
    cfg.mlp.weight_decay = mlp_weight_decay
    cfg.mlp.optim_strategy = "adam"
    cfg.mlp.output_normalization = output_normalization
    return cfg



cfgs = easydict.EasyDict()

cfgs.sg2_256_ffhq = create_default_config(
        sgmodel = "sg2-256-ffhq", 
        latent_space = "w",
        num_samples = 10000,
        landmark_type = "mediapipe"
)





