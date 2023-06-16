import os
import gdown
import wget

output_directory = "pretrained_models/"
os.makedirs(output_directory, exist_ok=True)


sg2_256_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl"
sg2_1024_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl"
sg3_256_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-256x256.pkl"
sg3_1024_url ="https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl"

print("Downloading sg2")
wget.download(sg2_256_url, out=output_directory)
wget.download(sg2_1024_url, out=output_directory)


## e4e w + encoder ffhq 1024
# https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO/view
id = "1cUv_reLE6k3604or78EranS7XzuVMWeO"
gdown.download(id = id, output = "pretrained_models/e4e_ffhq_encode.pt",
              quiet=False, fuzzy=True)

# hyperstyle
#https://drive.google.com/file/d/1C3dEIIH1y8w1-zQMCyx7rDF0ndswSXh4/view?usp=sharing
id = "1C3dEIIH1y8w1-zQMCyx7rDF0ndswSXh4"
gdown.download(id = id, output = "pretrained_models/hyperstyle_ffhq.pt",
              quiet=False, fuzzy=True)

## e4e wencoder ffhq 1024
#https://drive.google.com/file/d/1M-hsL3W_cJKs77xM1mwq2e9-J0_m7rHP/view
id = "1M-hsL3W_cJKs77xM1mwq2e9-J0_m7rHP"
gdown.download(id = id, output = "pretrained_models/faces_w_encoder.pt",
              quiet=False, fuzzy=True)

id = "1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn"
gdown.download(id = id, output = "pretrained_models/model_ir_se50.pth",quiet=False, fuzzy=True)

