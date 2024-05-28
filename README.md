# Multimodal Network based on Textual Prompt (TPMNet): A practical Transformer for Image Deraining
<hr />

**Abstract:** *Image restoration technology aims to recover high-quality visual content from degraded images affected by weather conditions such as rain, fog, and snow, which can severely impact the performance of advanced visual tasks like image classification and object detection. Traditional methods often rely on simple linear mappings and face limitations in handling complex rain patterns. However, there remains a need for models that can handle complex rain patterns and maintain high computational efficiency. To address these challenges, we propose TPMNet, a multimodal Transformer framework specifically designed for image deraining. TPMNet integrates semantic prompts and multimodal learning mechanisms through the Text-Enhanced Depth and Spatial Attention (TEDSA) block, the Gated Contextual Integration Network (GCIN) block, and a multi-scale hybrid strategy. These components together enhance the model's adaptability and restoration performance across various degradation types. Extensive experiments on multiple public benchmark datasets demonstrate that TPMNet achieves superior state-of-the-art performance in image deraining tasks. Our model not only effectively removes rain streaks while preserving image details but also provides a natural, precise, and controllable interactive approach for future low-level image restoration research.*
<hr />

## Network Architecture

<img src = "https://imgur.com/MohHjAP.jpg"> 

## Installation 

This repository is built in PyTorch 1.11.0 and tested on Ubuntu 20.04 environment (Python3.8, CUDA11.3).
Follow these intructions

1. Clone our repository
```
git clone https://github.com/Wosaw/TPMNet.git
cd TPMNet
```

2. Make conda environment
```
conda create -n TPMNet python=3.8
conda activate TPMNet
```

3. Install dependencies
```
conda install pytorch=1.11 torchvision cudatoolkit -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
```

4. Install basicsr
```
python setup.py develop --no_cuda_ext
```
5. Add basicsr to path
```
export PYTHONPATH="${PYTHONPATH}:/path/to/TPMNet/basicsr"
```
### Download datasets 

Please download the  datasets from [BaiduYun](https://pan.baidu.com/s/1iYyHenRQDFVsKLwgtrAUWw?pwd=bna3)

## Training

1. To download Rain13K training and testing data, run
```
python download_data.py --data train-test
```

2. To train TPMNet with default settings, run
```
cd TPMNet
./train.sh Deraining/Options/TPMNet_config.yml
```

**Note:** The above training script uses 1 GPU by default. To use any other number of GPUs, modify [TPMNet/train.sh](../train.sh) and [Deraining/Options/TPMNet_config.yml](Deraining/Options/TPMNet_config.yml)

## Evaluation

1. Download the pre-trained [model](https://drive.google.com/file/d/1A3A5SEkbJYJ4pOt1B6KdNfuEQFR7gIE6/view?usp=drive_link) and place it in `./pretrained_models/`

2. Download test datasets (Test100, Rain100H, Rain100L, Test1200, Test2800), run 
```
python download_data.py --data test
```

3. Testing
```
python test.py
```

#### To reproduce PSNR/SSIM scores of Table 3 in paper, run

```
python  ./Deraining/calculate.py
```

