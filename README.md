# SpaDAC: SPAtially embedded Deep Attentional graph Clustering
## Overview
We proposed an unsupervised multi-model **SPAtially embedded Deep Attentional graph Clustering (SpaDAC)** method, which uses the deep learning framework to learn low-dimensional embeddings for spatial transcriptomics data. SpaDAC can efficiently identify spatial domains while reconstructing denoised gene expression profiles. We applied it to sixteen datasets covering three situations. Benchmark results demonstrated that SpaDAC outperforms other algorithms in most cases. We expected SpaDAC to offer a valuable computational tool for researchers to understand tissue organization and function.

<p align="center">
	<img src="https://user-images.githubusercontent.com/79891846/230755717-c2359e39-293a-4ae1-bf25-8bd21a1ea9e2.jpg" width="367.9" height="622.7" alt="Image">
</p>

## Tutorial
#### Start by grabbing this source codes:
```
git clone https://github.com/huoyuying/SpaDAC.git
cd SpaDAC
```
### 1. Folder hierarchy
#### We take 10X sample 151673 as an running example. After grabbing the source codes, you can see the folder hierarchy as follows:
```
.
├── datasets
│   └── 151673
│       └── notpca
├── denoising
│   └── 151673_cor
├── image_feature_learning_tool
│   ├── cut_image
│   │   └── 151673
│   │       ├── cor
│   │       ├── img_151673_224
│   │       └── img_151673_299
│   ├── inception_resnet_v2
│   │   ├── _
│   │   └── variables
│   ├── inception_v3
│   │   └── variables
│   └── resnet50
│       ├── _
│       └── variables
├── pretrain
│   └── 151673
├── result
│   └── 151673
├── pretrain_dual.py    # dual-mode pretraining code
├── pretrain_triple.py  # triple-mode pretraining code
├── daegc_dual.py       # dual-mode training code
├── daegc_triple.py     # triple-mode training code
├── model.py
├── model_plus.py
├── utils.py
└── evaluation.py
```
### 2. Virtual environment
#### (Recommended) Using python virtual environment with  [`conda`](https://anaconda.org/)
```
# Configuring the virtual environment
conda create -n SpaDAC_env python=3.8
conda activate SpaDAC_env
pip install -r SpaDAC_requirement.txt
```
### 3. Usage
#### 3-1 Image cutting (optional)
```
cd SpaDAC/image_feature_learning_tool
pip install -r muse_requirement.txt
cd cut_image
python deal_cut.py --name 151673 --size 224 (or 299)
```
##### ```--name```: Name of sample
##### ```--size```: Image resolution
#### 3-2 Extraction of morphological features (optional)
```
cd SpaDAC/image_feature_learning_tool
python deal_inception.py --name 151673 --model inception_v3 (or inception_resnet_v2 or resnet50)
```
##### ```--name```: Name of sample
##### ```--model```: The Convolutional Neural Network used
#### 3-3  Calculation of morphological similarity network and adjacency matrix (optional)
```
cd SpaDAC/image_feature_learning_tool
python deal_network.py --name 151673 --model inception_v3 (or inception_resnet_v2 or resnet50) --distance euc (or cos or pea) --neighbor 6 (or 4)
```
##### ```--name```: Name of sample
##### ```--model```: The Convolutional Neural Network used
##### ```--distance```: The distance used to measure the similarity between cells
##### ```--neighbor```: The number of neighbors per cell
#### 3-4  pretraining
```
cd SpaDAC
python pretrain_plus.py --name 151673 --exp 3000 --adj adj1 --img adj6 --max_epoch 50
```
##### ```--name```: Name of sample
##### ```--exp```: The number of highly variable features(HVGs) selected
##### ```--adj```: The 01-Matrix of whether cells are neighbors or not, based on geographical similarity
##### ```--img```: The 01-Matrix of whether cells are neighbors or not, based on morphological similarity
##### ```--max_epoch```: The number of iterations of this training
#### 3-5 fine-tunning
```
cd SpaDAC
python daegc_plus.py --name 151673 --exp 3000 --adj adj1 --img adj6 --epoch 49 --max_epoch 100
```
##### ```--name```: Name of sample
##### ```--exp```: The number of highly variable features(HVGs) selected
##### ```--adj```: The 01-Matrix of whether cells are neighbors or not, based on geographical similarity
##### ```--img```: The 01-Matrix of whether cells are neighbors or not, based on morphological similarity
##### ```--epoch```: The number of iterations of last training
##### ```--max_epoch```: The number of iterations of this training
#### 3-6 Clustering and optimization
```
cd SpaDAC
python clustering.py
```
#### 3-7 Denoising of gene expression profile
```
cd SpaDAC/denoising
python denoising.py
```
### 4. Download data
|      Platform      |       Tissue     |    SampleID   |
|:----------------:|:----------------:|:------------:|
| [10x Visium](https://support.10xgenomics.com) | Human dorsolateral pre-frontal cortex (DLPFC) | [151507,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151507_filtered_feature_bc_matrix.h5) [151508,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151508_filtered_feature_bc_matrix.h5) [151509,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151509_filtered_feature_bc_matrix.h5) [151510,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151510_filtered_feature_bc_matrix.h5) [151669,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151669_filtered_feature_bc_matrix.h5) [151670,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151570_filtered_feature_bc_matrix.h5) [151671,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151671_filtered_feature_bc_matrix.h5) [151672,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151672_filtered_feature_bc_matrix.h5) [151673,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151673_filtered_feature_bc_matrix.h5) [151674,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151674_filtered_feature_bc_matrix.h5) [151675,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151675_filtered_feature_bc_matrix.h5) [151676](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151676_filtered_feature_bc_matrix.h5)
| [10x Visium](https://support.10xgenomics.com) | Mouse brain section| [Sagittal-Anterior,](https://www.10xgenomics.com/resources/datasets/mouse-brain-serial-section-1-sagittal-anterior-1-standard-1-1-0) [Sagittal-Posterior](https://www.10xgenomics.com/resources/datasets/mouse-brain-serial-section-1-sagittal-posterior-1-standard-1-1-0)
| [10x Visium](https://support.10xgenomics.com) | Human breast cancer| [Ductal Carcinoma In Situ & Invasive Carcinoma](https://www.10xgenomics.com/resources/datasets/human-breast-cancer-ductal-carcinoma-in-situ-invasive-carcinoma-ffpe-1-standard-1-3-0) 
| [Stereo-Seq](https://www.biorxiv.org/content/10.1101/2021.01.17.427004v2) | Mouse olfactory bulb| [Olfactory bulb](https://github.com/BGIResearch/stereopy) 
| [ST](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE111672) |  Pancreatic ductal adenocarcinoma tissue| [PDAC1,](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE111672&format=file&file=GSE111672%5FPDAC%2DA%2Dindrop%2Dfiltered%2DexpMat%2Etxt%2Egz) [PDAC2](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE111672&format=file&file=GSE111672%5FPDAC%2DB%2Dindrop%2Dfiltered%2DexpMat%2Etxt%2Egz) 

Spatial transcriptomics data of other platforms can be downloaded https://www.spatialomics.org/SpatialDB/

### 5. Contact
Feel free to submit an issue or contact us at 21121732@bjtu.edu.cn for problems about the packages.
