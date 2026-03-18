# This is a Python implementation of the paper "DPGF-Net: Dual-Prior Guided Fusion Network for Joint Assessment of Perceptual Quality and Semantic Consistency in AI-Generated Images".

🚀2026 CVPR Accepted

### Environment
Before running, please create a suitable conda environment.
- conda env create -f environment.yaml
- conda activate AIGC

### Test
To test the model, first download the trained weights from [BaiduDisk](https://pan.baidu.com/s/1q7Je2b3yK8An8-XSOhVdfA?pwd=0221).  
- run `dual_predict_dir.py`.  

### Train
To retrain the model:  
1. Download the pre-trained weights from [BaiduDisk](https://pan.baidu.com/s/1q7Je2b3yK8An8-XSOhVdfA?pwd=0221).  
2. Prepare your own dataset and run `train_dual.py`.  

To reproduce the entire training process:  
1. Collect sufficient real-world images from [SyreaNet](https://github.com/RockWenJJ/SyreaNet), or prepare them yourself.
2. Run `data.py` for data synthesis.  
3. Use the synthesized dataset to pretrain the model via `pretrain.py`.
4. Run `train_dual.py`.  

### Notice
When running these codes, please replace them with your own file path according to the code prompts.  

### Acknowledgements
1. Underwater image synthesis coefficients are computed using the method from [hainh/sea-thru](https://github.com/hainh/sea-thru).  
2. We thank the authors of [DysenNet](https://ieeexplore.ieee.org/abstract/document/10852362) for providing the complete SOTA dataset.
