# This is a Python implementation of the paper "DPGF-Net: Dual-Prior Guided Fusion Network for Joint Assessment of Perceptual Quality and Semantic Consistency in AI-Generated Images".

### Environment
Before running, please create a suitable conda environment.
- conda env create -f environment.yaml
- conda activate AIGC
- Download the compiled [Reiqa](https://pan.baidu.com/s/1VGA-Xxgr3uT6K1EIkFxfEQ?pwd=0221) code files to replace the contents of the ReIQA_main folder.

### Test
To test the model, first download the trained weights from [BaiduDisk](https://pan.baidu.com/s/13amXPeCtI-SDndy6ihb-HQ?pwd=0221).  
- Run `test_alignment.sh`.
- Run `test_quality.sh`.  

### Train
To retrain the model:  
1. Download the relevant dataset from [BaiduDisk](https://pan.baidu.com/s/1Q-04YzcXyMefLDxQUKG2Ug?pwd=0221).
2. Place it in the dataset folder.
3. Run `train.py`.  

### Notice
When running these codes, please replace them with your own file path according to the code prompts.  


### Acknowledgements
1. Our content extractor and distortion extractor are based on [Re-IQA](https://github.com/avinabsaha/ReIQA).
