# Visible-Infrared Person Re-Identification via Coupled Channel Destylization and Spatial Refinement
## Prepare Dataset
Download the datasets ([SYSU-MM01](https://github.com/wuancong/SYSU-MM01), [RegDB](https://gitcode.com/Premium-Resources/47a2b) , [LLCM](https://github.com/ZYK100/LLCM)), and then unzip them to your_dataset_dir.
## Training
Train a model by
```bash
python train.py --dataset sysu --gpu 0 
```
- `--dataset`：which dataset "sysu", "regdb" or "llcm".
## Evaluation
Test a model on SYSU-MM01 or RegDB dataset by
```bash
python test.py --mode all --resume 'model_path' --gpu 0 --dataset sysu
```
## Citation
If you use this code for your research, please cite
