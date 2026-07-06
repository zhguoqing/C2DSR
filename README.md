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
python test.py --mode all --resume 'model_path' --gpu 0 --dataset sysu --trial 1
```
- `--dataset`：which dataset "sysu", "regdb" or "llcm".
- `--mode`："all" or "indoor" all search or indoor search (only for sysu dataset).
- `--trial`：testing trial (only for RegDB dataset).
- `--resume`：the saved model path.
## Citation
If you use this code for your research, please cite
```bash

```
