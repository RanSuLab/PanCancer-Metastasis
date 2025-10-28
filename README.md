
## Pan-cancer distant metastasis prediction based on graph neural network

### Keywords
Distant metastasis, prognosis, distant metastasis-free survival, whole-slide images, graph neural network

### Dataset
Public data: [TCGA-BLCA](https://portal.gdc.cancer.gov/projects/TCGA-BLCA); [TCGA-CESC](https://portal.gdc.cancer.gov/projects/TCGA-CESC);[TCGA-STAD](https://portal.gdc.cancer.gov/projects/TCGA-STAD); [TCGA-PAAD](https://portal.gdc.cancer.gov/projects/TCGA-PAAD)


## How to Use

### Step 1: Extract image patches from WSIs
```bash
python ./preprocessing/extract_patches.py \
  --img_dir ../test_data/WSI \
  --save_dir ../test_data/patches \
  --coord_dir ../test_data/coordinate \
  --patch_size 512 \
  --min_patches 1
```

### Step 2: Extract patch-level deep features (e.g., using pretrained encoders such as DenseNet).
```bash
python ./preprocessing/extract_features.py \
  --img_path ../test_data/patches \
  --graphs_path ../test_data/features \
  --graphs_name CESC_me \
  --batch_size 20 \
  --label 0 
```

### Step 3: Split graph-level data into K folds for cross-validation
```bash
python ./preprocessing/kfold_split.py \
    --data_path ../test_data/features \
    --save_path ../test_data/5_fold \
    --k_fold 5
```

### Step 4: Train the Model
```bash
python Train.py \
  --model GAT_Net \
  --epochs 200 \
  --batch_size 16 \
  --lr 1e-4 \
  --reg 0.09 \
  --dropout 0.9 \
  --kfold 5 \
  --data_path ../test_data/5_fold \
  --save_model_path ../test_data/model_save \
  --gpu 0
```