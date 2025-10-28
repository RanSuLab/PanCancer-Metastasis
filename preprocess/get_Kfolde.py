import argparse
import numpy as np
from random import shuffle
import os

def check_graph(train_path, test_path):
    print('-- Checking overlap between train and test datasets --')
    train_list = np.load(train_path, allow_pickle=True).tolist()
    test_list = np.load(test_path, allow_pickle=True).tolist()

    train_wsi = {d['WSI_name'].split('+')[0] for d in train_list}
    test_wsi = {d['WSI_name'].split('+')[0] for d in test_list}

    print(f'Train WSI count: {len(train_wsi)} | Test WSI count: {len(test_wsi)}')

    overlap = train_wsi & test_wsi
    if overlap:
        print(f'⚠️ {len(overlap)} overlapping WSIs found:')
        for name in overlap:
            print(name)
    else:
        print('✅ No overlapping WSIs detected.')

def split_list(wsi_list, k):
    n = len(wsi_list)
    fold_size = round(n / k)
    folds = [wsi_list[i*fold_size:(i+1)*fold_size] for i in range(k-1)]
    folds.append(wsi_list[(k-1)*fold_size:])
    return folds

def kfold_split(all_data, save_dir, fold_idx, list0, list1):
    new_0 = [d for d in all_data if d['WSI_name'].split('+')[0] in list0]
    new_1 = [d for d in all_data if d['WSI_name'].split('+')[0] in list1]

    test_data = np.array(new_0 + new_1)
    train_data = np.array([d for d in all_data if d not in test_data])

    os.makedirs(f"{save_dir}/{fold_idx+1}", exist_ok=True)
    np.save(f"{save_dir}/{fold_idx+1}/train.npy", train_data, allow_pickle=True)
    np.save(f"{save_dir}/{fold_idx+1}/test.npy", test_data, allow_pickle=True)

    check_graph(f"{save_dir}/{fold_idx+1}/train.npy", f"{save_dir}/{fold_idx+1}/test.npy")

def main(args):
    print(f"==> Performing {args.k_fold} fold split for data:")
    print(f"data_path: {args.data_path}")
    print(f"Save to: {args.save_path}\n")

    all_data = np.load(args.data_path, allow_pickle=True).tolist()


    wsi_0 = list(set(d['WSI_name'].split('+')[0] for d in all_data if d['flow_y'] == 0))
    wsi_1 = list(set(d['WSI_name'].split('+')[0] for d in all_data if d['flow_y'] == 1))

    folds_0 = split_list(wsi_0, args.k_fold)
    folds_1 = split_list(wsi_1, args.k_fold)

    for i in range(args.k_fold):
        print(f"\n--- Fold {i+1}/{args.k_fold} ---")
        kfold_split(all_data, args.save_path, i, folds_0[i], folds_1[i])

    print(f"\n✅ K-fold split complete. Saved to: {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-fold graph data splitter")
    parser.add_argument("--data_path", type=str, required=True, help="Path to features")
    parser.add_argument("--save_path", type=str, required=True, help="Output folder for k-fold splits")
    parser.add_argument("--k_fold", type=int, default=5, help="Number of folds (default: 5)")
    args = parser.parse_args()
    main(args)
