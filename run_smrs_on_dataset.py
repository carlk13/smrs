import argparse
import ssl
import os
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from pathlib import Path

from smrs import sparse_modeling_representative_selection
from torchvision.utils import save_image

# needed when downloading datasets behind certain firewalls/proxies.
ssl._create_default_https_context = ssl._create_unverified_context


def get_data_for_smrs(dataset_name: str, split: str, filter_class: int = None, data_root: str = './data'):
    """
    Loads a dataset, optionally filters it for a single class, and prepares it for SMRS.

    The SMRS algorithm expects data in a specific shape: a single matrix Y of
    shape (D, N), where D is the number of features (e.g., flattened image pixels)
    and N is the number of samples.

    Args:
        dataset_name (str): The name of the dataset (e.g., 'cifar10', 'mnist').
        split (str): Which dataset split to use, 'train' or 'test'.
        filter_class (int, optional): The class index to filter for. If None,
                                      the entire dataset split is used. Defaults to None.
        data_root (str): The root directory where datasets are stored/downloaded.

    Returns:
        tuple: A tuple containing:
            - Y (torch.Tensor): The prepared data tensor of shape (D, N).
            - labels (torch.Tensor): The corresponding labels for the N samples.
    """
    print(f"Loading '{dataset_name}' dataset ({split} split)...")
    
    # Use a simple ToTensor transform, as SMRS doesn't need data augmentation.
    transform = transforms.ToTensor()

    dataset_map = {
        'cifar10': datasets.CIFAR10,
        'cifar100': datasets.CIFAR100,
        'mnist': datasets.MNIST,
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_class = dataset_map[dataset_name]
    is_train = (split == 'train')
    full_dataset = dataset_class(root=data_root, train=is_train, download=True, transform=transform)

    # ----------------------
    # Filtering Logic 
    # ----------------------
    if filter_class is not None:
        print(f"Filtering for class index: {filter_class}")
        try:
            # PyTorch datasets use .targets or .labels
            targets = torch.tensor(full_dataset.targets if hasattr(full_dataset, 'targets') else full_dataset.labels)
            indices = (targets == filter_class).nonzero(as_tuple=True)[0]
            if len(indices) == 0:
                print(f"Warning: Class index {filter_class} not found or has no samples in the {split} split.")
                return None, None
            dataset = Subset(full_dataset, indices)
            print(f"Found {len(dataset)} samples for class {filter_class}.")
        except AttributeError:
            print(f"Warning: Could not filter dataset '{dataset_name}' by class.")
            dataset = full_dataset # Fallback to using the full dataset
    else:
        print("No class filter applied, using all samples.")
        dataset = full_dataset

    # ---------------------- 
    # Data Preparation 
    # ----------------------
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
    
    all_images = []
    all_labels = []
    for images, labels in tqdm(loader, desc="Aggregating data"):
        all_images.append(images)
        all_labels.append(labels)

    images_tensor = torch.cat(all_images, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    # Reshape the data for SMRS: (N, C, H, W) -> (N, D) -> (D, N)
    # N = number of samples, D = feature dimension (C*H*W)
    N = images_tensor.shape[0]
    Y = images_tensor.view(N, -1).T  # Reshape to (NumFeatures, NumSamples)
    
    # SMRS expects float64 for precision
    Y = Y.to(torch.float64)

    print(f"Data prepared for SMRS. Shape: {Y.shape} (Features, Samples)")
    return Y, labels_tensor

def _get_raw_dataset_for_saving(dataset_name: str, split: str, data_root: str = './data'):
    """
    Helper function to load the full dataset with only ToTensor transform for saving purposes.
    This ensures pixels are scaled to [0.0, 1.0].
    """
    os.makedirs(data_root, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(), # Only scales to [0.0, 1.0], no mean/std normalization
    ])

    dataset_map = {
        'cifar10': datasets.CIFAR10,
        'cifar100': datasets.CIFAR100,
        'mnist': datasets.MNIST,
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_class = dataset_map[dataset_name]
    is_train = (split == 'train')
    dataset = dataset_class(root=data_root, train=is_train, download=True, transform=transform)
    return dataset

def save_representative_images(
    selected_indices: np.ndarray,  
    dataset_name: str,
    split: str,
    filter_class: int = None,
    output_dir: str = "./smrs_representative_images",
    data_root: str = './data' 
):
    """
    Saves the images corresponding to the representative indices found by SMRS.

    Args:
        selected_indices (np.ndarray): A NumPy array of indices representing the
                                       selected representatives from the SMRS algorithm.
                                       These indices are relative to the data matrix 'Y'
                                       that was passed to SMRS.
        dataset_name (str): The name of the dataset ('cifar10', 'cifar100', 'mnist').
        split (str): The dataset split used ('train' or 'test').
        filter_class (int, optional): The class index that the dataset was filtered for
                                      before running SMRS. If None, it means SMRS was run
                                      on the entire dataset (or split). Defaults to None.
        output_dir (str): The directory where the representative images will be saved.
                          Defaults to "./smrs_representative_images".
        data_root (str): The root directory where datasets are stored/downloaded.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving representative images to: {output_dir}/")

    # Load the full, raw dataset (only ToTensor applied) using the same data_root
    full_dataset = _get_raw_dataset_for_saving(dataset_name, split, data_root)

    final_image_indices_to_save = []

    if filter_class is not None:
        # Reconstruct the original indices of the filtered class from the full dataset
        original_indices_of_filtered_class = [
            i for i, (_, label) in enumerate(full_dataset) if label == filter_class
        ]
        
        if len(selected_indices) > 0 and len(original_indices_of_filtered_class) > 0:
            final_image_indices_to_save = [
                original_indices_of_filtered_class[idx] 
                for idx in selected_indices if idx < len(original_indices_of_filtered_class)
            ]
        else:
            print("Warning: No valid indices to map for saving (selected_indices empty or filter_class data empty).")

    else:
        # If no class was filtered, selected_indices directly correspond
        # to the indices in the 'full_dataset'.
        final_image_indices_to_save = selected_indices.tolist()

    if not final_image_indices_to_save:
        print("No representative images to save (resulting list of indices was empty).")
        return

    # Save each representative image
    for i, original_idx in enumerate(final_image_indices_to_save):
        try:
            image_tensor, image_label = full_dataset[original_idx]
        except IndexError:
            print(f"Error: Could not retrieve image at original index {original_idx}. Skipping.")
            continue # Skip to the next index if there's an issue
        
        # Create a descriptive filename
        if filter_class is not None:
            filename = f"rep_smrs_idx_{i:03d}_orig_idx_{original_idx:05d}_class_{filter_class}.png"
        else:
            filename = f"rep_smrs_idx_{i:03d}_orig_idx_{original_idx:05d}_class_{image_label}.png"
            
        save_path = os.path.join(output_dir, filename)
        save_image(image_tensor, save_path)
        print(f"Saved representative image {i+1}/{len(selected_indices)}: {filename}")

    print(f"Finished saving {len(final_image_indices_to_save)} representative images.")

def main():
    """
    Main function to run Sparse Modeling Representative Selection (SMRS) on a dataset.
    
    This script allows you to load a standard vision dataset (like CIFAR-10 or MNIST),
    optionally filter it for a single class, and then run the SMRS algorithm to find
    a small set of representatives.

    """
    parser = argparse.ArgumentParser(description="Run SMRS on a specified dataset and class.")

    # ----------------------
    # Data Arguments
    # ----------------------
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'mnist'], help='Dataset to use.')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'test'], help='Dataset split to use (train or test).')
    parser.add_argument('--filter_class', type=int, default=None,
                        help='(Optional) The integer index of a class to filter for. If not provided, uses the whole dataset.')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for storing/downloading datasets.')


    # ---------------------- 
    # SMRS Algorithm Arguments 
    # ----------------------
    parser.add_argument("--alpha", type=int, default=5,
                        help="Regularization parameter for SMRS, typically in [2, 50].")
    parser.add_argument("--r", type=int, default=0,
                        help="Target dimensionality for optional projection. Enter 0 to use original data.")
    parser.add_argument("--max_iterations", type=int, default=None,
                        help="Maximum number of ADMM iterations for SMRS.")
    parser.add_argument("--verbose", action="store_true",
                        help="If set, prints SMRS progress during iterations.")

    args = parser.parse_args()

    # 1. Load and prepare the data
    Y, labels = get_data_for_smrs(
        dataset_name=args.dataset,
        split=args.split,
        filter_class=args.filter_class
    )

    if Y is None or Y.shape[1] == 0:
        print("No data to process. Exiting.")
        return

    # 2. Run the SMRS algorithm
    print('-' * 80)
    print(f"Running SMRS with alpha = {args.alpha}...")
    
    selected_indices, C = sparse_modeling_representative_selection(
        Y=Y,
        alpha=args.alpha,
        r=args.r,
        verbose=args.verbose,
        max_iterations=args.max_iterations 
    )

    print('-' * 80)
    print(f"SMRS completed. Found {len(selected_indices)} representatives.")
    print(f"Representative Indices: {selected_indices}")
    print('-' * 80)

    output_images_dir = f"./smrs_reps_{args.dataset}_{args.split}_class{args.filter_class if args.filter_class is not None else 'all'}_alpha{args.alpha}"
    save_representative_images(
        selected_indices=selected_indices,
        dataset_name=args.dataset,
        split=args.split,
        filter_class=args.filter_class,
        output_dir=output_images_dir,
        data_root=args.data_root 
    )

    print('-' * 80)
    print("Execution finished.")

if __name__ == "__main__":
    main()