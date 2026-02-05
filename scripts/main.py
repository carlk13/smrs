import argparse
import logging
import sys
import os
from omegaconf import OmegaConf

# Ensure src is in path if running directly without install
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


import clip
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from typing import List

from smrs import sparse_modeling_representative_selection, find_representatives

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("overrides", nargs="*", help="Any key=value overrides (e.g., defaults.alpha=10)")
    args = parser.parse_args()

    base_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_dotlist(args.overrides)
    cfg = OmegaConf.merge(base_conf, cli_conf)
    
    device = cfg.model.device if torch.cuda.is_available() and cfg.model.device == "cuda" else "cpu"
    logger.info(f"Using device: {device}")
    model, preprocess = clip.load(cfg.model.clip_model, device=device)

    logger.info(f"Loading queries from {cfg.data.query_file}")
    with open(cfg.data.query_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip() != ""]
    queries = queries[: cfg.data.amount_queries]
    
    tokenized = clip.tokenize(queries)
    query_features = batched_encode_text(model, tokenized)

    full_dataset = CIFAR10(root=cfg.data.image_root, train=True, download=True, transform=preprocess)

    transform = preprocess
    full_dataset = CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    image_embeddings = []
    image_labels = []

    # First pictures per class
    class_counts = {i: 0 for i in range(10)}
    selected_indices = []

    for idx, (img, label) in enumerate(full_dataset):
        if class_counts[label] < cfg.data.amount_images_per_class:
            selected_indices.append(idx)
            class_counts[label] += 1
        if all(
            count == cfg.data.amount_images_per_class for count in class_counts.values()
        ):
            break

    subset = Subset(full_dataset, selected_indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False)
    logger.info(f"Prepared {cfg.data.amount_images_per_class * 10} images")


    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Encoding images"):
            images = images.to(device)
            features = model.encode_image(images)
            features /= features.norm(dim=-1, keepdim=True)
            image_embeddings.append(features.cpu())
            image_labels.extend(labels)

    image_features = torch.cat(image_embeddings, dim=0)

    cosine_similarity = image_features @ query_features.T

    # Threshhold cosine similarity if wanted
    if cfg.algorithm.threshold_cosine_matrix:
        cosine_similarity = torch.where(cosine_similarity >= 0.25, 1.0, 0.0)

    logger.info(f"Cosine similarity shape: {cosine_similarity.shape}")

    logger.info(f"Running SMRS with alpha={cfg.defaults.alpha}")
    indices_with_images_pruning, C = sparse_modeling_representative_selection(
        Y=cosine_similarity,
        alpha=cfg.defaults.alpha,
        verbose=cfg.defaults.verbose,
        max_iterations=cfg.defaults.max_iterations,
    )

    save_queries_to_txt_file(
        selected_indices=indices_with_images_pruning,
        queries=queries,
        with_pruning=True,
        args=cfg,
    )

    if cfg.algorithm.run_without_pruning:
        logger.info("Running selection without pruning...")
        indices_no_pruning = find_representatives(
            C, thr=cfg.algorithm.selection_threshold, q=2
        )
        save_queries_to_txt_file(indices_no_pruning, queries, False, cfg)

    if cfg.algorithm.run_query_features_only:
        logger.info("Running smrs for query_feature matrix")
        indices_queries_only_pruning, C2 = sparse_modeling_representative_selection(
            Y=query_features.T,
            alpha=cfg.defaults.alpha,
            verbose=cfg.defaults.verbose,
            max_iterations=cfg.defaults.max_iterations,
        )

        save_queries_to_txt_file(
            selected_indices=indices_queries_only_pruning,
            queries=queries,
            with_pruning=True,
            args=cfg,
            queries_only=True,
        )

        if cfg.algorithm.run_without_pruning:
            logger.info("Running smrs for query_feature matrix without pruning")

            # equivalent to running query selection on C2 again
            indices_queries_only_wo_pruning = find_representatives(C2, thr=0.99, q=2)
            save_queries_to_txt_file(
                selected_indices=indices_queries_only_wo_pruning,
                queries=queries,
                with_pruning=False,
                args=cfg,
                queries_only=True,
            )

    logger.info("SMRS finished.")


def batched_encode_text(
    model, tokenized_texts: torch.Tensor, batch_size: int = 64
) -> torch.Tensor:
    """
    Efficiently encodes a list of pre-tokenized text inputs into normalized CLIP text embeddings.

    Args:
        model: The CLIP model (or a similar model with an `encode_text` method).
        tokenized_texts (torch.Tensor): A tensor containing tokenized text inputs.
        batch_size (int): The number of text inputs to process in each batch. Defaults to 64.

    Returns:
        torch.Tensor: A concatenated tensor of normalized text embeddings, residing on the CPU."""
    device = next(model.parameters()).device
    all_features = []

    with torch.no_grad():
        for i in tqdm(
            range(0, len(tokenized_texts), batch_size), desc="Encoding text", leave=True
        ):
            batch = tokenized_texts[i : i + batch_size].to(device)
            features = model.encode_text(batch)
            features /= features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


def save_queries_to_txt_file(
    selected_indices: torch.Tensor,
    queries: List[str],
    with_pruning: bool,
    args,
    queries_only: bool = False,
) -> None:
    """
    Saves a subset of text queries to a plain text file.

    The file name is constructed using `args.amount_queries`, `args.amount_images_per_class`,
    and a fixed suffix "_pruning.txt".

    Args:
        selected_indices (torch.Tensor): A list of tensors containing indices of the queries to save.

        queries (List[str]): The complete list of available text queries.
        queries_only(bool): For running smrs on the query matrix. Default = False.
        with_pruning(bool) : If True, the filename will include "_pruning.txt".
                             If False, the filename will be "_images.txt".
        args: An object (argparse.Namespace) containing `amount_queries` and `amount_images_per_class`
              attributes used for constructing the output filename.
    """
    # Determine the file suffix based on the with_pruning parameter
    file_middle = "" if queries_only else f"_{args.data.amount_images_per_class}_images"
    file_suffix = "_pruning.txt" if with_pruning else "_without_pruning.txt"
    filename = f"{args.data.amount_queries}_queries{file_middle}{file_suffix}"

    queries_images = [queries[i] for i in selected_indices]
    with open(filename, "w") as f:
        for query in queries_images:
            f.write(query + "\n")


if __name__ == "__main__":
    main()
