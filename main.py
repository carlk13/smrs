import argparse
import clip
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from typing import List

from smrs import (
    sparse_modeling_representative_selection,
    find_representatives,
)


def main():
    parser = argparse.ArgumentParser(description="Run smrs.")
    # Query arguments
    parser.add_argument("query_file", type=str, help="Path to query file")
    parser.add_argument(
        "amount_queries",
        type=int,
        help="The amount of queries which should be used for smrs",
    )

    # Dataset arguments
    parser.add_argument(
        "amount_images_per_class",
        type=int,
        help="Amount of pictures taken from each CIFAR10 class for smrs",
    )

    # SMRS arguments

    parser.add_argument(
        "--alpha",
        type=int,
        help="regularization parameter, typically in [2, 50].",
        default=5,
    )
    parser.add_argument(
        "--r",
        type=int,
        help="target dimensionality for optional projection, enter 0 to use original data.",
        default=0,
    )
    parser.add_argument(
        "--max_iterations", type=int, help="maximum number of ADMM iterations"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="if True, prints information during iterations",
    )
    parser.add_argument(
        "--run_without_pruning",
        action="store_true",
        help="Runs smrs also without pruning",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # -------------------------------
    #  2. Load pictures and prepare queries
    # -------------------------------

    with open(args.query_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip() != ""]

    # Select queries
    queries = queries[: args.amount_queries]

    # Tokenize the text
    tokenized = clip.tokenize(queries)
    query_features = batched_encode_text(model, tokenized, batch_size=64)

    print("-" * 80)
    print(f"query_features.shape: {query_features.shape}")
    print("-" * 80)
    print(f"Prepared {args.amount_queries} queries")
    print("-" * 80)

    # -------------------------------
    #  2. Load pictures from CIFAR10
    # -------------------------------
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
        if class_counts[label] < args.amount_images_per_class:
            selected_indices.append(idx)
            class_counts[label] += 1
        if all(
            count == args.amount_images_per_class for count in class_counts.values()
        ):
            break

    subset = Subset(full_dataset, selected_indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False)
    print(f"Prepared {args.amount_images_per_class * 10} images")
    print("-" * 80)

    # -------------------------------
    # 3. Calculate picture-Embeddings
    # -------------------------------

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Encoding images"):
            images = images.to(device)
            features = model.encode_image(images)
            features /= features.norm(dim=-1, keepdim=True)
            image_embeddings.append(features.cpu())
            image_labels.extend(labels)

    image_features = torch.cat(image_embeddings, dim=0)

    # -------------------------------
    # 4. Calculate Cosine-Similarity
    # -------------------------------
    cosine_similarity = image_features @ query_features.T

    print("-" * 80)
    print(f"Cosine similarity shape: {cosine_similarity.shape}")
    print("-" * 80)

    # -------------------------------
    # 5. Run smrs
    # -------------------------------

    print(
        f"Runninng with {args.amount_queries} queries and {args.amount_images_per_class} images per class. "
    )
    print("-" * 80)

    indices_with_images_pruning, C = sparse_modeling_representative_selection(
        Y=cosine_similarity,
        alpha=args.alpha,
        r=args.r,
        verbose=args.verbose,
        max_iterations=args.max_iterations,
    )

    save_queries_to_txt_file(
        selected_indices=indices_with_images_pruning,
        queries=queries,
        with_pruning=True,
        args=args,
    )

    print("Running smrs for query_feature matrix")
    print("-" * 80)
    indices_queries_only_pruning, C2 = sparse_modeling_representative_selection(
        Y=query_features.T,
        alpha=args.alpha,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
    )

    save_queries_to_txt_file(
        selected_indices=indices_queries_only_pruning,
        queries=queries,
        with_pruning=True,
        args=args,
        queries_only=True,
    )

    if args.run_without_pruning:
        print(
            f"Runninng with {args.amount_queries} queries and {args.amount_images_per_class} images per class without pruning. "
        )
        print("-" * 80)

        # Is equivalent to running query selection on C again
        indices_with_images_wo_pruning = find_representatives(C, thr=0.99, q=2)

        save_queries_to_txt_file(
            indices_with_images_wo_pruning,
            queries=queries,
            with_pruning=False,
            args=args,
        )

        print("Running smrs for query_feature matrix without pruning")
        print("-" * 80)

        # equivalent to running query selection on C2 again
        indices_queries_only_wo_pruning = find_representatives(C2, thr=0.99, q=2)
        save_queries_to_txt_file(
            selected_indices=indices_queries_only_wo_pruning,
            queries=queries,
            with_pruning=False,
            args=args,
            queries_only=True,
        )

    print("SMRS finished.")
    print("-" * 80)


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
    file_middle = "" if queries_only else f"_{args.amount_images_per_class}_images"
    file_suffix = "_pruning.txt" if with_pruning else "_without_pruning.txt"
    filename = f"{args.amount_queries}_queries{file_middle}{file_suffix}"

    queries_images = [queries[i] for i in selected_indices]
    with open(filename, "w") as f:
        for query in queries_images:
            f.write(query + "\n")


if __name__ == "__main__":
    main()
