import argparse
import random
import os


def main():
    parser = argparse.ArgumentParser(description="Randomly sample queries from a file.")
    parser.add_argument("input_file", help="Path to input text file")
    parser.add_argument("output_file", help="Path to output text file")
    parser.add_argument("amount_needed", type=int, help="Amount of queries to sample")
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    random.seed(args.seed)

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < args.amount_needed:
        raise ValueError(
            f"Only {len(lines)} lines present, but {args.amount_needed} needed."
        )

    sampled = random.sample(lines, args.amount_needed)

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(sampled))

    print(f"Saved {args.amount_needed} random queries to: {args.output_file}")


if __name__ == "__main__":
    main()
