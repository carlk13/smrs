import random
import argparse

# Seed setzen für reproduzierbare Ergebnisse
random.seed(1)

parser = argparse.ArgumentParser(description="Process some files.")

parser.add_argument("input_file", help="Path to input file")
parser.add_argument("output_file", help="Path to output file")

args = parser.parse_args()


# Zeilen einlesen
with open(args.input_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# Prüfen, ob genug Zeilen vorhanden sind
if len(lines) < 128:
    raise ValueError(f"Nur {len(lines)} Zeilen vorhanden, 128 benötigt.")

# 128 zufällige auswählen
sampled = random.sample(lines, 128)

# Speichern
with open(args.output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(sampled))

print("128 zufällige Queries gespeichert in:", args.output_file)
