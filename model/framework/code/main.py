# imports
import os
import csv
import sys
import numpy as np

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# current file directory
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, ".."))
import dotP_encoder as de


# read SMILES from .csv file, assuming one column with header
with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    smiles_list = [r[0] for r in reader]

# run dotP model
embeddings, valid_idx = de.encode_molecules(smiles_list, reduced=True)  # reduced len 512, not reduced len 3072
outputs = np.array(embeddings, dtype=str)

#check input and output have the same lenght
input_len = len(smiles_list)
output_len = len(outputs)
assert input_len == output_len

# write output in a .csv file
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow([f"dim_{str(i).zfill(3)}" for i in range(outputs.shape[1])])
    for o in outputs:
        writer.writerow(o)
