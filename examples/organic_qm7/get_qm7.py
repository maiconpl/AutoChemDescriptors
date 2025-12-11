# 08/12/25

# https://quantum-machine.org/datasets/

import numpy as np
from scipy.io import loadmat
import random

# --- 1. Load the QM7 Dataset ---
# You'll need to download the QM7 dataset file (e.g., qm7.mat)
# and replace 'path/to/qm7.mat' with the actual file path.
try:
    # QM7 data is commonly stored in a .mat (MATLAB) file format
    # which contains arrays for Z (atomic charges) and R (coordinates).
    #qm7_data = loadmat('path/to/qm7.mat')
    qm7_data = loadmat('/mnt/d/workspace/AutoChemDescriptors.0.0.1/examples/organic_qm7/qm7.mat')
except FileNotFoundError:
    print("Error: QM7 data file not found. Please download it and update the path.")
    # Exiting or creating a dummy for demonstration
    # In a real scenario, you'd handle the download or error
    exit()

# The data typically includes:
# 'Z': Atomic charges (7165)
# 'R': Cartesian coordinates (7165 x 23 x 3)
# 'T': Atomization energies (7165)
Z = qm7_data['Z']
R = qm7_data['R']
TOTAL_MOLECULES = Z.shape[0] # Should be 7165

# --- 2. Determine the Sample Size ---
SAMPLE_SIZE = 5000
if SAMPLE_SIZE > TOTAL_MOLECULES:
    print(f"Warning: Requested size ({SAMPLE_SIZE}) is greater than total molecules in QM7 ({TOTAL_MOLECULES}). Using all molecules.")
    SAMPLE_SIZE = TOTAL_MOLECULES

# --- 3. Randomly Select Indices ---
# Get a list of indices from 0 up to 7164
all_indices = list(range(TOTAL_MOLECULES))
# Randomly select 5000 unique indices
random_indices = random.sample(all_indices, SAMPLE_SIZE)

# --- 4. Create the List of Random Molecules ---
# We will combine the coordinates (R) and charges (Z) for each molecule
# and store them as tuples in the Python list.

random_molecules_list = []

for i in random_indices:
    # Each molecule entry will be a tuple: (atomic_charges_array, coordinates_array)
    molecule_data = (Z[i], R[i])
    random_molecules_list.append(molecule_data)

# --- 5. Output the Result ---
print(f"\nSuccessfully created a list of {len(random_molecules_list)} random molecules.")
print("The list contains tuples, where each tuple is (Atomic Charges, Cartesian Coordinates).")
print(f"Example of the first element's structure (Atomic Charges): \n{random_molecules_list[0][0][:5]}...")
print(f"Example of the first element's structure (Coordinates shape): \n{random_molecules_list[0][1].shape}")

print(random_molecules_list)
