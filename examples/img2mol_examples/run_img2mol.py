import torch
from img2mol.inference import *
import img2mol
from img2mol.cddd_server import CDDDRequest

###
import os

folder_path = '/home/maicon/workspace/AutoChemDescriptors.0.0.2/examples/img2mol_examples/molecules_png/'

# Get a list of all files in the directory
all_files = os.listdir(folder_path)

# Filter for .png files using a list comprehension
png_files = [os.path.join(folder_path, file) for file in all_files if file.endswith('.png')]


#    # Perform operations with each PNG file

###

device = "cuda:0" if torch.cuda.is_available() else "cpu"
img2mol = Img2MolInference(model_ckpt=None, device=device)
cddd_server = CDDDRequest(host="http://ec2-18-157-240-87.eu-central-1.compute.amazonaws.com")

#res = img2mol(filepath="examples/digital_example1.png")#, cddd_server=cddd_server)

for file_path in png_files:
    res = img2mol(filepath=file_path, cddd_server=cddd_server)
    print(file_path)
    #print(res["smiles"])

#for png_file in png_files:
#    print(res["smiles"])
