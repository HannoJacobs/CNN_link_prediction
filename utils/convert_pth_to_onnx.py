import onnx
import torch
import torch.onnx
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import time
from Full_work.ConvNN import NeuralNet
from Full_work.ConvNN import LinkPredictionDataset

device = torch.device("mps") # M1 Macbook gpu training
pth_model_path = "/Users/hannojacobs/Documents/2_Uni/_4_COS_project/model_saves/final_presentation_points/ConvNN_final.pth"
full_df = pd.read_csv("/Users/hannojacobs/Documents/2_Uni/_4_COS_project/Dataset/official_dataset/train_head.csv")

model = NeuralNet().to(device)

# Load the trained model from file
model.load_state_dict(torch.load(pth_model_path))

# Set the model to evaluation mode
model.eval()

train_dataset = DataLoader(LinkPredictionDataset(full_df), batch_size=100, shuffle=False)

in1 = None
in2 = None

for batch_num, (id1, id2, label) in enumerate(train_dataset):
    if batch_num > 0:
        break
    id1, id2, label = id1.to(device), id2.to(device), label.to(device)
    output = model.forward(id1, id2)
    in1, in2 = id1, id2

id1, id2 = in1, in2
id1, id2 = id1.to(device), id2.to(device)
outputs = model.forward(id1, id2)


dummy_input = (id1, id2)

print(dummy_input == (id1, id2))

# model.forward(dummy_input)

print("done")

# Specify the path to save the ONNX model
onnx_model_path = "/Users/hannojacobs/Documents/2_Uni/_4_COS_project/model_saves/final_presentation_points/ConvNN_final.onnx"

# Export the model
torch.onnx.export(  model,         # Model being run
                    dummy_input,           # Model input (or a tuple for multiple inputs)
                    onnx_model_path,       # Where to save the model
                    export_params=True,    # Store the trained parameter weights inside the model file
                    opset_version=10,      # The ONNX version to export the model to
                    do_constant_folding=True,  # Whether to execute constant folding for optimization
                    input_names = ['id1', 'id2'],   # the model's input names
                    output_names = ['label'], # the model's output names
                    dynamic_axes={'modelInput' : {0 : 'batch_size'},    # Variable length axes
                                    'modelOutput' : {0 : 'batch_size'}})


