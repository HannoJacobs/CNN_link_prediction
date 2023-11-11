import sys, os, time
from pathlib import Path
print("\n")
FILE = Path(__file__)
project_root = FILE.parent.parent
sys.path.insert(1, str(project_root.resolve()))

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# device = torch.device("cpu") # Normal training for non-setup GPU
device = torch.device("mps") # M1 Macbook gpu training
max_num_batches_to_run = 1e20 # set this to a low value for debugging functionality

df_path = "Dataset/train.csv" # path to the official train.csv file
out_df_path = "Dataset/test_predicted.csv" # path to the official train.csv file
load_model_path = "model_saves/ConvNN_final.pth"  # model to load if retrain=False
base_model_save_dir = "model_saves/" # relative location to save this model if retrain=True
output_results_save_file_name = "model_saves/metrics_results.txt" # metrics save file
this_file_name = str(os.path.basename(__file__))

class NeuralNet(nn.Module):
    def __init__(self, input_size=836624, embedding_dim=50, dropout_rate=0.2):
        super(NeuralNet, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2, stride=1)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, padding=2, stride=1)
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.seq = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(in_features=(32 * embedding_dim), out_features=1),
            nn.Sigmoid(),
        )
    
    def forward(self, id1, id2):
        id1_embedded = self.embedding(id1) # embed the IDs to put it through the NN
        id2_embedded = self.embedding(id2) # embed the IDs to put it through the NN
        x = id1_embedded + id2_embedded
        
        x = x.unsqueeze(1) # Reshape for Conv1d
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        y = self.seq(x).squeeze() # Pass through the sequential block
        return y

class test_csv_LinkPredictionDataset(Dataset):
    """
    This is the Loader that will load the dataset into a form 
    that the "DataLoader" can use for training in batches.
    """
    def __init__(self, train_df):
        self.train_df = train_df
    
    def __len__(self):
        return len(self.train_df)
    
    def __getitem__(self, id_):
        df = self.train_df
        id1 = df.iloc[id_]['id1']
        id2 = df.iloc[id_]['id2']
        return id1, id2

if __name__ == "__main__":
    main_start_time = time.time()
    full_df = pd.read_csv(df_path)
    test_dataset = DataLoader(test_csv_LinkPredictionDataset(full_df), batch_size=100, shuffle=False)

    model = NeuralNet().to(device)
    model.load_state_dict(torch.load(load_model_path))
    print(f"Trained model loaded from {load_model_path}")

    all_predictions = []
    model.eval()
    with torch.no_grad():
        for batch_num, (id1, id2) in enumerate(test_dataset):
            if batch_num > max_num_batches_to_run:
                break # this will break out if you are just debugging print and file writes

            if batch_num % 100 == 0:
                print(f"test_csv predictions are {100*(batch_num/len(test_dataset)):.0f}% done")

            id1, id2 = id1.to(device), id2.to(device)
            outputs = model.forward(id1, id2)
            predicted = (outputs > 0.5).int()
            
            # Move the predictions to CPU and convert to numpy
            predicted = predicted.cpu().numpy()
            all_predictions.extend(predicted)

    full_df['pred_labels'] = all_predictions
    full_df.to_csv(out_df_path, index=False)

    print(f"\n\nmain() run_time = {((time.time() - main_start_time)/60):.2f} min\n\n")
