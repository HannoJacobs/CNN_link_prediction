import sys, os, time
from pathlib import Path
print("\n")
FILE = Path(__file__)
project_root = FILE.parent.parent
sys.path.insert(1, str(project_root.resolve()))

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

##### start set these variables #####
##### start set these variables #####
# device = torch.device("cpu")  # Normal training for non-setup GPU
device = torch.device("mps")# M1 Macbook gpu training

full_run = True # set this to False for debugging of printing and file writing
retrain = True  # True if you want to retrain
num_epochs = 10 # the num epochs you want to train for

optimizers_to_test = ["Adam"]
loss_metrics_to_test = ["BCE"]
##### end set these variables #####
##### end set these variables #####

df_path = "Dataset/train.csv" # path to the official train.csv file
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
        # embed the IDs to put it through the NN
        id1_embedded = self.embedding(id1)
        id2_embedded = self.embedding(id2)
        x = id1_embedded + id2_embedded
        
        x = x.unsqueeze(1) # Reshape for Conv1d
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        y = self.seq(x).squeeze() # Pass through the sequential block
        return y

class LinkPredictionDataset(Dataset):
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
        label = df.iloc[id_]['label']
        return id1, id2, label

def calc_metrics(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    with torch.no_grad():
        for counter1, (id1, id2, label) in enumerate(dataloader):
            if full_run == False:
                if counter1 > 0:
                    break # this will break out if you are just debugging print and file writes
            id1, id2, label = id1.to(device), id2.to(device), label.to(device)
            outputs = model.forward(id1, id2)
            predicted = (outputs > 0.5).float()

            for counter2, (ground_truth, predicted_label) in enumerate(zip(label, predicted)):
                if ground_truth == 1 and predicted_label == 1:
                    TP += 1
                    continue
                elif ground_truth == 0 and predicted_label == 0:
                    TN += 1
                    continue
                elif ground_truth == 0 and predicted_label == 1:
                    FP += 1
                    continue
                elif ground_truth == 1 and predicted_label == 0:
                    FN += 1
                    continue

    correct = TP + TN
    total = TP + TN + FP + FN

    # Calculate Precision, Recall, and F1 Score
    precision = round((TP/(TP + FP)),4) if (TP + FP) > 0 else 0
    recall = round((TP/(TP + FN)),4) if (TP + FN) > 0 else 0
    f1_score = round((2*(precision*recall)/(precision + recall)),4) if (precision + recall) > 0 else 0
    accuracy = round(((correct/total)*100),4)

    dict_results = {
        "f1_score": f1_score,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "correct": correct,
        "total": total,
    }

    return dict_results

def determine_train_test_metrics(model_, train_dataset_, test_dataset_, device_):
    metrics_test_start_time = time.time()
    print("\n")
    print(f"Calculating metrics...")
    train_metrics = calc_metrics(model_, train_dataset_, device=device_)
    print(f"Train_metrics = {train_metrics}")
    test_metrics = calc_metrics(model_, test_dataset_, device=device_)
    print(f"Test_metrics = {test_metrics}")
    print(f"Metrics test time = {((time.time() - metrics_test_start_time)/60):.2f} min\n")
    return train_metrics, test_metrics

if __name__ == "__main__":
    main_start_time = time.time()
    accuracy_results = []
    full_df = pd.read_csv(df_path)
    full_df = full_df.drop(['id'], axis=1)
    train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=None) # , random_state=42)
    train_dataset = DataLoader(LinkPredictionDataset(train_df), batch_size=100, shuffle=True)
    test_dataset = DataLoader(LinkPredictionDataset(test_df), batch_size=100, shuffle=False)  # No need to shuffle test data

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    NeuralNet_model = NeuralNet().to(device)
    learning_rate = 0.001

    dict_loss_metrics = {
                    "BCE": nn.BCELoss(),
                    "CrossEntropyLoss": nn.CrossEntropyLoss(),
                    "MSE": nn.MSELoss(),
                    "SmoothL1Loss": nn.SmoothL1Loss(),
                    "BCELogits": nn.BCEWithLogitsLoss(),
                    }

    dict_optimizers = {
        "Adam": optim.Adam(NeuralNet_model.parameters(), lr=learning_rate),
        "AdamW": optim.AdamW(NeuralNet_model.parameters(), lr=learning_rate),
        "Adamax": optim.Adamax(NeuralNet_model.parameters(), lr=learning_rate), 
        "ASGD": optim.ASGD(NeuralNet_model.parameters(), lr=learning_rate),
        "Adagrad": optim.Adagrad(NeuralNet_model.parameters(), lr=learning_rate),
        "SGD": optim.SGD(NeuralNet_model.parameters(), lr=learning_rate),
        "SGDMomentum": optim.SGD(NeuralNet_model.parameters(), lr=learning_rate, momentum=0.9),
        "RMSprop": optim.RMSprop(NeuralNet_model.parameters(), lr=learning_rate),
        "Adadelta": optim.Adadelta(NeuralNet_model.parameters(), lr=1.0),
    }

    loss_metrics = []
    for metric_name in loss_metrics_to_test:
        loss_metrics.append(dict_loss_metrics[metric_name])

    optimizer_funcs = []
    for optimizer_name in optimizers_to_test:
        optimizer_funcs.append(dict_optimizers[optimizer_name])

    if retrain:
        for optimizer, optimizer_name in zip(optimizer_funcs, optimizers_to_test):
            print("\n" + "#"*100 + "\n" + "#"*100 + "\n")
            metric_number = 0
            for metric, metric_name in zip(loss_metrics, loss_metrics_to_test):
                metric_number += 1
                print("\n" + "#"*50 + "\n" + "#"*50 + "\n")

                ################## train for other types of loss and check their efficacy ##################
                print(f"{optimizer_name}\t{metric_name}\ttraining started...")
                total_train_start_time = time.time()
                time1 = time.time()
                NeuralNet_model.train()
                for epoch in range(0, num_epochs):
                    epoch_start_time = time.time()
                    for batch_num, (id1, id2, label) in enumerate(train_dataset):
                        id1, id2, label = id1.to(device), id2.to(device), label.to(device)
                        optimizer.zero_grad()
                        output = NeuralNet_model.forward(id1, id2)
                        float_label = label.float()
                        loss = metric(output, float_label)
                        loss.backward()

                        if full_run == False:
                            if batch_num == 1:
                                break # this will break out if you are just debugging print and file writes
                        
                        if batch_num % 1000 == 0:
                            time2 = time.time()
                            print(f"\nTrain_time = {(time2 - time1):.2f} sec")
                            time1 = time2
                            total_progress = (100/num_epochs)*(epoch + batch_num/len(train_dataset))
                            print(f"total retrain progress = {total_progress:.0f}%")
                            print(f"epoch {epoch+1}/{num_epochs} is {100*(batch_num/len(train_dataset)):.0f}% done.\tLoss: {loss.item():.6f}")
                    print("\n", "#"*15, f"\nepoch {epoch+1} train_time = {((time.time() - epoch_start_time)/60):.2f} min\n", "#"*15)
                print(f"Training ended... total_train_time = {((time.time() - total_train_start_time)/60):.2f} min")
                train_metrics, test_metrics = determine_train_test_metrics(model_=NeuralNet_model, train_dataset_=train_dataset, test_dataset_=test_dataset, device_=device)

                retrain_model_path = os.path.join(base_model_save_dir, f"{this_file_name[:-3]}_{num_epochs}_epochs_{optimizer_name}_{metric_number}_{metric_name}_{train_metrics['accuracy']}%_{test_metrics['accuracy']}%.pth")

                current_result = [num_epochs, optimizer_name, metric_number, metric_name, train_metrics, test_metrics]
                accuracy_results.append(current_result)
                torch.save(NeuralNet_model.state_dict(), retrain_model_path) # Save model to a file
                print(f"Trained model saved to {retrain_model_path}\n")

        print("#"*100, f"\nsorted_accuracy results:\n")
        # Sort the list of lists by the test_metrics['accuracy'] value (x[5]['accuracy'])
        sorted_accuracy_results = sorted(accuracy_results, key=lambda x: x[5]['accuracy'], reverse=True)
        total_print_string = f"full_run={full_run} - retrain={retrain}\n{this_file_name} - sorted based on test_accuracy metrics:\n\n"
        for num, i in enumerate(sorted_accuracy_results):
            print_string = f"{num+1}: num_epochs={i[0]}\toptimizer_name={i[1]}\tloss_metric={i[3]}\ntrain_metrics={i[4]}\ntest_metrics={i[5]}\n"
            print(f"{print_string}")
            total_print_string += f"{print_string}\n"
        print("#"*100, "\n", "#"*100, "\n")

        with open(output_results_save_file_name, "a") as file:
            file.write("\n" + "#"*100 + "\n" + total_print_string + f"main() run_time = {((time.time() - main_start_time)/60):.2f} min\n" + "#"*100)

    else: # load model from a file
        NeuralNet_model.load_state_dict(torch.load(load_model_path))
        print(f"Trained model loaded from {load_model_path}")
        train_metrics, test_metrics = determine_train_test_metrics(model_=NeuralNet_model, train_dataset_=train_dataset, test_dataset_=test_dataset, device_=device)

        with open(output_results_save_file_name, "a") as file:
            file.write("\n" + "#"*100 + f"\nretrain={str(retrain)}\n\n{this_file_name} - metrics:\ntrain_metrics={train_metrics}\ntest_metrics={test_metrics}\n" + f"\nmain() run_time = {((time.time() - main_start_time)/60):.2f} min\n" + "#"*100)

    print(f"\n\nmain() run_time = {((time.time() - main_start_time)/60):.2f} min\n\n")
