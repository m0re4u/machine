import os
import torch
import argparse

import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from machine.util.agent import ModelAgent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiagnosticDataset(Dataset):
    def __init__(self, dirname):
        files = [f for f in os.listdir(
            dirname) if os.path.isfile(os.path.join(dirname, f))]
        self.embeddings = []
        self.labels = []
        for file in files:
            data = np.load(os.path.join(dirname, file))
            self.embeddings.extend(data[:, 0])
            self.labels.extend(data[:, 1])

    def __getitem__(self, index):
        X = torch.Tensor(self.embeddings[index])
        y = torch.Tensor([self.labels[index]])
        return X, y

    def __len__(self):
        return len(self.embeddings)

def validation_eval(val_set, model):
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True, num_workers=0)
    correct = 0
    for n, batch in enumerate(val_loader):
        data_in = batch[0].to(device)
        label = batch[1].type(torch.long).to(device)
        _, output = model(data_in).max(1)
        local_acc = sum(label.flatten() == output)
        correct += local_acc.item()
    return correct / len(val_set)

def main(args):
    # Create dataset
    dataset = DiagnosticDataset(args.data_dir)
    # Split data
    split = int(np.floor(0.2 * len(dataset)))
    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset) - split, split])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)

    model = torch.nn.Sequential(
        torch.nn.Linear(128, 2),
        torch.nn.ReLU(),
        torch.nn.Softmax(dim=1)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.NLLLoss()
    for i in range(5):
        for n, batch in enumerate(train_loader):
            data_in = batch[0].to(device)
            label = batch[1].type(torch.long).to(device)
            optimizer.zero_grad()
            output = model(data_in)
            loss = criterion(output, label.flatten())
            loss.backward()
            optimizer.step()
            if n % 200 == 0:
                val_acc = validation_eval(val_set, model)
                train_acc = validation_eval(train_set, model)
                print(f"Epoch: {i:2} - Step {n:4} - Loss: {loss:1.03} - TAcc: {train_acc:1.04} - VAcc: {val_acc:1.04}")

    torch.save(model.state_dict(), 'trained_diag.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/reason_dataset/", type=str,
                        help="Directory pointing to training data (REQUIRED)")

    args = parser.parse_args()
    main(args)
