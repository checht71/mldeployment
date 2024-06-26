"""This module saves a Keras model to BentoML."""

from pathlib import Path
import bentoml
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(13456, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        #print(x.size())
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.size())
        #torch.Size([BATCH_SIZE, 16, 29, 29])
        x = x.view(BATCH_SIZE, -1)
        #print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model_and_save_it_to_bento(model_file: Path) -> None:
    # PyTorch models inherit from torch.nn.Module
    """Loads a keras model from disk and saves it to BentoML."""
    # For loading the entire model 
    if torch.cuda.is_available():
        map_location = torch.device('cuda')
    else:
        map_location = torch.device('cpu')

    model = torch.load('full_model_final_86', map_location=map_location)
    bento_model = bentoml.pytorch.save_model("torch_model_86", model)
    print(f"Bento model tag = {bento_model.tag}")


if __name__ == "__main__":
    load_model_and_save_it_to_bento(Path("model"))
