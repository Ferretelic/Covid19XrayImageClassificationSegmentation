import pickle
import os

from efficientnet_pytorch import EfficientNet
import torch

from dataset import create_classification_dataloaders
from utils import reset_directory, train_classification_model
from history import plot_classification_history

num_epochs = 20
mode = "classification"
device = torch.device("cuda")

reset_directory(mode)

dataset_path = "/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/Covid19XrayImageClassificationDataset"
labels = sorted(os.listdir(dataset_path))
image_size = (224, 224)
batch_size = 32
train_dataloader, validation_dataloader = create_classification_dataloaders(dataset_path, image_size, "train", batch_size), create_segmentation_dataloaders(dataset_path, image_size, "validation", batch_size)

model = EfficientNet.from_pretrained("efficientnet-b1")
model._fc = torch.nn.Linear(model._fc.in_features, len(labels))
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True)
criterion = torch.nn.CrossEntropyLoss()


history = train_classification_model(model, optimizer, criterion, scheduler, num_epochs, train_dataloader, validation_dataloader, device)
plot_classification_history(history, num_epochs, mode)

with open("./histories/history.pkl", "wb") as f:
  pickle.dump(history, f)
