import pickle

import torch
import segmentation_models_pytorch

from dataset import create_segmentation_dataloaders
from utils import reset_directory, train_segmentation_model
from history import plot_segmentation_history

num_epochs = 50
mode = "segmentation"
device = torch.device("cuda")

reset_directory(mode)

dataset_path = "/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/Covid19XrayImageSegmentationDataset"
image_size = (224, 224)
batch_size = 16
train_dataloader, validation_dataloader = create_segmentation_dataloaders(dataset_path, image_size, "train", batch_size), create_segmentation_dataloaders(dataset_path, image_size, "validation", batch_size)

model = segmentation_models_pytorch.Unet("resnet101", encoder_weights="imagenet", classes=1, activation=None).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True)

history = train_segmentation_model(model, criterion, optimizer, scheduler, num_epochs, train_dataloader, validation_dataloader, device)
plot_segmentation_history(history, num_epochs, mode)

with open("./histories/segmentation/history.pkl", "wb") as f:
  pickle.dump(history, f)
