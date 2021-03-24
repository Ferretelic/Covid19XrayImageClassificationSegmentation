import os

import torch
from albumentations import HorizontalFlip, Normalize, Resize, Compose, Blur, ShiftScaleRotate, Cutout
from albumentations.pytorch import ToTensorV2
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

class Covi19XrayImageDatasetClassification(torch.utils.data.Dataset):
  def __init__(self, dataset_path, image_size, mode):
    images, labels = [], []
    label_names = sorted(os.listdir(dataset_path))
    label2index = {label: index for index, label in enumerate(label_names)}

    for label_name in label_names:
      label_path = os.path.join(dataset_path, label_name)
      images += [os.path.join(label_path, image_name) for image_name in os.listdir(label_path)]
      labels += [label2index[label_name]] * len(images)

    random = np.random.RandomState(0)
    inaccuracies = random.permutation(np.arange(len(images)))
    train_inaccuracies, test_inaccuracies = train_test_split(inaccuracies, random_state=0, test_size=0.2)
    validation_inaccuracies, test_inaccuracies = train_test_split(test_inaccuracies, random_state=0, test_size=0.5)

    train_transform, validation_transform, test_transform = self.compose_transforms(image_size)

    if mode == "train":
      self.images, self.labels = np.array(images)[train_inaccuracies], np.array(labels)[train_inaccuracies]
      self.transform = train_transform

    elif mode == "validation":
      self.images, self.labels = np.array(images)[validation_inaccuracies], np.array(labels)[validation_inaccuracies]
      self.transform = validation_transform

    else:
      self.images, self.labels = np.array(images)[test_inaccuracies], np.array(labels)[test_inaccuracies]
      self.transform = test_transform

  def __getitem__(self, index):
    image = self.transform(image=np.asarray(Image.open(self.images[index]).convert("RGB")))["image"]
    label = torch.tensor(self.labels[index], dtype=torch.long)

    return image, label

  def __len__(self):
    return len(self.images)

  def compose_transforms(self, image_size):
    train_transform = Compose([
      Resize(*image_size, p=1.0),
      HorizontalFlip(p=0.5),
      Blur(p=0.5),
      ShiftScaleRotate(p=0.5),
      Cutout(max_h_size=20, max_w_size=20, p=0.5),
      Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], p=1.0),
      ToTensorV2()
    ])
    validation_transform = Compose([
      Resize(*image_size, p=1.0),
      Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], p=1.0),
      ToTensorV2()
    ])
    test_transform = Compose([
      Resize(*image_size, p=1.0),
      Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], p=1.0),
      ToTensorV2()
    ])

    return train_transform, validation_transform, test_transform

class Covi19XrayImageDatasetSegmentation(torch.utils.data.Dataset):
  def __init__(self, dataset_path, image_size, mode):
    image_path, mask_path = os.path.join(dataset_path, "images"), os.path.join(dataset_path, "masks")
    image_names = os.listdir(image_path)
    mask_names = os.listdir(mask_path)
    image_exists = ["mask_{}.png".format(os.path.splitext(image_name)[0]) in mask_names for image_name in image_names]

    images = [os.path.join(image_path, image_name) for image_name in image_names]
    masks = [os.path.join(mask_path, "mask_{}.png".format(os.path.splitext(image_name)[0])) for image_name in image_names]

    random = np.random.RandomState(0)
    inaccuracies = random.permutation(np.arange(len(images))[image_exists])
    train_inaccuracies, test_inaccuracies = train_test_split(inaccuracies, random_state=0, test_size=0.2)
    validation_inaccuracies, test_inaccuracies = train_test_split(test_inaccuracies, random_state=0, test_size=0.5)

    train_transform, validation_transform, test_transform = self.compose_transforms(image_size)

    if mode == "train":
      self.images, self.masks = np.array(images)[train_inaccuracies], np.array(masks)[train_inaccuracies]
      self.transform = train_transform

    elif mode == "validation":
      self.images, self.masks = np.array(images)[validation_inaccuracies], np.array(masks)[validation_inaccuracies]
      self.transform = validation_transform

    else:
      self.images, self.masks = np.array(images)[test_inaccuracies], np.array(masks)[test_inaccuracies]
      self.transform = test_transform

  def __getitem__(self, index):
    image = np.asarray(Image.open(self.images[index]).convert("RGB"))
    mask = np.asarray(Image.open(self.masks[index])) / 255.

    augmented = self.transform(image=image, mask=mask)
    image, mask = augmented["image"], augmented["mask"].unsqueeze(0).float()
    return image, mask

  def __len__(self):
    return len(self.images)

  def compose_transforms(self, image_size):
    train_transform = Compose([
      Resize(*image_size, p=1.0),
      HorizontalFlip(p=0.5),
      Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], p=1.0),
      ToTensorV2()
    ])
    validation_transform = Compose([
      Resize(*image_size, p=1.0),
      Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], p=1.0),
      ToTensorV2()
    ])
    test_transform = Compose([
      Resize(*image_size, p=1.0),
      Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], p=1.0),
      ToTensorV2()
    ])

    return train_transform, validation_transform, test_transform


def create_classification_dataloaders(dataset_path, image_size, mode, batch_size):
  dataloader = torch.utils.data.DataLoader(Covi19XrayImageDatasetClassification(dataset_path, image_size, mode), batch_size=batch_size)
  return dataloader

def create_segmentation_dataloaders(dataset_path, image_size, mode, batch_size):
  dataloader = torch.utils.data.DataLoader(Covi19XrayImageDatasetSegmentation(dataset_path, image_size, mode), batch_size=batch_size)
  return dataloader
