import torch
from PIL import Image
import segmentation_models_pytorch
import matplotlib.pyplot as plt
import numpy as np

from dataset import Covi19XrayImageDatasetSegmentation

num_images = 3

dataset_path = "/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/Covid19XrayImageSegmentationDataset"
image_size = (224, 224)
test_dataset = Covi19XrayImageDatasetSegmentation(dataset_path, image_size, "test")

device = torch.device("cuda")
model = segmentation_models_pytorch.Unet("resnet101", encoder_weights=None, classes=1, activation=None).to(device)
model.load_state_dict(torch.load("./model_archives/model_resnet101_17.pth"))

figure, axes = plt.subplots(num_images, 2, figsize=(5, 7.5))
sample_image_index = np.random.choice(np.arange(len(test_dataset)), num_images)

for index, ((ax1, ax2), image_index) in enumerate(zip(axes, sample_image_index)):
  image = np.asarray(Image.open(test_dataset.images[image_index]).resize(image_size).convert("RGB"))
  true_mask = np.asarray(Image.open(test_dataset.masks[image_index]).resize(image_size))
  output = model(test_dataset[image_index][0].unsqueeze(0).to(device))
  predicted_mask = torch.sigmoid(output).cpu().squeeze().detach().numpy()

  ax1.imshow(image)
  ax1.imshow(true_mask, cmap="gray", alpha=0.6)
  ax1.set_xticks([])
  ax1.set_yticks([])
  ax2.imshow(image)
  ax2.imshow(predicted_mask, cmap="gray", alpha=0.6)
  ax2.set_xticks([])
  ax2.set_yticks([])

  if index == 0:
    ax1.set_title("True Mask")
    ax2.set_title("Predicted Mask")

figure.tight_layout()
figure.savefig("./images/predicted_sample_resnet101.png")
