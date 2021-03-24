import os

import matplotlib.pyplot as plt
import seaborn as sns

def plot_classification_history(history, num_epochs, history_directory):
  train_losses, validation_losses = history["train_losses"], history["validation_losses"]
  train_accuracies, validation_accuracies = history["train_accuracies"], history["validation_accuracies"]

  plt.figure()
  plt.title("Loss")
  sns.lineplot(x=range(num_epochs), y=train_losses, legend="brief", label="train loss")
  sns.lineplot(x=range(num_epochs), y=validation_losses, legend="brief", label="validation loss")
  plt.xlabel("epoch")
  plt.ylabel("loss")
  plt.savefig(os.path.join("histories", history_directory, "loss.png"))

  plt.figure()
  plt.title("Accuracy")
  sns.lineplot(x=range(num_epochs), y=train_accuracies, legend="brief", label="train accuracy")
  sns.lineplot(x=range(num_epochs), y=validation_accuracies, legend="brief", label="validation accuracy")
  plt.xlabel("epoch")
  plt.ylabel("accuracy")
  plt.savefig(os.path.join("histories", history_directory, "accuracy.png"))


def plot_segmentation_history(history, num_epochs, history_directory):
  train_losses, validation_losses = history["train_losses"], history["validation_losses"]
  train_dices, validation_dices = history["train_dices"], history["validation_dices"]

  plt.figure()
  plt.title("Loss")
  sns.lineplot(x=range(num_epochs), y=train_losses, legend="brief", label="train loss")
  sns.lineplot(x=range(num_epochs), y=validation_losses, legend="brief", label="validation loss")
  plt.xlabel("epoch")
  plt.ylabel("loss")
  plt.savefig(os.path.join("histories", history_directory, "loss.png"))

  plt.figure()
  plt.title("Dice")
  sns.lineplot(x=range(num_epochs), y=train_dices, legend="brief", label="train dice")
  sns.lineplot(x=range(num_epochs), y=validation_dices, legend="brief", label="validation dice")
  plt.xlabel("epoch")
  plt.ylabel("dice")
  plt.savefig(os.path.join("histories", history_directory, "dice.png"))
