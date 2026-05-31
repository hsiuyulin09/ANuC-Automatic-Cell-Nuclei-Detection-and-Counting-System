import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import BCEDiceLoss, H5Dataset, UNet, set_device

DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 60
DEFAULT_MODEL_SEED = 77
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_THRESHOLD = 0.5

def train_model(config):
    paths = config["paths"]
    train_config = config.get("training") or {}

    h5_path = Path(paths["dataset_output"])
    model_output = Path(paths["model_output"])

    batch_size = train_config.get("batch_size")
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE

    epochs = train_config.get("epochs")
    if epochs is None:
        epochs = DEFAULT_EPOCHS

    seed = train_config.get("seed")
    if seed is None:
        seed = DEFAULT_MODEL_SEED

    learning_rate = train_config.get("learning_rate")
    if learning_rate is None:
        learning_rate = DEFAULT_LEARNING_RATE

    threshold = train_config.get("threshold")
    if threshold is None:
        threshold = DEFAULT_THRESHOLD

    if not h5_path.exists():
        raise FileNotFoundError(f"h5 dataset not exist: {h5_path}")

    model_output.parent.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 固定weight initiation的random
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = set_device()
    print(f"using device. {device}")

    train_dataset = H5Dataset(h5_path=h5_path, mode="train", augment=True, seed=seed)
    val_dataset = H5Dataset(h5_path=h5_path, mode="val", augment=False, seed=seed)

    if len(train_dataset) == 0:
        print("training dataset is empty")
        return

    if len(val_dataset) == 0:
        print("validation dataset is empty")
        return

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model = UNet().to(device)
    loss_function = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam 依權重過去更新狀況用一個參數微調當次 learning rate

    loss_history = []

    for epoch in range(epochs):
        train_loader.dataset.set_epoch(epoch) # train_loader.dataset指向H5Dataset #呼叫set_epoch(epoch)

        model.train()
        train_loss = 0.0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = loss_function(output, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct_pixels = 0
        total_pixels = 0
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                output = model(img)
                loss = loss_function(output, mask)
                val_loss += loss.item()

                prediction = torch.sigmoid(output) >= threshold
                correct_pixels += (prediction == mask).sum().item()
                total_pixels += torch.numel(prediction) # .numel() number of element

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = correct_pixels / total_pixels
        loss_history.append(avg_train_loss)

        if (epoch + 1) % 1 == 0 or epoch == 0:
            print(f"Loss after iteration {epoch + 1}: {avg_train_loss}")

    print("Final statistics")
    print(f"Final training loss: {avg_train_loss}")
    print(f"Final validation loss: {avg_val_loss}")
    print(f"Final validation accuracy: {avg_val_acc * 100} %")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, label="Training Loss", color="#1f42b4", linewidth=2, marker="o", markersize=4)
    plt.title("Learning Curve", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.legend()
    plt.tight_layout()
    plt.show()

    torch.save(model.state_dict(), str(model_output))
    print(f"model weight saved at {model_output}")
