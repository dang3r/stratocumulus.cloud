import random
from collections import Counter
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


class StratocumulusDataset(Dataset):
    """
    Binary classification dataset for Stratocumulus cloud detection.

    Labels:
        1 = Stratocumulus (Sc)
        0 = All other cloud types
    """

    def __init__(self, data_dir, transform=None):
        """
        Initialize the dataset by collecting all image paths and their labels.

        Args:
            data_dir (str): Path to CCSN_v2 directory containing cloud type folders
            transform: Optional torchvision transforms to apply to images

        TODO: Implement this method
        Hints:
            - Use Path(data_dir).glob() to find all .jpg files
            - For each image, determine if it's in the 'Sc' folder (label=1) or not (label=0)
            - Store image paths and corresponding labels in lists
            - Save the transform for later use
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.img_to_label = dict()

        for image_path in self.data_dir.rglob("*.jpg"):
            terms = ["sc", "stratocumulus"]
            is_sc = any(term in str(image_path).lower() for term in terms)
            self.img_to_label[image_path] = int(is_sc)

        self.imgs = list(self.img_to_label.keys())
        self.labels = list(self.img_to_label.values())

    def __len__(self):
        """
        Return the total number of images in the dataset.

        TODO: Implement this method
        Hint: Return the length of your image paths list
        """
        return len(self.img_to_label)

    def __getitem__(self, idx) -> Tuple[Image.Image, int]:
        """
        Load and return a single image and its label.

        Args:
            idx (int): Index of the sample to load

        Returns:
            tuple: (image, label) where image is a tensor and label is 0 or 1

        TODO: Implement this method
        Hints:
            - Get the image path at index idx
            - Load the image using PIL.Image.open()
            - Convert to RGB (in case of grayscale)
            - Apply self.transform if it exists
            - Get the corresponding label
            - Return (image, label)
        """
        img_path = self.imgs[idx]
        label = self.img_to_label[img_path]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


train_transforms = transforms.Compose(
    [
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
val_transforms = transforms.Compose(
    [
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def stratified_split(dataset, val_split=0.2, seed=42):
    sc = [idx for idx in range(len(dataset)) if dataset.labels[idx] == 1]
    nsc = [idx for idx in range(len(dataset)) if dataset.labels[idx] != 1]

    random.seed(seed)
    random.shuffle(sc)
    random.shuffle(nsc)

    sc_split = int(len(sc) * val_split)
    nsc_split = int(len(nsc) * val_split)

    train_indices = sc[sc_split:] + nsc[nsc_split:]
    val_indices = sc[:sc_split] + nsc[:nsc_split]

    return train_indices, val_indices


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    prec = 0.0
    spec = 0.0
    rec = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        tf_labels = labels.reshape(labels.shape[0], 1).float()
        results = model(images)
        loss = criterion(results, tf_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predictions = (torch.sigmoid(results) > 0.5).float()
        total += tf_labels.size(0)
        correct += (predictions == tf_labels).sum().item()
        ntp, ntn, nfp, nfn = confusion_matrix(predictions, labels)
        tp += ntp
        tn += ntn
        fp += nfp
        fn += nfn
        prec += precision(ntp, nfp)
        rec += recall(ntp, nfn)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    print(f"Epoch loss: {epoch_loss}, Epoch accuracy: {epoch_acc}")
    return epoch_loss, epoch_acc


def confusion_matrix(predictions, labels):
    """
    Compute the confusion matrix for binary classification.

    Args:
        predictions: Binary predictions (0 or 1), shape (N,) or (N, 1)
        labels: Ground truth labels (0 or 1), shape (N,) or (N, 1)

    Returns:
        tuple: (TP, TN, FP, FN) as integers

    TODO: Implement this function
    Hints:
        - First ensure predictions and labels have the same shape (flatten if needed)
        - TP: Count where (labels == 1) AND (predictions == 1)
        - TN: Count where (labels == 0) AND (predictions == 0)
        - FP: Count where (labels == 0) AND (predictions == 1)
        - FN: Count where (labels == 1) AND (predictions == 0)
        - Use .sum().item() to convert torch tensor counts to Python integers
        - Remember: Sc (class 1) is the positive class
    """
    predictions = predictions.flatten()
    labels = labels.flatten()

    tp = ((predictions == 1) & (labels == 1)).sum()
    tn = ((predictions == 0) & (labels == 0)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()
    return tp.item(), tn.item(), fp.item(), fn.item()


def precision(tp, fp):
    """
    Compute precision: Of all positive predictions, how many were correct?

    Args:
        tp: True positives count
        fp: False positives count

    Returns:
        float: Precision score between 0 and 1, or 0.0 if undefined

    TODO: Implement this function
    Hints:
        - Formula: TP / (TP + FP)
        - Handle division by zero: if (TP + FP) == 0, return 0.0
    """
    return tp / (tp + fp) if (tp + fp) != 0 else 0.0


def recall(tp, fn):
    """
    Compute recall (sensitivity): Of all actual positives, how many did we find?

    Args:
        tp: True positives count
        fn: False negatives count

    Returns:
        float: Recall score between 0 and 1, or 0.0 if undefined

    TODO: Implement this function
    Hints:
        - Formula: TP / (TP + FN)
        - Handle division by zero: if (TP + FN) == 0, return 0.0
    """
    return tp / (tp + fn) if (tp + fn) != 0 else 0.0


def specificity(tn, fp):
    """
    Compute specificity: Of all actual negatives, how many did we correctly reject?

    Args:
        tn: True negatives count
        fp: False positives count

    Returns:
        float: Specificity score between 0 and 1, or 0.0 if undefined

    TODO: Implement this function
    Hints:
        - Formula: TN / (TN + FP)
        - Handle division by zero: if (TN + FP) == 0, return 0.0
    """
    return tn / (tn + fp) if (tn + fp) != 0 else 0.0


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            tf_labels = labels.reshape(labels.shape[0], 1).float()
            results = model(images)
            loss = criterion(results, tf_labels)
            running_loss += loss.item()
            predictions = (torch.sigmoid(results) > 0.5).float()
            total += tf_labels.size(0)
            correct += (predictions == tf_labels).sum().item()
            tp, tn, fp, fn = confusion_matrix(predictions, labels)
            print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def get_model():
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[-1].in_features, 1),
    )
    return model


def dataloaders() -> Tuple[DataLoader, DataLoader]:
    dataset = StratocumulusDataset("clouds_train")
    print(f"Total number of images: {len(dataset)}")

    #  sample image
    img, label = dataset[0]
    print(f"Image shape: {img.size}, Label: {label}")

    # check class distribution

    labels = [label for _, label in dataset]
    label_counts = Counter(labels)
    print(f"Class distribution: {label_counts}")
    print("Stratocumulus distribution", label_counts[1])
    print("Non-stratocumulus distribution", label_counts[0])

    # Stratified split
    train_idx, val_idx = stratified_split(dataset, val_split=0.2, seed=42)

    print(f"\nTrain/Val Split:")
    print(f"  Training samples: {len(train_idx)}")
    print(f"  Validation samples: {len(val_idx)}")

    train_labels = [dataset.labels[idx] for idx in train_idx]
    val_labels = [dataset.labels[idx] for idx in val_idx]

    train_dist = Counter(train_labels)
    val_dist = Counter(val_labels)

    print(f"\nTraining set distribution:")
    print(f"  Sc: {train_dist[1]}, Other: {train_dist[0]}")
    print(f"  Ratio: 1:{train_dist[0] / train_dist[1]:.2f}")

    print(f"\nValidation set distribution:")
    print(f"  Sc: {val_dist[1]}, Other: {val_dist[0]}")
    print(f"  Ratio: 1:{val_dist[0] / val_dist[1]:.2f}")

    train_dataset = StratocumulusDataset("clouds_train", transform=train_transforms)
    val_dataset = StratocumulusDataset("clouds_train", transform=val_transforms)
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4)

    return train_loader, val_loader


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    train_loader, val_loader = dataloaders()
    print("Dataloader info")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    images, labels = next(iter(train_loader))
    print(f"Image shape: {images.shape}, Label shape: {labels.shape}")
    print(f"Label values: {labels.tolist()}")

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6.48], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    patience = 10
    best_val_acc = 0.0
    best_epoch = 0

    for e in range(30):
        print(f"Epo{e}")
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        loss, acc = validate(model, val_loader, criterion, device)
        if acc > best_val_acc:
            best_val_acc = acc
            best_epoch = e
            torch.save(model.state_dict(), f"best_model_{best_epoch}.pth")
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping")
                break
        print(f"Validation loss: {loss:.4f}, Validation accuracy: {acc:.4f}")


def infer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    model.to(device)

    ds = StratocumulusDataset(
        Path("clouds_test") / "stratocumulus clouds", transform=val_transforms
    )
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    correct_predictions = 0
    total_predictions = 0
    all_predictions = []
    all_labels = []
    for image, label in dl:
        image = image.to(device)
        label = label.to(device)
        with torch.no_grad():
            outputs = model(image)
            sig = torch.sigmoid(outputs)
            predictions = sig > 0.5
            print(sig, predictions)
            print(f"Predictions: {predictions}, Labels: {label}")
            correct_predictions += predictions.sum().item()
            total_predictions += predictions.numel()
            all_predictions.extend(predictions)
            all_labels.extend(label)

    # convert to tensors
    all_predictions = torch.tensor(all_predictions)
    all_labels = torch.tensor(all_labels)

    tp, tn, fp, fn = confusion_matrix(all_predictions, all_labels)
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Correct/total: {correct_predictions}/{total_predictions}")


if __name__ == "__main__":
    import sys

    if sys.argv[1] == "test":
        infer()
    else:
        train()
