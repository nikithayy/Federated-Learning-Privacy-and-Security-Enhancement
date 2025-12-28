import os
import json
import hashlib
import random
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import imageio
from skimage import restoration
from skimage.util import img_as_float
from skimage.color import rgba2rgb
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models


try:
    from medmnist import DermaMNIST
except Exception:
    DermaMNIST = None

# Directories
MEDMNIST_RAW = "../Dataset_Derma/data_raw"
PREPROCESS_DIR = "../Dataset_Derma/data_preprocessed"
CLEAN_OUTPUT_DIR = os.path.join(PREPROCESS_DIR, "dermanist_clean")
POISON_OUTPUT_DIR = os.path.join(PREPROCESS_DIR, "dermanist_poison_flip")

# Partitioning / poisoning defaults
NUM_CLIENTS = 20
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

POISONED_CLIENT_NUM = 4
FLIPPING_RATES = [0.1, 0.25, 0.5, 0.75]

# Training config for initial model
NUM_CLASSES = 7
IMAGE_SIZE = 28
INITIAL_MODEL_PRETRAINED = "initial_resnet18_pretrained.pth"
DEVICE = "cpu"

# -------------------------
# Utility functions
# -------------------------
def sha1_of_image_uint8(img_uint8: np.ndarray) -> str:
    """Computes a SHA-1 hash of an image in uint8 format."""
    return hashlib.sha1(img_uint8.tobytes()).hexdigest()

# -------------------------
# SimpleImageDataset is used by FL client/server code
# -------------------------
from torch.utils.data import Dataset
from PIL import Image as PILImage

class SimpleImageDataset(Dataset):
    """Custom PyTorch Dataset that loads images and labels from a CSV file."""
    def __init__(self, csv_file_or_df, root_dir=None, transform=None):
        if isinstance(csv_file_or_df, pd.DataFrame):
            self.df = csv_file_or_df.reset_index(drop=True)
        else:
            self.df = pd.read_csv(csv_file_or_df)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = str(row["filename"])
        if os.path.isabs(filename):
            img_path = filename
        else:
            img_path = os.path.join(self.root_dir, filename) if self.root_dir else filename
        img = PILImage.open(img_path).convert("RGB")
        label = int(row["label"])
        if self.transform:
            img = self.transform(img)
        return img, label

# -------------------------
# Create / save initial model (untrained architecture, saved weights)
# -------------------------
def create_and_save_initial_model(save_path: str = INITIAL_MODEL_PRETRAINED, num_classes: int = NUM_CLASSES, device: str = DEVICE):
    """Initializes a ResNet-18 model adapted for 28x28 image inputs."""
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    torch.save(model.state_dict(), save_path)
    print(f"Saved initial model state_dict to: {save_path}")
    return save_path

# -------------------------
# Data preparation (clean and poisoned)
# -------------------------
def prepare_derma_data(
    medmnist_raw: str = MEDMNIST_RAW,
    preprocess_dir: str = PREPROCESS_DIR,
    clean_output_dir: str = CLEAN_OUTPUT_DIR,
    poisoned_output_dir: str = POISON_OUTPUT_DIR,
    num_clients: int = NUM_CLIENTS,
    poisoned_client_num: int = POISONED_CLIENT_NUM,
    flipping_rates: List[float] = FLIPPING_RATES,
    seed: int = RANDOM_SEED,
    image_size: int = IMAGE_SIZE
) -> Tuple[str, str]:
    """Prepare data and return (clean_output_dir, poisoned_output_dir)."""
    if DermaMNIST is None:
        raise RuntimeError("medmnist DermaMNIST not available. Install medmnist package.")

    os.makedirs(medmnist_raw, exist_ok=True)
    os.makedirs(preprocess_dir, exist_ok=True)
    os.makedirs(clean_output_dir, exist_ok=True)
    os.makedirs(poisoned_output_dir, exist_ok=True)

    rng = np.random.RandomState(seed)
    med_train = DermaMNIST(root=medmnist_raw, split="train", download=True, size=image_size)

    # Deduplicate + denoise
    seen_hash = set()
    processed = []
    for i in range(len(med_train)):
        img_pil, label = med_train[i]
        label = int(label.item())
        img = np.array(img_pil)

        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 4:
            img = (rgba2rgb(img / 255.0) * 255).astype(np.uint8)

        img_float = img_as_float(img)
        img_uint8 = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)

        img_hash = sha1_of_image_uint8(img_uint8)
        if img_hash in seen_hash:
            continue
        seen_hash.add(img_hash)

        try:
            denoise = restoration.denoise_bilateral(img_float, channel_axis=-1)
        except Exception:
            denoise = img_float
        denoise_uint8 = (np.clip(denoise, 0, 1) * 255).astype(np.uint8)

        processed.append({
            "image": denoise_uint8,
            "label": label,
            "orig_index": i
        })

    # Build index by label
    labels = np.array([r["label"] for r in processed])
    num_classes_in_data = labels.max() + 1
    idx_by_label = [np.where(labels == cid)[0].tolist() for cid in range(num_classes_in_data)]

    # Dirichlet partitioning
    client_index = [[] for _ in range(num_clients)]
    for class_id, idxs in enumerate(idx_by_label):
        if len(idxs) == 0:
            continue
        rng.shuffle(idxs)
        proportions = rng.dirichlet([0.5] * num_clients)
        splits = (np.cumsum(proportions) * len(idxs)).astype(int)
        prev = 0
        for cid, cut in enumerate(splits):
            client_index[cid].extend(idxs[prev:cut])
            prev = cut
        client_index[-1].extend(idxs[prev:])

    client_data = {cid: [processed[i] for i in client_index[cid]] for cid in range(num_clients)}

    # Augmentation transform for saving
    save_transform = transforms.Compose([   
        transforms.ToPILImage(),     
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.2,0.2,0.2)], p=0.5),
        transforms.ToTensor()
    ])

    # Save clean per-client datasets
    total_images = 0
    for cid, recs in client_data.items():
        client_dir = os.path.join(clean_output_dir, f"Client_{cid}")
        img_dir = os.path.join(client_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        records = []
        for idx, rec in enumerate(recs):
            img = rec["image"]
            label = rec["label"]
            img_tensor = save_transform(img)
            img_aug = (img_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            filename = f"{idx}.png"
            filepath = os.path.join(img_dir, filename)
            imageio.imwrite(filepath, img_aug)
            records.append({"filename": filename, "label": int(label)})
            total_images += 1
        pd.DataFrame(records).to_csv(os.path.join(client_dir, "labels.csv"), index=False)

    # Create poisoned datasets by label flipping
    poisoned_client_ids = rng.choice(num_clients, poisoned_client_num, replace=False).tolist()
    client_flip_map = dict(zip(poisoned_client_ids, flipping_rates[:poisoned_client_num]))
    corruption_summary = {}
    client_data_corrupted = {}

    for cid, recs in client_data.items():
        recs_copy = [dict(r) for r in recs]
        flipped_count = 0
        total = len(recs_copy)
        if cid in client_flip_map:
            flip = client_flip_map[cid]
            n = len(recs_copy)
            num_to_flip = int(round(flip * n))
            if num_to_flip > 0:
                flip_idx = rng.choice(n, num_to_flip, replace=False)
                for idx in flip_idx:
                    orig = recs_copy[idx]["label"]
                    choices = [l for l in range(num_classes_in_data) if l != orig]
                    new_label = int(rng.choice(choices))
                    recs_copy[idx]["label"] = new_label
                    flipped_count += 1
        client_data_corrupted[cid] = recs_copy
        corruption_summary[cid] = {
            "total": total,
            "flipped": flipped_count,
            "percent": 100 * flipped_count / total if total > 0 else 0
        }

    # Save poisoned clients images and labels
    total_poisoned_images = 0
    for cid, recs in client_data_corrupted.items():
        client_dir = os.path.join(poisoned_output_dir, f"Client_{cid}")
        img_dir = os.path.join(client_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        records = []
        for idx, rec in enumerate(recs):
            img = rec["image"]
            label = rec["label"]
            filename = f"{idx}.png"
            filepath = os.path.join(img_dir, filename)
            imageio.imwrite(filepath, img)
            records.append({"filename": filename, "label": int(label)})
            total_poisoned_images += 1
        pd.DataFrame(records).to_csv(os.path.join(client_dir, "labels.csv"), index=False)

    # Summary metadata
    meta = {
        "poisoned_client_ids": poisoned_client_ids,
        "client_flip_map": client_flip_map,
        "seed": seed,
        "corruption_summary": corruption_summary
    }
    with open(os.path.join(poisoned_output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Prepared data. Clean images: {total_images}. Poisoned images: {total_poisoned_images}. Poisoned clients: {poisoned_client_ids}")
    return clean_output_dir, poisoned_output_dir


if __name__ == "__main__":
    """ Entry point of this script"""
    os.makedirs(PREPROCESS_DIR, exist_ok=True)
    try:
        clean_dir, poison_dir = prepare_derma_data()
    except RuntimeError as e:
        print("Error preparing derma data:", e)
    create_and_save_initial_model()
