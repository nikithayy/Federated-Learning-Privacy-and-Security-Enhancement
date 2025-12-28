import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import flwr as fl
from derma import SimpleImageDataset
from medmnist import DermaMNIST
import numpy as np


class FLClient(fl.client.NumPyClient):
    """Federated Learning client using PyTorch and SimpleImageDataset."""

    def __init__(self, data_dir, model_path, device="cpu"):
        self.device = torch.device(device)
        self.model = self.load_model(model_path)
        self.train_loader, self.test_loader = self.load_data(data_dir)
        self.client_name = os.path.basename(os.path.normpath(data_dir))

    def get_properties(self, config):
        return {"client_name": self.client_name}

    def load_model(self, model_path):
        """
        Load and prepare ResNet-18 model for 28x28 RGB images.
        Accepts either a plain state_dict or a checkpoint containing "state_dict".
        """
        model = models.resnet18(weights=None)
        # Adapt conv1 and fc for 3-channel 28x28 input and 7 classes
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 7)  # 7 classes

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        state = torch.load(model_path, map_location=self.device)
        # If checkpoint contains nested state dict
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Allow loading if keys match (otherwise let it raise)
        model.load_state_dict(state)
        model.to(self.device)
        return model

    def load_data(self, data_dir):
        """
        Load and split dataset into training and testing DataLoaders.
        """
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        csv_path = os.path.join(data_dir, "labels.csv")
        img_dir = os.path.join(data_dir, "images")
        if not os.path.exists(csv_path) or not os.path.exists(img_dir):
            raise FileNotFoundError(f"CSV or image directory not found: {data_dir}")

        train_dataset = SimpleImageDataset(csv_path, root_dir=img_dir, transform=transform)
        # Use medmnist DermaMNIST for global test split
        test_dataset = DermaMNIST(root="Dataset_Derma/data_raw", split="test", download=True, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        return train_loader, test_loader

    def get_parameters(self, config):
        """
        Return model parameters as a list of NumPy arrays.
        Use .cpu().numpy().copy() to ensure they are contiguous and not tied to graph.
        """
        return [val.detach().cpu().numpy().copy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """
        Set model parameters from the provided list.
        Match each parameter to the state's key and preserve dtype.
        """
        state_dict = self.model.state_dict()
        if len(parameters) != len(state_dict):
            raise ValueError(f"Parameter length mismatch. Expected {len(state_dict)} arrays, got {len(parameters)}.")

        new_state = {}
        for (k, v_tensor), param in zip(state_dict.items(), parameters):
            # Convert incoming param (numpy) to tensor with proper dtype & device
            arr = np.array(param, copy=True)
            tensor = torch.from_numpy(arr).to(self.device)
            # Ensure dtype matches existing state tensor
            if tensor.dtype != v_tensor.dtype:
                tensor = tensor.to(v_tensor.dtype)
            new_state[k] = tensor
        self.model.load_state_dict(new_state)

    def _prepare_labels_for_loss(self, labels):
        """
        Ensure labels are shape (N,) and dtype long. Handles:
         - shape (N, 1) -> squeeze -> (N,)
         - one-hot (N, C) -> argmax -> (N,)
         - floats -> cast to long
        """
        # If numpy array, convert to torch tensor
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        # Move to CPU temporarily for shape ops (we convert dtype/device after)
        if labels.device != torch.device("cpu"):
            labels_cpu = labels.cpu()
        else:
            labels_cpu = labels

        # If one-hot vector (N, C), use argmax
        if labels_cpu.dim() == 2 and labels_cpu.size(1) > 1:
            labels_cpu = labels_cpu.argmax(dim=1)

        # If shape (N,1), squeeze
        if labels_cpu.dim() == 2 and labels_cpu.size(1) == 1:
            labels_cpu = labels_cpu.squeeze(1)

        # Now ensure 1D
        if labels_cpu.dim() == 0:
            # single scalar -> make 1D
            labels_cpu = labels_cpu.unsqueeze(0)

        # Cast to long (needed for CrossEntropyLoss)
        labels_cpu = labels_cpu.long()

        return labels_cpu.to(self.device)

    def fit(self, parameters, config):
        """
        Train the model on local data and return updated parameters.
        """
        self.set_parameters(parameters)
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        total_loss, correct, total_examples = 0.0, 0, 0
        num_epochs = 3

        for epoch in range(num_epochs):
            for imgs, labels in self.train_loader:
                imgs = imgs.to(self.device)
                labels = self._prepare_labels_for_loss(labels)

                optimizer.zero_grad()
                outputs = self.model(imgs)  # shape (N, C)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_examples += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / max(1, len(self.train_loader))
        accuracy = correct / max(1, total_examples) if total_examples > 0 else 0.0

        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        model_save_path = os.path.join(model_dir, f"{self.client_name}.pth")
        torch.save(self.model.state_dict(), model_save_path)

        return self.get_parameters(config), total_examples, {
            "loss": avg_loss,
            "accuracy": accuracy,
            "client_name": self.client_name
        }

    def evaluate(self, parameters, config):
        """
        Evaluate the model on the test dataset.
        """
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        loss_total, correct, total_examples = 0.0, 0, 0

        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs = imgs.to(self.device)
                labels = self._prepare_labels_for_loss(labels)

                outputs = self.model(imgs)
                loss = criterion(outputs, labels)
                loss_total += loss.item() * imgs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_examples += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = loss_total / max(1, total_examples)
        accuracy = correct / max(1, total_examples) if total_examples > 0 else 0.0
        # return float(avg_loss), int(total_examples), {
        #     "loss": float(avg_loss),
        #     "accuracy": float(accuracy),
        #     "client_name": self.client_name,
        # }
        return float(avg_loss), int(total_examples), {"accuracy": float(accuracy), "client_name": self.client_name}


def start_client_process(client_id, poisoned=False,
                         base_dir="D:/Masters/CSCI735-Found_Intell_Security_Sys/Project/Dataset_Derma/data_preprocessed",
                         model_path="initial_resnet18_pretrained.pth",
                         device="cpu"):
    """
    Start a Flower client process with the given configuration.
    """
    folder = "dermanist_poison_flip" if poisoned else "dermanist_clean"
    data_dir = os.path.join(base_dir, folder, f"Client_{client_id}")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    client = FLClient(data_dir, model_path, device=device)

    print(f"[Runner] Starting client: {client.client_name} (poisoned={poisoned})")

    fl.client.start_client(
        server_address="localhost:8080",
        client=client.to_client()
    )