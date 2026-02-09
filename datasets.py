import torch
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms


def get_iris(test_size=0.2, random_state=42):
    """Carga el dataset Iris, aplica normalización y one-hot encoding.
    Devuelve los datos en forma de tensores de PyTorch."""

    # Cargar dataset Iris
    iris = load_iris()
    X = iris.data  # Características
    y = iris.target.reshape(-1, 1)  # Etiquetas (reshape para OneHotEncoder)

    # One-hot encoding de las etiquetas
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)

    # Normalizar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=test_size,
                                                        random_state=random_state)

    # Convertir a tensores de PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    return (X_train_tensor, y_train_tensor), (X_test_tensor, y_test_tensor)


# Transformaciones para MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def get_mnist():
    """Carga el dataset MNIST con transformaciones."""
    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_ds, test_ds
