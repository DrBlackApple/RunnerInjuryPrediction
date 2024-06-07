import numpy as np


def sigmoid(x: np.ndarray, *, alpha=1, derivative=False):
    if derivative:
        return alpha * sigmoid(x) * (1 - sigmoid(x))
    else:
        return 1 / (1 + np.exp(-alpha * x))


def relu(x: np.ndarray, *, derivative=False):
    if derivative:
        return np.where(x > 0, 1, 0)
    else:
        return np.maximum(0, x)


def binary_crossentropy_loss(y_true, y_pred, *, epsilon=1e-8, derivative=False):
    # Vérification des dimensions

    if y_true.ndim != 1:
        y_true = y_true.flatten().reshape((1))
    if y_pred.ndim != 1:
        y_pred = y_pred.flatten().reshape((1))

    # Vérification de la forme
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    # Calcul de la fonction de perte
    # Pour éviter les valeurs nulles ou les dépassements de capacité
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    if derivative:
        return -np.mean(y_true / y_pred_clipped - (1 - y_true) / (1 - y_pred_clipped))
    else:
        return -np.mean(
            y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
        )


def glorot_uniform(input_dim: int, output_dim: int):
    """
    Initialise les poids d'une couche dense avec la distribution uniforme de Glorot.

    Args:
        input_dim (int): La dimension d'entrée de la couche.
        output_dim (int): La dimension de sortie de la couche.

    Returns:
        np.ndarray: Un tableau de forme (input_dim, output_dim) contenant les poids initialisés.
    """
    limit = np.sqrt(6 / (input_dim + output_dim))
    return np.random.uniform(low=-limit, high=limit, size=(input_dim, output_dim))
