#!/usr/bin/python
"""
Run a MLP experiment from a yaml file

Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import torch
import torch.nn as nn
import numpy as np # type: ignore
import toupee


def FGSM(model: toupee.model.Model, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Fast Gradient Sign Method (FGSM) adversarial attack.

    Computes the signed gradient of the loss with respect to the input,
    which can be used to create adversarial perturbations.

    Args:
        model: The toupee Model to attack
        X: Input data as numpy array (NHWC format for images)
        Y: True labels as numpy array (one-hot encoded)

    Returns:
        Signed gradients as numpy array, same shape as X
    """
    # WARNING: this treats everything as a single batch!

    # Convert to tensors
    X_tensor = torch.from_numpy(X.astype(np.float32))
    Y_tensor = torch.from_numpy(Y.astype(np.float32))

    # Handle image data: NHWC -> NCHW
    is_image = X_tensor.dim() == 4
    if is_image:
        X_tensor = X_tensor.permute(0, 3, 1, 2)

    # Enable gradient computation for input
    X_tensor = X_tensor.to(model.device)
    X_tensor.requires_grad_(True)
    Y_tensor = Y_tensor.to(model.device)

    # Forward pass
    model._model.eval()
    prediction = model._model(X_tensor)

    # Compute loss
    # Handle CrossEntropyLoss which expects class indices
    if isinstance(model._loss, nn.CrossEntropyLoss):
        targets = Y_tensor.argmax(dim=1)
    else:
        targets = Y_tensor

    loss = model._loss(prediction, targets)

    # Backward pass to get gradients
    loss.backward()

    # Get signed gradient
    signed_grad = X_tensor.grad.sign()

    # Convert back to NHWC if needed
    if is_image:
        signed_grad = signed_grad.permute(0, 2, 3, 1)

    return signed_grad.cpu().numpy()


def PGD(model: toupee.model.Model, X: np.ndarray, Y: np.ndarray,
        epsilon: float = 0.3, alpha: float = 0.01, num_iter: int = 40,
        random_start: bool = True) -> np.ndarray:
    """
    Projected Gradient Descent (PGD) adversarial attack.

    A stronger iterative attack that applies FGSM multiple times
    with smaller step sizes, projecting back to the epsilon ball.

    Args:
        model: The toupee Model to attack
        X: Input data as numpy array (NHWC format for images)
        Y: True labels as numpy array (one-hot encoded)
        epsilon: Maximum perturbation magnitude
        alpha: Step size for each iteration
        num_iter: Number of iterations
        random_start: Whether to start with random perturbation

    Returns:
        Adversarial examples as numpy array, same shape as X
    """
    X_tensor = torch.from_numpy(X.astype(np.float32))
    Y_tensor = torch.from_numpy(Y.astype(np.float32))

    # Handle image data: NHWC -> NCHW
    is_image = X_tensor.dim() == 4
    if is_image:
        X_tensor = X_tensor.permute(0, 3, 1, 2)

    X_tensor = X_tensor.to(model.device)
    Y_tensor = Y_tensor.to(model.device)

    # Initialize adversarial example
    if random_start:
        delta = torch.empty_like(X_tensor).uniform_(-epsilon, epsilon)
        delta = torch.clamp(X_tensor + delta, 0, 1) - X_tensor
    else:
        delta = torch.zeros_like(X_tensor)

    delta.requires_grad_(True)

    model._model.eval()

    for _ in range(num_iter):
        # Forward pass
        prediction = model._model(X_tensor + delta)

        # Compute loss
        if isinstance(model._loss, nn.CrossEntropyLoss):
            targets = Y_tensor.argmax(dim=1)
        else:
            targets = Y_tensor

        loss = model._loss(prediction, targets)

        # Backward pass
        loss.backward()

        # Update delta
        grad_sign = delta.grad.sign()
        delta = delta + alpha * grad_sign
        delta = torch.clamp(delta, -epsilon, epsilon)
        delta = torch.clamp(X_tensor + delta, 0, 1) - X_tensor
        delta = delta.detach()
        delta.requires_grad_(True)

    # Get adversarial examples
    adv_X = X_tensor + delta

    # Convert back to NHWC if needed
    if is_image:
        adv_X = adv_X.permute(0, 2, 3, 1)

    return adv_X.detach().cpu().numpy()
