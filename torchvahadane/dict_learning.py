"""
code directly adapted from https://github.com/rfeinman/pytorch-lasso
"""
import warnings
import torch

from .optimizers import coord_descent, ista
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from .eps import get_eps
# min_{D in C} = (1/n) sum_{i=1}^n (1/2)||x_i-Dalpha_i||_2^2 + lambda1||alpha_i||_1 + lambda1_2||alpha_i||_2^2


def lasso_loss(X, Z, weight, alpha=1.0):
    X_hat = torch.matmul(Z, weight.T)
    lambda2 = 10e-10
    lambda1 = 0.1
    loss = 0.5 * (X - X_hat).norm(p=2).pow(2) + weight.norm(p=1) * \
        lambda1 + lambda2 * weight.norm(p=2).pow(2)
    return loss.mean()


def dict_evaluate(X, weight, alpha, **kwargs):
    X = X.to(weight.device)
    Z = sparse_encode(X, weight, alpha, **kwargs)
    loss = lasso_loss(X, Z, weight, alpha)
    return loss


def dict_learning(X, n_components, alpha=1.0, constrained=True, persist=False,
                  lambd=1e-2, steps=60, device='cpu', progbar=True,
                  **solver_kwargs):
    n_samples, n_features = X.shape
    X = X.to(device)
    weight = torch.empty(n_features, n_components, device=device)
    nn.init.orthogonal_(weight)
    if constrained:
        weight = F.normalize(weight, dim=0)
    Z0 = None

    losses = torch.zeros(steps, device=device)
    with tqdm(total=steps, disable=not progbar) as progress_bar:
        for i in range(steps):
            # infer sparse coefficients and compute loss
            Z = sparse_encode(X, weight, alpha, Z0, **solver_kwargs)
            losses[i] = lasso_loss(X, Z, weight, alpha)
            if persist:
                Z0 = Z

            # update dictionary
            if constrained:
                weight = update_dict(weight, X, Z, positive=True)
            else:
                weight = update_dict_ridge(X, Z, lambd=lambd)

            # update progress bar
            progress_bar.set_postfix(loss=losses[i].item())
            progress_bar.update(1)

    return weight, losses


def update_dict(dictionary, X, Z, random_seed=None, positive=True,
                eps=1e-10):
    """Update the dense dictionary factor in place.

    Modified from `_update_dict` in sklearn.decomposition._dict_learning

    Parameters
    ----------
    dictionary : Tensor of shape (n_features, n_components)
        Value of the dictionary at the previous iteration.
    X : Tensor of shape (n_samples, n_features)
        Data matrix.
    code : Tensor of shape (n_samples, n_components)
        Sparse coding of the data against which to optimize the dictionary.
    random_seed : int
        Seed for randomly initializing the dictionary.
    positive : bool
        Whether to enforce positivity when finding the dictionary.
    eps : float
        Minimum vector norm before considering "degenerate"
    """
    n_components = dictionary.size(1)
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Residuals
    R = X - torch.matmul(Z, dictionary.T)  # (n_samples, n_features)
    for k in range(n_components):
        # Update k'th atom
        R += torch.outer(Z[:, k], dictionary[:, k])
        dictionary[:, k] = torch.matmul(Z[:, k], R)
        if positive:
            dictionary[:, k].clamp_(0, None)

        # Re-scale k'th atom
        atom_norm = dictionary[:, k].norm()
        if atom_norm < eps:
            # note that the random generated number can be all negative
            # and the clamp will get you a zero vector with zero norm --> 0/0 = nan
            dictionary[:, k].normal_()
            dictionary[:, k] = dictionary[:, k].abs()
            # might be
            if positive:
                # if all negative this line will create a zero vector
                dictionary[:, k].clamp_(0, None)
            # another layer of protection
            column_norm = dictionary[:, k].norm() + get_eps(dictionary)
            dictionary[:, k] /= column_norm
            # Set corresponding coefs to 0
            Z[:, k].zero_()  # TODO: is this necessary?
        else:
            dictionary[:, k] /= atom_norm
            R -= torch.outer(Z[:, k], dictionary[:, k])

    return dictionary


def update_dict_ridge(x, z, lambd=1e-4):
    """Update an (unconstrained) dictionary with ridge regression

    This is equivalent to a Newton step with the (L2-regularized) squared
    error objective:
        f(V) = (1/2N) * ||Vz - x||_2^2 + (lambd/2) * ||V||_2^2

    x : a batch of observations with shape (n_samples, n_features)
    z : a batch of code vectors with shape (n_samples, n_components)
    lambd : weight decay parameter
    """
    rhs = torch.mm(z.T, x)
    M = torch.mm(z.T, z)
    M.diagonal().add_(lambd * x.size(0))
    L = torch.linalg.cholesky(M)
    V = torch.cholesky_solve(rhs, L).T

    return V


_init_defaults = {
    'ista': 'zero',
    'cd': 'zero',
}


def ridge(b, A, alpha=1e-4):
    # right-hand side
    rhs = torch.matmul(A.T, b)
    # regularized gram matrix
    M = torch.matmul(A.T, A)
    M.diagonal().add_(alpha)
    # solve
    L, info = torch.linalg.cholesky_ex(M)
    if info != 0:
        raise RuntimeError("The Gram matrix is not positive definite. "
                           "Try increasing 'alpha'.")
    x = torch.cholesky_solve(rhs, L)
    return x


def initialize_code(x, weight, alpha, mode):
    n_samples = x.size(0)
    n_components = weight.size(1)
    if mode == 'zero':
        z0 = x.new_zeros(n_samples, n_components)
    elif mode == 'unif':
        z0 = x.new(n_samples, n_components).uniform_(-0.1, 0.1)
    elif mode == 'transpose':
        z0 = torch.matmul(x, weight)
    elif mode == 'ridge':
        z0 = ridge(x.T, weight, alpha=alpha).T
    else:
        raise ValueError("invalid init parameter '{}'.".format(mode))

    return z0


def sparse_encode(x, weight, alpha=0.1, z0=None, algorithm='ista', init=None,
                  **kwargs):
    n_samples = x.size(0)
    n_components = weight.size(1)

    # initialize code variable
    if z0 is not None:
        assert z0.shape == (n_samples, n_components)
    else:
        if init is None:
            init = _init_defaults.get(algorithm, 'zero')
        elif init == 'zero' and algorithm == 'iter-ridge':
            warnings.warn("Iterative Ridge should not be zero-initialized.")
        z0 = initialize_code(x, weight, alpha, mode=init)

    # perform inference
    if algorithm == 'cd':
        z = coord_descent(x, weight, z0, alpha, **kwargs)
    elif algorithm == 'ista':
        z = ista(x, z0, weight, alpha, **kwargs)
    else:
        raise ValueError("invalid algorithm parameter '{}'.".format(algorithm))

    return z
