import torch
torch.set_default_dtype(torch.float64)
print('PyTorch version:', torch.__version__)
torch.manual_seed(0)

import numpy as np
import matplotlib.pyplot as plt

# For data
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler



# --------------------------------------------------------------------------
# Load iris dataset
data = load_iris()
X = data['data']          # shape (150, 4)
y = data['target']        # classes: 0=setosa, 1=versicolor, 2=virginica

# Convert to binary classification: setosa (0) vs not-setosa (1)
y = (y != 0).astype(float)

# Standardize features
X = StandardScaler().fit_transform(X)

# Convert to torch tensors
X = torch.tensor(X, dtype=torch.get_default_dtype())
y = torch.tensor(y, dtype=torch.get_default_dtype())

n, d = X.shape

# Add bias column (intercept)
X = torch.cat([torch.ones(n, 1, dtype=X.dtype), X], dim=1)
d = d + 1
print(f"Tensors: X={tuple(X.shape)}, y={tuple(y.shape)}, d={d}")



# --------------------------------------------------------------------------
def logistic_loss(w, X, y):
    z = X @ w
    p = torch.sigmoid(z)
    eps = 1e-12
    nll = - (y * torch.log(p + eps) + (1 - y) * torch.log(1 - p + eps)).mean()
    return nll

# Initialize parameters
w = torch.nn.Parameter(torch.rand(X.shape[1], dtype=torch.get_default_dtype()), requires_grad=True)

# Quick sanity check
loss = logistic_loss(w, X, y)
print("Initial loss:", loss.item())



# --------------------------------------------------------------------------
# Exercice: Gradient & Hessian
def logistic_loss(w, X, y):
    z = X @ w
    p = torch.sigmoid(z)
    eps = 1e-12
    nll = -(y * torch.log(p + eps) + (1 - y) * torch.log(1 - p + eps)).mean()
    return nll

grad = torch.autograd.functional.grad(lambda w_: logistic_loss(w_, X, y), w)
H = torch.autograd.functional.hessian(lambda w_: logistic_loss(w_, X, y), w)

print("||grad||:", float(torch.norm(grad)))
print("H shape:", tuple(H.shape))



# --------------------------------------------------------------------------
#Exercice: Gradient Descent
n_epoch = 100
alpha = 1.0       # taux d'apprentissage
eps = 1e-2        # seuil de convergence pour la norme du gradient
loss_grad = []

# Réinitialisation des poids
w = torch.nn.Parameter(torch.rand(X.shape[1], dtype=torch.get_default_dtype()), requires_grad=True)

for epoch in range(n_epoch):
    # Calcul de la perte
    loss = logistic_loss(w, X, y)
    loss_grad.append(float(loss))
    
    # Calcul du gradient
    grad = torch.autograd.functional.grad(lambda w_: logistic_loss(w_, X, y), w)[0]
    
    # Mise à jour du vecteur poids w selon la règle GD
    with torch.no_grad():
        w -= alpha * grad
    
    # Critère d'arrêt basé sur la norme du gradient
    if torch.norm(grad) < eps:
        print(f"GD early stop at epoch {epoch}")
        break

print("Final loss:", loss_grad[-1])



# --------------------------------------------------------------------------
# Newton's method
n_epoch = 100
eps = 1e-2
loss_newton = []

# Réinitialiser w
w = torch.nn.Parameter(torch.rand(X.shape[1], dtype=torch.get_default_dtype()), requires_grad=True)

for epoch in range(n_epoch):
    # Calculer la perte
    loss = logistic_loss(w, X, y)
    loss_newton.append(float(loss))
    
    # Calculer gradient et Hessienne avec la fonction lambda
    grad = torch.autograd.functional.grad(lambda w_: logistic_loss(w_, X, y), w)[0]
    H = torch.autograd.functional.hessian(lambda w_: logistic_loss(w_, X, y), w)
    
    # Résoudre H * step = grad pour obtenir le pas de Newton
    step = torch.linalg.solve(H, grad)
    
    # Mise à jour de w avec le pas de Newton (sans trace dans autograd)
    with torch.no_grad():
        w -= step
    
    # Critère d'arrêt
    if torch.norm(grad) < eps:
        print(f"Newton early stop at epoch {epoch}")
        break
print("Final loss:", loss_newton[-1])



# --------------------------------------------------------------------------
# Compare convergence
plt.figure(figsize=(8, 5))
plt.plot(loss_grad, label='Gradient Descent')
plt.plot(loss_newton, label='Newton Method')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Comparison of Loss Convergence')
plt.legend()
plt.grid(True)
plt.show()