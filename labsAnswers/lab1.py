import torch
print('PyTorch version:', torch.__version__)
torch.manual_seed(0)


# --------------------------------------------------------------------------
# Create basic tensors
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

print('x =', x)
print('y =', y)
z = x + y
print('z = x + y =', z)
print('z.shape =', z.shape)
print(" ")
print(" ")



# --------------------------------------------------------------------------
# Exercice 1.1
print("--------- Exercice 1.1 ---------")
A = torch.randn(3, 3)
A_T = A.t()

assert A.shape == (3, 3)
assert A_T.shape == (3, 3)
print('A =\n', A)
print('A_T =\n', A_T)
print(" ")
print(" ")



# --------------------------------------------------------------------------
# Exercice 2.1
print("--------- Exercice 2.1 ---------")
x = torch.tensor([4.0], requires_grad=True)
f = x**2
print('f(x) =', f.item())

x = torch.tensor([2.0], requires_grad=True)
f = 3 * x**3 + 2 * x**2 + 5
print('f(2) =', f.item())
print(" ")
print(" ")



# --------------------------------------------------------------------------
# Exercice 3.1
print("--------- Exercice 3.1 ---------")
x = torch.tensor([4.0], requires_grad=True)
f = x**2
f.backward()
print('df/dx at x=4 =', x.grad.item())

x = torch.tensor([2.0], requires_grad=True)
f = 3 * x**3 + 2 * x**2 + 5
f.backward()
print('df/dx at x=2 =', x.grad.item())
assert abs(x.grad.item() - 44.0) < 1e-5
print(" ")
print(" ")



# --------------------------------------------------------------------------
# Exercice 4.1
print("--------- Exercice 4.1 ---------")
x = torch.tensor([2.0], requires_grad=True)
f1 = 3*x**3 + 2*x**2 + 5
f1.backward()
print('After first backward, grad =', x.grad.item())

x.grad.zero_()
f2 = (x + 1)**2
f2.backward()
print('After zero + second backward, grad =', x.grad.item())

x = torch.tensor([2.0], requires_grad=True)
f = (x + 1)**2
f.backward()
print('grad after first backward =', x.grad.item())

# Appel backward() une deuxième fois sans remettre à zéro les gradients
f = (x + 1)**2
f.backward()
print('grad after second backward (no zero_) =', x.grad.item())

print(" ")
print(" ")
print("Commentaire d'observation:")
print("Les gradients s'accumulent naturellement dans x.grad.")
print("Après deux backward(), le gradient affiché est le double de celui après un seul backward.")