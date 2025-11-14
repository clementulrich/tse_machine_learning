import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
print('PyTorch version:', torch.__version__)


# --------------------------------------------------------------------------
# Exercice 1.1
print("--------- Exercice 1.1 ---------")
# Charger le dataset d'entraînement avec la transformation ToTensor()
train_data = datasets.FashionMNIST(
    root='data/',       # dossier local pour stocker les données
    train=True,         # jeu d'entraînement
    download=True,      # télécharge si non présent
    transform=ToTensor()  # transformation des images en tenseurs
)

# Charger le dataset de test de la même façon
test_data = datasets.FashionMNIST(
    root='data/',
    train=False,        # jeu de test
    download=True,
    transform=ToTensor()
)

print(f"Train set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")
print(" ")
print(" ")



# --------------------------------------------------------------------------
# Exercice 1.2
print("--------- Exercice 1.2 ---------")
# Récupérer une image et son label depuis le dataset d'entraînement
image, label = train_data[0]  # premier échantillon

print(f"Label: {label}")
print(f"Image shape: {image.shape}")  # forme tensorielle, normalement [1, 28, 28]
print(f"Valeurs min/max pixel: {image.min().item()}/{image.max().item()}")  # valeurs normalisées

# Afficher l'image (tensor CxHxW, ici C=1)
plt.imshow(image.squeeze(), cmap='gray')  # retirer la dimension canal pour l'affichage
plt.title(f"Label: {label}")
plt.axis('off')
plt.show()
print(" ")
print(" ")



# --------------------------------------------------------------------------
# Exercice 2.1
print("--------- Exercice 2.1 ---------")
# Création du DataLoader pour le jeu d'entraînement
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Création du DataLoader pour le jeu de test
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

print(f"Train loader batches: {len(train_loader)}")
print(f"Test loader batches: {len(test_loader)}")
print(" ")
print(" ")



# --------------------------------------------------------------------------
# Exercice 3.1
print("--------- Exercice 3.1 ---------")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(" ")
print(" ")



# --------------------------------------------------------------------------
# Exercice 4.1
print("--------- Exercice 4.1 ---------")
# TODO: Implement the autoencoder
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Aplatir l'entrée (images 28x28) en vecteur 784
        self.flatten = nn.Flatten()
        # Encoder sequence : 784 -> 256 -> tanh -> 10
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.Tanh(),
            nn.Linear(256, 10)
        )
        
    def forward(self, x):
        initial_shape = x.shape  # ex : (batch, 1, 28, 28)
        x = self.flatten(x)      # aplati en (batch, 784)
        x = self.encoder(x)      # sortie (batch, 10)
        # Ici, selon consigne, retourne la sortie telle quelle.
        # Si vous devez la reformer en image, il faut un decodeur (à préciser)
        return x

# Instanciation et transfert sur device (cpu ou gpu)
model = NeuralNetwork().to(device)
print(model)
print(" ")
print(" ")



# --------------------------------------------------------------------------
# Exercice 5.1
print("--------- Exercice 5.1 ---------")
# Critère de perte : mean squared error pour erreur de reconstruction
loss_fn = nn.MSELoss()

# Optimiseur Adam pour les paramètres du modèle
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(loss_fn)
print(optimizer)
print(" ")
print(" ")



# --------------------------------------------------------------------------
# Exercice 5.1
print("--------- Exercice 5.1 ---------")
def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, _ in dataloader:  # on ne s'intéresse pas aux labels pour l'autoencodeur
        X_batch = X_batch.to(device)
        
        # Forward pass
        output = model(X_batch)
        
        # Calculer la perte entre sortie et entrée (reconstruction)
        loss = loss_fn(output, X_batch.view(output.shape))
        
        # Backward pass et optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
    
    avg_loss = total_loss / len(dataloader.dataset)
    print(f"Train loss: {avg_loss:.4f}")
    return avg_loss
print(" ")
print(" ")



# --------------------------------------------------------------------------
# Exercice 6.2
print("--------- Exercice 6.2 ---------")
def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            loss = loss_fn(output, X_batch.view(output.shape))
            total_loss += loss.item() * X_batch.size(0)
    
    avg_loss = total_loss / len(dataloader.dataset)
    print(f"Test loss: {avg_loss:.4f}")
    return avg_loss
print(" ")
print(" ")



# --------------------------------------------------------------------------
# Exercice 6.3
print("--------- Exercice 6.3 ---------")
n_epochs = 5

for epoch in range(1, n_epochs + 1):
    print(f"Epoch {epoch}/{n_epochs}")
    
    train_loss = train_loop(train_loader, model, loss_fn, optimizer, device)
    test_loss = test_loop(test_loader, model, loss_fn, device)
    
    print("-" * 30)