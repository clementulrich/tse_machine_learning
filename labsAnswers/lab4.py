import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
print('PyTorch version:', torch.__version__)

device = torch.device("cpu")


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

# Création des dataloaders
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)



# --------------------------------------------------------------------------
# Exercice 2.1
print("--------- Exercice 2.1 ---------")
# TODO: Implement the autoencoder
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.Tanh(),
            nn.Linear(256, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784)
        )
        
    def forward(self, x):
        initial_shape = x.shape  # e.g. (batch_size, 28, 28)
        x = self.flatten(x)      # aplatir en (batch_size, 784)
        x = self.encoder(x)      # encoder vers un vecteur latent 2D
        x = self.decoder(x)      # decoder vers un vecteur de taille 784
        return x.view(initial_shape)  # reshape en (batch_size, 28, 28)

model = NeuralNetwork().to(device)
print(model)
print(" ")
print(" ")



# --------------------------------------------------------------------------
# Exercice 3.1
print("--------- Exercice 3.1 ---------")
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(" ")
print(" ")



# --------------------------------------------------------------------------
# Exercice 4.1
print("--------- Exercice 4.1 ---------")
def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch, (X, _) in enumerate(dataloader):
        X = X.to(device)
        
        # Forward pass : reconstruction
        reconstruction = model(X)
        
        # Calcul de la perte
        loss = loss_fn(reconstruction, X)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Training loss: {avg_loss:.4f}")
    return avg_loss
print(" ")
print(" ")



# --------------------------------------------------------------------------
# Exercice 4.2
print("--------- Exercice 4.2 ---------")
def test(dataloader, model, loss_fn, device):
    model.eval()  # mode évaluation, désactivation du dropout, batchnorm, etc.
    total_loss = 0
    
    with torch.no_grad():  # pas de calcul de gradients
        for X, _ in dataloader:
            X = X.to(device)
            reconstruction = model(X)
            loss = loss_fn(reconstruction, X)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Test loss: {avg_loss:.4f}")
    return avg_loss
print(" ")
print(" ")



# --------------------------------------------------------------------------
# Exercice 4.3
print("--------- Exercice 4.3 ---------")
num_epochs = 5

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    train_loss = train(train_dataloader, model, loss_function, optimizer, device)
    test_loss = test(test_dataloader, model, loss_function, device)
    
    print(f"Epoch {epoch+1} - Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}\n")
print(" ")
print(" ")
