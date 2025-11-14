import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

print(" ")

# --------------------------------------------------------------------------
# Exercice 1
print("--------- Exercice 1 ---------")
# Generate a sine wave dataset
def generate_data(seq_length, num_samples):
    x = np.linspace(0, 100, num_samples)
    data = np.sin(x)
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)

print("La fonction generate_data crée une série temporelle en générant une ")
print("suite de valeurs sinusoïdales uniformément espacées. Elle découpe ensuite")
print("cette série en séquences de longueur fixe, chaque séquence étant associée")
print("à la valeur suivante à prédire. Ce procédé permet de préparer facilement ")
print("des données pour entraîner un modèle à prédire la prochaine valeur d'une ")
print("série à partir de ses valeurs précédentes.")
print(" ")
print(" ")



# --------------------------------------------------------------------------
# Exercice 2
print("--------- Exercice 2 ---------")
sequences, targets = generate_data(30, 1000)
print("Voir le code")
print(" ")
print(" ")


# --------------------------------------------------------------------------
# Exercice 3
print("--------- Exercice 3 ---------")
def plot_sequences_with_targets(sequences, targets, num_to_plot=5):
    plt.figure(figsize=(12, 6))
    for i in range(num_to_plot):
        seq = sequences[i]
        target = targets[i]
        plt.plot(range(len(seq)), seq, label=f'Sequence n°{i+1}')
        plt.scatter(len(seq), target, color='red')  # la cible à la fin de la séquence
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.title('Sequences and their next-step targets')
    plt.legend()
    plt.show()

print("Voir schéma ci présent")
plot_sequences_with_targets(sequences, targets, num_to_plot=5)

print(" ")
print(" ")



# --------------------------------------------------------------------------
# Exercice 4
print("--------- Exercice 4 ---------")
print("Le perceptron multicouche (MLP) utilisé ici sert à prédire la ")
print("prochaine valeur d'une série temporelle en se basant sur un ensemble")
print("de valeurs précédentes. Concrètement, le modèle reçoit en entrée ")
print("une séquence de données (par exemple, 30 valeurs), et il doit produire")
print("une unique valeur en sortie, qui est la prévision de la donnée suivante")
print("dans la série. Ce type de modèle apprend à capturer les relations entre ")
print("les valeurs passées pour estimer la valeur future.")
print(" ")
print(" ")



# --------------------------------------------------------------------------
# Exercice 5
print("--------- Exercice 5 ---------")
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(30, 64),  # exemple: entrée taille 30 (sequence length)
            nn.ReLU(),
            nn.Linear(64, 1)    # sortie une valeur scalaire (prédiction prochaine valeur)
        )

    def forward(self, x):
        return self.model(x).squeeze()

# Création du modèle, critère et optimiseur
model = SimpleMLP()
criterion = nn.MSELoss()  # erreur quadratique moyenne pour régression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Supposons que train_loader et test_loader soient déjà définis
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    loss_epoch = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()

    if epoch % 1 == 0:
        model.eval()
        loss_test = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss_test += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss train: {loss_epoch:.4f}, Loss test: {loss_test:.4f}")

print(" ")
print(" ")



# --------------------------------------------------------------------------
# Exercice 6
print("--------- Exercice 6 ---------")
