# tse_machine_learning

Ce dépôt regroupe mes travaux pratiques réalisés dans le cadre d’un cours de Machine Learning et Deep Learning.  
Il comprend à la fois les fichiers de consignes (format Jupyter Notebook) et mes réponses personnelles dans les scripts Python (labX.py).


---

## Structure du dépôt

├── labs/
│ ├── PyTorch_Lab1_introtorch.ipynb
│ ├── PyTorch_Lab2_newton.ipynb
│ ├── PyTorch_Lab3_mlp.ipynb
│ ├── PyTorch_Lab4_autoencodeur.ipynb
│ ├── PyTorch_Lab5_CNN_corr.ipynb
│ ├── PyTorch_Lab6_rnn.ipynb
│ └── PyTorch_Lab7_Transformer.ipynb
├── labsAnswers/
│ ├── data/
│ │ └── (datasets utilisés, ex: FashionMNIST)
│ ├── lab1.py
│ ├── lab2.py
│ ├── lab3.py
│ ├── lab4.py
│ └── lab6.py
├── README.md
└── .gitignore


- **`labs/`** : Notebooks Jupyter contenant les consignes originales, explications et squelettes de code à compléter.
- **`labsAnswers/`** : Mes scripts de réponses pour chaque TP, organisés par numéro de lab. Le dossier `data/` contient les datasets nécessaires au bon fonctionnement (ex : FashionMNIST au format attendu par PyTorch).


---

## Objectifs pédagogiques

- Apprendre et pratiquer le Machine Learning et le Deep Learning avec Python et PyTorch.
- Mettre en oeuvre des architectures variées : MLP, autoencodeur, CNN, RNN, Transformer.
- Expérimenter la génération de données synthétiques, la manipulation de séries temporelles et les visualisations associées.
- Utiliser de vrais jeux de données comme MNIST/FashionMNIST pour l’illustration.


---

## Environnement requis

- Python ≥ 3.8 recommandé
- PyTorch ≥ 1.8
- Jupyter Notebook (pour les consignes et l’exploration interactive)
- (Optionnel) Matplotlib, Numpy, torchvision

Installation rapide (sous Windows) :

`pip install torch torchvision matplotlib jupyter`


---

## Exécution

- Les fichiers `.ipynb` (dans `labs/`) sont à ouvrir avec Jupyter Notebook ou VS Code.
- Les fichiers `.py` (dans `labsAnswers/`) peuvent être lancés dans le terminal pour tester les réponses, en veillant à mettre les jeux de données dans `data/` si besoin.


---

## Notes

- Chaque lab répond aux consignes du notebook correspondant.
- Les datasets doivent être présents dans `labsAnswers/data/` pour un fonctionnement hors-ligne.
- Ce travail illustre un parcours progressif d’apprentissage du Deep Learning, de l’introduction à la mise en oeuvre d’architectures avancées.


---

_Tout retour ou suggestion d’amélioration est le bienvenu._
