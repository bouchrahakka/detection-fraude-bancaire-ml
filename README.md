# Détection de fraude bancaire par Machine Learning

Ce projet a été réalisé pendant mon stage ouvrier de première année à la banque populaire.

## Le problème

Une banque traite des milliers de transactions chaque jour et seule une infime partie sont frauduleuses. L'objectif de ce projet est de tester si le Machine Learning peut aider à repérer automatiquement ces transactions suspectes.

## Données utilisées

J'ai travaillé sur le dataset **Credit Card Fraud Detection**, disponible publiquement sur Kaggle : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Le fichier de données n'est pas inclus ici (il est volumineux et déjà accessible sur Kaggle). Pour reproduire le projet télécharge `creditcard.csv` depuis le lien ci-dessus et place-le à la racine du dossier.

## Ce que j'ai fait

- Exploration des données pour comprendre le déséquilibre entre transactions normales et frauduleuses
- Prétraitement : normalisation des variables, séparation en jeu d'entraînement et de test
- Test de deux approches différentes :
  - **Isolation Forest**, qui n'a pas besoin de connaître les vraies étiquettes
  - **Régression logistique**, entraînée avec les exemples déjà étiquetés
- Comparaison des deux modèles avec des métriques adaptées (accuracy seule n'a pas de sens ici vu le déséquilibre)

## Résultats

Les graphiques et scores obtenus sont dans le dossier [`resultats/`](./resultats).

## Pour lancer le projet

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python detection_fraude.py
```

## Outils utilisés

Python, Pandas, Scikit-learn, Matplotlib, Seaborn

---
HAKKA Bouchra - ENSMR
