import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_fscore_support
)

sns.set_theme(style="whitegrid")
os.makedirs("resultats", exist_ok=True)


def charger_donnees(chemin="creditcard.csv"):
    if os.path.exists(chemin):
        print(f"Chargement du dataset réel depuis {chemin}")
        return pd.read_csv(chemin)

    print("Fichier creditcard.csv introuvable -> génération d'un dataset SIMULÉ "
          "(même structure : Time, V1-V28, Amount, Class) pour tester le pipeline.")
    rng = np.random.default_rng(42)
    n_normal, n_fraude = 5000, 25

    def bloc(n, fraude=False):
        data = {"Time": rng.integers(0, 172800, n)}
        for i in range(1, 29):
            shift = 2.5 if fraude else 0
            data[f"V{i}"] = rng.normal(shift * rng.choice([-1, 1]), 1, n)
        data["Amount"] = rng.gamma(2, 250, n) if not fraude else rng.gamma(1.5, 600, n)
        data["Class"] = 1 if fraude else 0
        return pd.DataFrame(data)

    df = pd.concat([bloc(n_normal, False), bloc(n_fraude, True)], ignore_index=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


df = charger_donnees()
print("\nAperçu des données :")
print(df.head())
print(f"\nDimensions : {df.shape[0]} transactions, {df.shape[1]} colonnes")

nb_fraudes = int(df["Class"].sum())
nb_total = len(df)
taux_fraude = nb_fraudes / nb_total * 100
print(f"\nNombre de fraudes : {nb_fraudes} / {nb_total} transactions ({taux_fraude:.3f} %)")

plt.figure(figsize=(5, 4))
sns.countplot(x="Class", data=df, palette=["#1F3864", "#C99E3C"])
plt.title("Répartition des transactions (0 = normale, 1 = fraude)")
plt.xlabel("Classe"); plt.ylabel("Nombre de transactions"); plt.yscale("log")
plt.tight_layout(); plt.savefig("resultats/1_repartition_classes.png", dpi=150); plt.close()

plt.figure(figsize=(6, 4))
sns.boxplot(x="Class", y="Amount", data=df, palette=["#1F3864", "#C99E3C"])
plt.title("Distribution des montants par classe")
plt.ylim(0, df["Amount"].quantile(0.99))
plt.tight_layout(); plt.savefig("resultats/2_montants_par_classe.png", dpi=150); plt.close()

X = df.drop(columns=["Class"]); y = df["Class"]
scaler = StandardScaler()
X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\nTrain : {X_train.shape[0]} | Test : {X_test.shape[0]}")

print("\n=== Isolation Forest ===")
iso = IsolationForest(n_estimators=200, contamination=nb_fraudes / nb_total, random_state=42)
iso.fit(X_train)
pred_iso = iso.predict(X_test)
pred_iso = np.where(pred_iso == -1, 1, 0)
print(classification_report(y_test, pred_iso, target_names=["Normale", "Fraude"]))
prec_iso, rec_iso, f1_iso, _ = precision_recall_fscore_support(y_test, pred_iso, average="binary", zero_division=0)

print("\n=== Logistic Regression ===")
log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
pred_log = log_reg.predict(X_test)
proba_log = log_reg.predict_proba(X_test)[:, 1]
print(classification_report(y_test, pred_log, target_names=["Normale", "Fraude"]))
prec_log, rec_log, f1_log, _ = precision_recall_fscore_support(y_test, pred_log, average="binary", zero_division=0)
auc_log = roc_auc_score(y_test, proba_log)
print(f"ROC-AUC : {auc_log:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, preds, titre in zip(axes, [pred_iso, pred_log], ["Isolation Forest", "Logistic Regression"]):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normale", "Fraude"], yticklabels=["Normale", "Fraude"])
    ax.set_title(f"Matrice de confusion — {titre}"); ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
plt.tight_layout(); plt.savefig("resultats/3_matrices_confusion.png", dpi=150); plt.close()

fpr, tpr, _ = roc_curve(y_test, proba_log)
plt.figure(figsize=(5.5, 4.5))
plt.plot(fpr, tpr, color="#1F3864", label=f"ROC (AUC = {auc_log:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.xlabel("Taux de faux positifs"); plt.ylabel("Taux de vrais positifs")
plt.title("Courbe ROC — Logistic Regression"); plt.legend()
plt.tight_layout(); plt.savefig("resultats/4_courbe_roc.png", dpi=150); plt.close()

# Sauvegarde des métriques pour le rapport
with open("resultats/metrics.txt", "w") as f:
    f.write(f"nb_total={nb_total}\n nb_fraudes={nb_fraudes}\n taux_fraude={taux_fraude:.3f}\n")
    f.write(f"iso_precision={prec_iso:.3f}\n iso_recall={rec_iso:.3f}\n iso_f1={f1_iso:.3f}\n")
    f.write(f"log_precision={prec_log:.3f}\n log_recall={rec_log:.3f}\n log_f1={f1_log:.3f}\n log_auc={auc_log:.3f}\n")

print("\nTerminé. Graphiques et métriques sauvegardés dans resultats/")
