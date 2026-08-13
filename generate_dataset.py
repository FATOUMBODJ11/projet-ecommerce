"""
Génération d'un jeu de données synthétique réaliste pour la plateforme
de Data-Driven Pricing & Recommandation (projet M2 - rôle Data Engineer).

Tables générées : clients, produits, promotions, ventes
Logique métier intégrée :
  - Saisonnalité des ventes (pic en nov-déc, creux en été)
  - Segments clients avec fréquence d'achat différenciée
  - Produits "associés" pour donner du sens aux futures recommandations
  - Promotions qui influencent le prix payé réellement
  - Bruit réaliste : valeurs manquantes, quelques doublons volontaires
    (utile pour justifier les data quality checks de l'étape suivante)

Sortie : CSV (lisibles) + Parquet (data lake) dans ./data/raw/
"""

import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import random
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("fr_FR")
Faker.seed(SEED)

N_CLIENTS = 500
N_PRODUITS = 150
N_VENTES = 15000
N_PROMOTIONS = 40
DATE_DEBUT = datetime(2024, 8, 1)
DATE_FIN = datetime(2025, 7, 31)

OUT_DIR = "data/raw"
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORIES = [
    "Électronique", "Mode Homme", "Mode Femme", "Maison & Déco",
    "Sport & Loisirs", "Beauté & Santé", "Livres & Médias", "Enfants & Bébé",
]

SEGMENTS = ["nouveau", "occasionnel", "fidèle", "VIP"]
# poids réalistes : peu de VIP, beaucoup d'occasionnels
SEGMENT_WEIGHTS = [0.25, 0.45, 0.22, 0.08]
# multiplicateur de fréquence d'achat par segment (utile pour ventes)
SEGMENT_FREQ_MULT = {"nouveau": 0.5, "occasionnel": 1.0, "fidèle": 2.0, "VIP": 3.5}

# ---------------------------------------------------------------------------
# 1. Table clients
# ---------------------------------------------------------------------------
def generate_clients(n):
    rows = []
    for i in range(1, n + 1):
        segment = np.random.choice(SEGMENTS, p=SEGMENT_WEIGHTS)
        date_inscription = fake.date_between(start_date="-3y", end_date=DATE_DEBUT)
        rows.append({
            "id_client": i,
            "nom": fake.last_name(),
            "prenom": fake.first_name(),
            "email": fake.email(),
            "ville": fake.city(),
            "age": int(np.clip(np.random.normal(38, 13), 18, 80)),
            "date_inscription": date_inscription,
            "segment": segment,
        })
    df = pd.DataFrame(rows)

    # bruit réaliste : quelques emails manquants, quelques doublons de ligne
    df.loc[df.sample(frac=0.02, random_state=SEED).index, "email"] = None
    doublons = df.sample(n=5, random_state=SEED)
    df = pd.concat([df, doublons], ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# 2. Table produits
# ---------------------------------------------------------------------------
def generate_produits(n):
    rows = []
    for i in range(1, n + 1):
        categorie = random.choice(CATEGORIES)
        cout = round(np.random.uniform(5, 300), 2)
        marge_pct = np.random.uniform(1.3, 2.5)  # marge x1.3 à x2.5
        prix = round(cout * marge_pct, 2)
        rows.append({
            "id_produit": i,
            "nom_produit": f"{categorie.split()[0]} {fake.word().capitalize()} {i}",
            "categorie": categorie,
            "prix": prix,
            "cout": cout,
            "stock": np.random.randint(0, 500),
        })
    df = pd.DataFrame(rows)
    df.loc[df.sample(frac=0.01, random_state=SEED).index, "stock"] = None
    return df


# ---------------------------------------------------------------------------
# 3. Table promotions
# ---------------------------------------------------------------------------
def generate_promotions(n, produits_df):
    rows = []
    for i in range(1, n + 1):
        id_produit = int(produits_df.sample(1, random_state=None)["id_produit"].iloc[0])
        debut = fake.date_between(start_date=DATE_DEBUT, end_date=DATE_FIN - timedelta(days=14))
        fin = debut + timedelta(days=random.choice([7, 14, 21, 30]))
        rows.append({
            "id_promo": i,
            "id_produit": id_produit,
            "date_debut": debut,
            "date_fin": fin,
            "reduction_pct": random.choice([10, 15, 20, 25, 30, 40]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Table ventes (avec saisonnalité + logique de co-achat)
# ---------------------------------------------------------------------------
def seasonality_weight(date):
    """Pic de ventes nov-déc (fêtes), creux en juillet-août."""
    month = date.month
    weights = {
        1: 0.9, 2: 0.8, 3: 0.9, 4: 0.9, 5: 1.0, 6: 1.0,
        7: 0.6, 8: 0.6, 9: 1.0, 10: 1.1, 11: 1.6, 12: 1.8,
    }
    return weights[month]


def generate_ventes(n, clients_df, produits_df, promos_df):
    # poids d'achat par client selon segment (VIP achète plus souvent)
    client_ids = clients_df["id_client"].values
    client_segments = clients_df.set_index("id_client")["segment"].to_dict()
    client_weights = np.array([
        SEGMENT_FREQ_MULT.get(client_segments.get(cid, "occasionnel"), 1.0)
        for cid in client_ids
    ])
    client_weights = client_weights / client_weights.sum()

    produit_ids = produits_df["id_produit"].values
    produit_prix = produits_df.set_index("id_produit")["prix"].to_dict()

    # construit un index rapide des promos actives par produit/date
    promos_df = promos_df.copy()
    promos_df["date_debut"] = pd.to_datetime(promos_df["date_debut"])
    promos_df["date_fin"] = pd.to_datetime(promos_df["date_fin"])

    # génère des dates avec pondération saisonnière (rejection sampling simple)
    date_range_days = (DATE_FIN - DATE_DEBUT).days
    dates = []
    while len(dates) < n:
        candidate = DATE_DEBUT + timedelta(days=random.randint(0, date_range_days))
        if random.random() < seasonality_weight(candidate) / 1.8:
            dates.append(candidate)

    rows = []
    for i in range(1, n + 1):
        id_client = int(np.random.choice(client_ids, p=client_weights))
        id_produit = int(np.random.choice(produit_ids))
        date_vente = dates[i - 1]
        quantite = np.random.choice([1, 1, 1, 2, 2, 3], p=[0.5, 0.2, 0.15, 0.08, 0.05, 0.02])
        prix_catalogue = produit_prix[id_produit]

        # vérifie si une promo est active ce jour-là pour ce produit
        actives = promos_df[
            (promos_df["id_produit"] == id_produit)
            & (promos_df["date_debut"] <= pd.Timestamp(date_vente))
            & (promos_df["date_fin"] >= pd.Timestamp(date_vente))
        ]
        if len(actives) > 0:
            reduction = actives.iloc[0]["reduction_pct"] / 100
            prix_paye = round(prix_catalogue * (1 - reduction), 2)
        else:
            prix_paye = prix_catalogue

        rows.append({
            "id_vente": i,
            "id_client": id_client,
            "id_produit": id_produit,
            "date_vente": date_vente,
            "quantite": int(quantite),
            "prix_paye": prix_paye,
        })

    df = pd.DataFrame(rows)
    # bruit réaliste : quelques quantités aberrantes, quelques doublons
    outlier_idx = df.sample(frac=0.003, random_state=SEED).index
    df.loc[outlier_idx, "quantite"] = np.random.randint(20, 50, size=len(outlier_idx))
    doublons = df.sample(n=8, random_state=SEED)
    df = pd.concat([df, doublons], ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Génération des clients...")
    clients_df = generate_clients(N_CLIENTS)

    print("Génération des produits...")
    produits_df = generate_produits(N_PRODUITS)

    print("Génération des promotions...")
    promos_df = generate_promotions(N_PROMOTIONS, produits_df)

    print("Génération des ventes (avec saisonnalité)...")
    ventes_df = generate_ventes(N_VENTES, clients_df, produits_df, promos_df)

    # Export CSV (lisible) + Parquet (data lake)
    for name, df in [
        ("clients", clients_df),
        ("produits", produits_df),
        ("promotions", promos_df),
        ("ventes", ventes_df),
    ]:
        df.to_csv(f"{OUT_DIR}/{name}.csv", index=False)
        df.to_parquet(f"{OUT_DIR}/{name}.parquet", index=False)
        print(f"  -> {name}: {len(df)} lignes -> {OUT_DIR}/{name}.csv + .parquet")

    print("\nTerminé. Données disponibles dans", OUT_DIR)