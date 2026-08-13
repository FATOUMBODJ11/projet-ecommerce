"""
Étape 3 - Construction du Data Lake en 3 couches (raw -> staging -> curated)

raw/      : données brutes telles que générées (jamais modifiées, source de vérité)
staging/  : données nettoyées (types corrects, doublons supprimés, NA traités)
curated/  : données enrichies et prêtes à l'emploi pour Data Scientist / BI Analyst
            (jointures, colonnes calculées : marge, mois, montant total, etc.)

Chaque transformation est loggée pour tracabilité (utile pour le rapport /
les data quality checks de l'étape suivante).
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

RAW_DIR = "data/raw"
STAGING_DIR = "data/staging"
CURATED_DIR = "data/curated"

for d in [STAGING_DIR, CURATED_DIR]:
    os.makedirs(d, exist_ok=True)

log = {"run_at": datetime.now().isoformat(), "steps": []}


def log_step(name, details):
    log["steps"].append({"step": name, **details})
    print(f"[{name}] {details}")


# ---------------------------------------------------------------------------
# RAW -> STAGING : nettoyage
# ---------------------------------------------------------------------------
def clean_clients():
    df = pd.read_csv(f"{RAW_DIR}/clients.csv")
    n_before = len(df)

    df = df.drop_duplicates(subset="id_client", keep="first")
    df["email"] = df["email"].fillna("email_inconnu@na.com")
    df["date_inscription"] = pd.to_datetime(df["date_inscription"])
    df["age"] = df["age"].clip(18, 90)

    log_step("clean_clients", {
        "lignes_avant": n_before, "lignes_apres": len(df),
        "doublons_supprimes": n_before - len(df),
    })
    df.to_parquet(f"{STAGING_DIR}/clients.parquet", index=False)
    return df


def clean_produits():
    df = pd.read_csv(f"{RAW_DIR}/produits.csv")
    n_before = len(df)

    df = df.drop_duplicates(subset="id_produit", keep="first")
    stock_median = df["stock"].median()
    n_stock_na = df["stock"].isna().sum()
    df["stock"] = df["stock"].fillna(stock_median)
    df["stock"] = df["stock"].astype(int)

    log_step("clean_produits", {
        "lignes_avant": n_before, "lignes_apres": len(df),
        "stock_manquants_imputes": int(n_stock_na),
    })
    df.to_parquet(f"{STAGING_DIR}/produits.parquet", index=False)
    return df


def clean_promotions():
    df = pd.read_csv(f"{RAW_DIR}/promotions.csv")
    df["date_debut"] = pd.to_datetime(df["date_debut"])
    df["date_fin"] = pd.to_datetime(df["date_fin"])
    log_step("clean_promotions", {"lignes": len(df)})
    df.to_parquet(f"{STAGING_DIR}/promotions.parquet", index=False)
    return df


def clean_ventes(clients_df, produits_df):
    df = pd.read_csv(f"{RAW_DIR}/ventes.csv")
    n_before = len(df)

    # doublons exacts
    df = df.drop_duplicates(subset="id_vente", keep="first")

    # écarte les ventes dont le client ou le produit n'existe pas (orphelines)
    valid_clients = set(clients_df["id_client"])
    valid_produits = set(produits_df["id_produit"])
    n_orphans = (~df["id_client"].isin(valid_clients) | ~df["id_produit"].isin(valid_produits)).sum()
    df = df[df["id_client"].isin(valid_clients) & df["id_produit"].isin(valid_produits)]

    # traite les quantités aberrantes (> 10 = probable erreur de saisie, on cap à la médiane+3*std)
    q_median, q_std = df["quantite"].median(), df["quantite"].std()
    cap = q_median + 3 * q_std
    n_outliers = (df["quantite"] > cap).sum()
    df.loc[df["quantite"] > cap, "quantite"] = int(q_median)

    df["date_vente"] = pd.to_datetime(df["date_vente"], format="mixed")

    log_step("clean_ventes", {
        "lignes_avant": n_before, "lignes_apres": len(df),
        "doublons_supprimes": n_before - len(df) - n_orphans,
        "orphelines_supprimees": int(n_orphans),
        "outliers_quantite_corriges": int(n_outliers),
    })
    df.to_parquet(f"{STAGING_DIR}/ventes.parquet", index=False)
    return df


# ---------------------------------------------------------------------------
# STAGING -> CURATED : enrichissement (jointures + colonnes calculées)
# ---------------------------------------------------------------------------
def build_curated_ventes(ventes_df, clients_df, produits_df):
    df = ventes_df.merge(
        clients_df[["id_client", "segment", "ville"]], on="id_client", how="left"
    ).merge(
        produits_df[["id_produit", "categorie", "prix", "cout"]], on="id_produit", how="left"
    )

    df["montant_total"] = (df["prix_paye"] * df["quantite"]).round(2)
    df["marge_unitaire"] = (df["prix_paye"] - df["cout"]).round(2)
    df["marge_totale"] = (df["marge_unitaire"] * df["quantite"]).round(2)
    df["annee_mois"] = df["date_vente"].dt.to_period("M").astype(str)

    log_step("curated_ventes", {"lignes": len(df), "colonnes": list(df.columns)})
    df.to_parquet(f"{CURATED_DIR}/ventes_enrichies.parquet", index=False)
    return df


def build_curated_kpis_mensuels(ventes_enrichies):
    kpi = ventes_enrichies.groupby("annee_mois").agg(
        chiffre_affaires=("montant_total", "sum"),
        marge_totale=("marge_totale", "sum"),
        nb_ventes=("id_vente", "count"),
        nb_clients_actifs=("id_client", "nunique"),
    ).reset_index().sort_values("annee_mois")

    log_step("curated_kpis_mensuels", {"lignes": len(kpi)})
    kpi.to_parquet(f"{CURATED_DIR}/kpis_mensuels.parquet", index=False)
    kpi.to_csv(f"{CURATED_DIR}/kpis_mensuels.csv", index=False)  # lisible direct pour le BI Analyst
    return kpi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== RAW -> STAGING ===")
    clients_df = clean_clients()
    produits_df = clean_produits()
    promos_df = clean_promotions()
    ventes_df = clean_ventes(clients_df, produits_df)

    print("\n=== STAGING -> CURATED ===")
    ventes_enrichies = build_curated_ventes(ventes_df, clients_df, produits_df)
    kpis = build_curated_kpis_mensuels(ventes_enrichies)

    with open(f"{STAGING_DIR}/../pipeline_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)

    print("\nData Lake construit :")
    print(f"  raw/      -> données brutes ({RAW_DIR})")
    print(f"  staging/  -> données nettoyées ({STAGING_DIR})")
    print(f"  curated/  -> données enrichies ({CURATED_DIR})")
    print("\nLog complet -> data/pipeline_log.json")
