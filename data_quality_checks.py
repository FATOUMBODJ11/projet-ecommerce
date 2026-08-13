"""
Étape 5 - Data Quality Checks

Vérifie la qualité des données à la couche STAGING (après nettoyage de base,
avant enrichissement) et produit un rapport structuré :
  - Complétude (valeurs manquantes)
  - Unicité (doublons sur clés primaires)
  - Cohérence référentielle (clés étrangères valides)
  - Validité des plages (prix > 0, quantité > 0, âge cohérent, dates OK)
  - Fraîcheur des données

Le rapport est exporté en JSON (pour intégration/monitoring) et en
Markdown (lisible, à inclure dans le rapport du chef de projet).

Usage :
    python data_quality_checks.py
"""

import pandas as pd
from datetime import datetime
import json
import os

STAGING_DIR = "data/staging"
REPORT_DIR = "data/quality_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

DATE_MIN_ATTENDUE = datetime(2024, 1, 1)
DATE_MAX_ATTENDUE = datetime.now()

report = {
    "run_at": datetime.now().isoformat(),
    "tables": {},
    "global_status": "OK",  # passera à "WARNING" ou "ERROR" si problème détecté
}


def add_check(table, check_name, status, details):
    """status: 'OK', 'WARNING', 'ERROR'"""
    report["tables"].setdefault(table, []).append({
        "check": check_name, "status": status, "details": details
    })
    if status == "ERROR":
        report["global_status"] = "ERROR"
    elif status == "WARNING" and report["global_status"] == "OK":
        report["global_status"] = "WARNING"


def check_completude(table, df, colonnes_critiques):
    for col in colonnes_critiques:
        n_missing = df[col].isna().sum()
        pct = round(100 * n_missing / len(df), 2) if len(df) else 0
        status = "OK" if pct == 0 else ("WARNING" if pct < 5 else "ERROR")
        add_check(table, f"completude_{col}", status,
                  f"{n_missing} valeurs manquantes ({pct}%)")


def check_unicite(table, df, cle_primaire):
    n_doublons = df[cle_primaire].duplicated().sum()
    status = "OK" if n_doublons == 0 else "ERROR"
    add_check(table, f"unicite_{cle_primaire}", status,
              f"{n_doublons} doublons détectés sur {cle_primaire}")


def check_coherence_referentielle(table, df, col_fk, valid_ids, nom_ref):
    n_invalid = (~df[col_fk].isin(valid_ids)).sum()
    status = "OK" if n_invalid == 0 else "ERROR"
    add_check(table, f"coherence_{col_fk}", status,
              f"{n_invalid} lignes avec {col_fk} inexistant dans {nom_ref}")


def check_plage(table, df, col, min_val=None, max_val=None, strict_positive=False):
    condition = pd.Series(True, index=df.index)
    label = []
    if strict_positive:
        condition &= df[col] > 0
        label.append("> 0")
    if min_val is not None:
        condition &= df[col] >= min_val
        label.append(f">= {min_val}")
    if max_val is not None:
        condition &= df[col] <= max_val
        label.append(f"<= {max_val}")

    n_invalid = (~condition).sum()
    status = "OK" if n_invalid == 0 else "WARNING"
    add_check(table, f"plage_{col}", status,
              f"{n_invalid} valeurs hors plage attendue ({col} {' & '.join(label)})")


def check_fraicheur(table, df, col_date):
    derniere_date = pd.to_datetime(df[col_date]).max()
    ecart_jours = (datetime.now() - derniere_date).days
    status = "OK" if ecart_jours < 30 else "WARNING"
    add_check(table, "fraicheur", status,
              f"Dernière donnée: {derniere_date.date()} ({ecart_jours} jours d'écart)")


if __name__ == "__main__":
    clients = pd.read_parquet(f"{STAGING_DIR}/clients.parquet")
    produits = pd.read_parquet(f"{STAGING_DIR}/produits.parquet")
    promotions = pd.read_parquet(f"{STAGING_DIR}/promotions.parquet")
    ventes = pd.read_parquet(f"{STAGING_DIR}/ventes.parquet")

    print("=== Vérification CLIENTS ===")
    check_completude("clients", clients, ["nom", "prenom", "ville", "segment"])
    check_unicite("clients", clients, "id_client")
    check_plage("clients", clients, "age", min_val=18, max_val=90)

    print("=== Vérification PRODUITS ===")
    check_completude("produits", produits, ["nom_produit", "categorie", "prix"])
    check_unicite("produits", produits, "id_produit")
    check_plage("produits", produits, "prix", strict_positive=True)
    check_plage("produits", produits, "stock", min_val=0)

    print("=== Vérification PROMOTIONS ===")
    check_unicite("promotions", promotions, "id_promo")
    check_coherence_referentielle("promotions", promotions, "id_produit",
                                    set(produits["id_produit"]), "produits")

    print("=== Vérification VENTES ===")
    check_completude("ventes", ventes, ["id_client", "id_produit", "date_vente"])
    check_unicite("ventes", ventes, "id_vente")
    check_coherence_referentielle("ventes", ventes, "id_client",
                                    set(clients["id_client"]), "clients")
    check_coherence_referentielle("ventes", ventes, "id_produit",
                                    set(produits["id_produit"]), "produits")
    check_plage("ventes", ventes, "quantite", strict_positive=True, max_val=20)
    check_plage("ventes", ventes, "prix_paye", strict_positive=True)
    check_fraicheur("ventes", ventes, "date_vente")

    # --- Export du rapport ---
    with open(f"{REPORT_DIR}/quality_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # version Markdown lisible pour le rapport du chef de projet
    md_lines = [f"# Rapport de qualité des données\n",
                f"Généré le : {report['run_at']}\n",
                f"**Statut global : {report['global_status']}**\n"]
    for table, checks in report["tables"].items():
        md_lines.append(f"\n## Table `{table}`\n")
        md_lines.append("| Check | Statut | Détail |")
        md_lines.append("|---|---|---|")
        for c in checks:
            icon = {"OK": "✅", "WARNING": "⚠️", "ERROR": "❌"}[c["status"]]
            md_lines.append(f"| {c['check']} | {icon} {c['status']} | {c['details']} |")

    with open(f"{REPORT_DIR}/quality_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n=== STATUT GLOBAL : {report['global_status']} ===")
    print(f"Rapport JSON -> {REPORT_DIR}/quality_report.json")
    print(f"Rapport Markdown -> {REPORT_DIR}/quality_report.md")
