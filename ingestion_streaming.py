"""
Étape 4b - Ingestion STREAMING (simulée)

Simule un flux de ventes en temps réel : toutes les X secondes, une nouvelle
vente "arrive" (comme un client qui achète en direct sur le site) et est
ajoutée aux données.

En production, ce serait remplacé par un vrai broker de messages (Kafka,
Kinesis, RabbitMQ). Ici, on simule le même principe avec un simple flux
Python en boucle, ce qui suffit pour démontrer la logique batch vs streaming
attendue par le sujet.

Usage :
    python ingestion_streaming.py                # tourne en continu (Ctrl+C pour arrêter)
    python ingestion_streaming.py --n 20 --delay 1   # génère 20 ventes, 1s d'intervalle
"""

import pandas as pd
import numpy as np
import argparse
import time
import os
import random
from datetime import datetime

RAW_VENTES = "data/raw/ventes.csv"
STREAM_LOG = "data/streaming_log.csv"


def load_reference_data():
    """Charge clients et produits existants pour générer des ventes cohérentes."""
    clients = pd.read_csv("data/raw/clients.csv")
    produits = pd.read_csv("data/raw/produits.csv")
    return clients, produits


def get_next_vente_id():
    """Récupère le prochain id_vente disponible en regardant le fichier existant."""
    if os.path.exists(RAW_VENTES):
        df = pd.read_csv(RAW_VENTES)
        return int(df["id_vente"].max()) + 1
    return 1


def generate_one_vente(next_id, clients, produits):
    """Génère une vente 'en direct', cohérente avec les données existantes."""
    id_client = int(clients.sample(1)["id_client"].iloc[0])
    produit_row = produits.sample(1).iloc[0]
    id_produit = int(produit_row["id_produit"])
    prix_catalogue = float(produit_row["prix"])

    quantite = np.random.choice([1, 1, 1, 2, 2, 3], p=[0.5, 0.2, 0.15, 0.08, 0.05, 0.02])

    return {
        "id_vente": next_id,
        "id_client": id_client,
        "id_produit": id_produit,
        "date_vente": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "quantite": int(quantite),
        "prix_paye": prix_catalogue,
    }


def append_vente(vente):
    """Ajoute la vente au fichier raw (source de vérité) et au log streaming."""
    df_new = pd.DataFrame([vente])

    # ajoute au fichier ventes brut (append, jamais d'écrasement)
    df_new.to_csv(RAW_VENTES, mode="a", header=False, index=False)

    # log dédié pour visualiser le flux streaming séparément
    write_header = not os.path.exists(STREAM_LOG)
    df_new.to_csv(STREAM_LOG, mode="a", header=write_header, index=False)


def run_streaming(n=None, delay=2):
    clients, produits = load_reference_data()
    next_id = get_next_vente_id()
    count = 0

    print(f"[STREAMING] Démarrage du flux simulé (intervalle: {delay}s)")
    print("[STREAMING] Ctrl+C pour arrêter" if n is None else f"[STREAMING] {n} ventes à générer")

    try:
        while n is None or count < n:
            vente = generate_one_vente(next_id, clients, produits)
            append_vente(vente)
            print(f"[STREAMING] Nouvelle vente ingérée -> {vente}")

            next_id += 1
            count += 1
            time.sleep(delay)
    except KeyboardInterrupt:
        print(f"\n[STREAMING] Arrêté manuellement. {count} ventes ingérées durant ce run.")

    print(f"[STREAMING] Terminé. Total ventes ingérées: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="Nombre de ventes à générer (défaut: infini)")
    parser.add_argument("--delay", type=float, default=2, help="Délai en secondes entre chaque vente")
    args = parser.parse_args()

    run_streaming(n=args.n, delay=args.delay)