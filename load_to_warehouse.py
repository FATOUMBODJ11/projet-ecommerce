"""
Étape 6 - Chargement dans le Data Warehouse (PostgreSQL)

Charge les données de la couche CURATED (nettoyées + enrichies) dans une
base PostgreSQL, avec un schéma relationnel propre (clés primaires,
clés étrangères), pour que le Data Scientist et le BI Analyst puissent
s'y connecter directement (SQL, notebooks, dashboard).

Prérequis :
    docker compose up -d          (lance Postgres + Adminer)
    pip install sqlalchemy psycopg2-binary

Usage :
    python load_to_warehouse.py
"""

import pandas as pd
from sqlalchemy import create_engine, text
import time
import os

# Fix pour un bug connu de psycopg2 sous Windows : force l'encodage client
# en UTF-8 (sinon Windows utilise parfois l'encodage régional par défaut,
# ce qui casse le décodage des messages de connexion Postgres).
os.environ["PGCLIENTENCODING"] = "UTF8"

DB_USER = "dataeng"
DB_PASSWORD = "dataeng_pwd"
DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "ecommerce_dw"

DB_URL = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

STAGING_DIR = "data/staging"
CURATED_DIR = "data/curated"

DDL = """
DROP TABLE IF EXISTS ventes CASCADE;
DROP TABLE IF EXISTS promotions CASCADE;
DROP TABLE IF EXISTS produits CASCADE;
DROP TABLE IF EXISTS clients CASCADE;

CREATE TABLE clients (
    id_client INTEGER PRIMARY KEY,
    nom VARCHAR(100),
    prenom VARCHAR(100),
    email VARCHAR(150),
    ville VARCHAR(100),
    age INTEGER,
    date_inscription DATE,
    segment VARCHAR(20)
);

CREATE TABLE produits (
    id_produit INTEGER PRIMARY KEY,
    nom_produit VARCHAR(150),
    categorie VARCHAR(50),
    prix NUMERIC(10,2),
    cout NUMERIC(10,2),
    stock INTEGER
);

CREATE TABLE promotions (
    id_promo INTEGER PRIMARY KEY,
    id_produit INTEGER REFERENCES produits(id_produit),
    date_debut DATE,
    date_fin DATE,
    reduction_pct NUMERIC(5,2)
);

CREATE TABLE ventes (
    id_vente INTEGER PRIMARY KEY,
    id_client INTEGER REFERENCES clients(id_client),
    id_produit INTEGER REFERENCES produits(id_produit),
    date_vente TIMESTAMP,
    quantite INTEGER,
    prix_paye NUMERIC(10,2),
    montant_total NUMERIC(10,2),
    marge_unitaire NUMERIC(10,2),
    marge_totale NUMERIC(10,2)
);

CREATE INDEX idx_ventes_date ON ventes(date_vente);
CREATE INDEX idx_ventes_client ON ventes(id_client);
CREATE INDEX idx_ventes_produit ON ventes(id_produit);
"""


def wait_for_db(engine, retries=10, delay=2):
    """Attend que Postgres soit prêt à accepter des connexions (utile juste après docker compose up)."""
    for i in range(retries):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("[DB] Connexion Postgres OK")
            return True
        except Exception as e:
            print(f"[DB] Postgres pas encore prêt ({i+1}/{retries})... {e}")
            time.sleep(delay)
    return False


def create_schema(engine):
    print("[DB] Création du schéma (tables + clés étrangères + index)...")
    with engine.begin() as conn:
        for statement in DDL.strip().split(";"):
            if statement.strip():
                conn.execute(text(statement))
    print("[DB] Schéma créé.")


def load_table(engine, df, table_name):
    df.to_sql(table_name, engine, if_exists="append", index=False, method="multi", chunksize=1000)
    print(f"[DB] {table_name}: {len(df)} lignes chargées.")


if __name__ == "__main__":
    engine = create_engine(DB_URL)

    if not wait_for_db(engine):
        print("[DB] Impossible de se connecter à Postgres. Vérifie que 'docker compose up -d' a bien été lancé.")
        exit(1)

    create_schema(engine)

    # on charge clients/produits/promotions depuis staging (nettoyés, non enrichis)
    clients = pd.read_parquet(f"{STAGING_DIR}/clients.parquet")
    produits = pd.read_parquet(f"{STAGING_DIR}/produits.parquet")
    promotions = pd.read_parquet(f"{STAGING_DIR}/promotions.parquet")

    # on charge ventes depuis curated (déjà enrichi avec montant_total, marges)
    ventes = pd.read_parquet(f"{CURATED_DIR}/ventes_enrichies.parquet")
    ventes = ventes[[
        "id_vente", "id_client", "id_produit", "date_vente", "quantite",
        "prix_paye", "montant_total", "marge_unitaire", "marge_totale"
    ]]

    load_table(engine, clients, "clients")
    load_table(engine, produits, "produits")
    load_table(engine, promotions, "promotions")
    load_table(engine, ventes, "ventes")


    print("\n[DB] Chargement terminé. Data Warehouse prêt.")
    print(f"[DB] Connexion : {DB_URL.replace(DB_PASSWORD, '****')}")
    print("[DB] Interface web Adminer : http://localhost:8081")
    print("     Système: PostgreSQL | Serveur: postgres | Utilisateur: dataeng | Base: ecommerce_dw")