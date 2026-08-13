# Data Engineering - Pipeline E-commerce

Documentation technique du pipeline de données pour la plateforme de
Data-Driven Pricing & Recommandation. Ce document explique comment lancer
le pipeline, où trouver les données, et comment s'y connecter (pour le
Data Scientist, le MLOps/Dev et le BI Analyst).

## 1. Vue d'ensemble du pipeline

```
generate_dataset.py          -> génère les données brutes synthétiques (une seule fois)
        |
        v
   data/raw/                 -> CSV + Parquet bruts (clients, produits, promotions, ventes)
        |
ingestion_streaming.py       -> simule des ventes en temps réel (optionnel, à la demande)
        |
        v
build_datalake.py            -> nettoie et enrichit les données
        |
        v
   data/staging/              -> données nettoyées (types corrects, doublons supprimés)
   data/curated/               -> données enrichies (jointures, marges, KPIs mensuels)
        |
        v
data_quality_checks.py       -> vérifie la qualité, produit un rapport
        |
        v
load_to_warehouse.py         -> charge staging/curated dans PostgreSQL
        |
        v
   PostgreSQL (data warehouse)  -> accessible par toute l'équipe

orchestration_flow.py (Prefect) -> enchaîne automatiquement les 3 dernières étapes
```

## 2. Prérequis pour lancer le pipeline

```bash
python -m venv venv
venv\Scripts\activate                 # Windows
pip install pandas numpy faker pyarrow "psycopg[binary]" sqlalchemy prefect
```

Docker Desktop doit être lancé (pour PostgreSQL).

## 3. Comment lancer le pipeline

### Option A - Étape par étape (pour comprendre/débugger)
```bash
python generate_dataset.py       # génère data/raw/ (à ne lancer qu'une fois)
docker compose up -d              # démarre PostgreSQL + Adminer
python build_datalake.py          # raw -> staging -> curated
python data_quality_checks.py     # vérifie la qualité
python load_to_warehouse.py       # charge dans PostgreSQL
```

### Option B - Tout en une commande (recommandé au quotidien)
```bash
docker compose up -d
python orchestration_flow.py
```
Ce script (Prefect) enchaîne `build_datalake.py` → `data_quality_checks.py` →
`load_to_warehouse.py`, et **s'arrête automatiquement** si la qualité des
données est en erreur (protège la base contre des données corrompues).

### Simuler de nouvelles ventes en temps réel (optionnel)
```bash
python ingestion_streaming.py --n 20 --delay 1
```
Puis relancer `python orchestration_flow.py` pour les intégrer.

## 4. Où sont les données

| Dossier / Fichier | Contenu | Qui l'utilise |
|---|---|---|
| `data/raw/` | Données brutes (jamais modifiées) | Traçabilité / debug |
| `data/staging/` | Nettoyées (types, doublons, NA) | Data Scientist (si besoin du détail brut) |
| `data/curated/ventes_enrichies.parquet` | Ventes + jointures + marges | **Data Scientist** (forecasting, pricing, reco) |
| `data/curated/kpis_mensuels.csv` | KPIs agrégés par mois | **BI Analyst** (dashboard) |
| `data/quality_reports/quality_report.md` | Rapport qualité lisible | **Chef de projet** (rapport final) |
| PostgreSQL (`ecommerce_dw`) | Tables relationnelles complètes | **Data Scientist / MLOps** (requêtes SQL, API) |

## 5. Se connecter au Data Warehouse (PostgreSQL)

**Attention** : le port par défaut (5432) est parfois occupé par un Postgres
déjà installé sur certaines machines (ça a été le cas sur la mienne). Le
projet utilise donc le port **5433** en local. Si ce port est libre chez
vous, vous pouvez remettre 5432 dans `docker-compose.yml` et
`load_to_warehouse.py` (variable `DB_PORT`).

**Identifiants :**
```
Host     : localhost
Port     : 5433
Database : ecommerce_dw
User     : dataeng
Password : dataeng_pwd
```

**Connexion Python (SQLAlchemy) :**
```python
from sqlalchemy import create_engine
engine = create_engine("postgresql+psycopg://dataeng:dataeng_pwd@localhost:5433/ecommerce_dw")
```

**Connexion via interface web (Adminer) :**
Ouvrir http://localhost:8081 (une fois `docker compose up -d` lancé)
- Système : PostgreSQL
- Serveur : `postgres`
- Utilisateur : `dataeng`
- Mot de passe : `dataeng_pwd`
- Base : `ecommerce_dw`

## 6. Schéma des tables (PostgreSQL)

**clients** : id_client (PK), nom, prenom, email, ville, age, date_inscription, segment
**produits** : id_produit (PK), nom_produit, categorie, prix, cout, stock
**promotions** : id_promo (PK), id_produit (FK), date_debut, date_fin, reduction_pct
**ventes** : id_vente (PK), id_client (FK), id_produit (FK), date_vente, quantite,
prix_paye, montant_total, marge_unitaire, marge_totale

## 7. Hypothèses prises sur les données synthétiques

- 500 clients, 150 produits, 40 promotions, ~15 000 ventes
- Période couverte : août 2024 à juillet 2025 (12 mois, pour capter la saisonnalité)
- Saisonnalité : pic de ventes en nov-décembre (fêtes), creux en juillet-août
- 4 segments clients (nouveau/occasionnel/fidèle/VIP) avec fréquences d'achat différenciées
- Bruit volontaire inclus : valeurs manquantes, doublons, quantités aberrantes
  (pour que les data quality checks aient un intérêt pédagogique réel)
- Toutes les données sont **synthétiques**, aucune donnée réelle utilisée (RGPD non applicable ici,
  mais à mentionner dans le rapport du chef de projet comme bonne pratique)

## 8. En cas de problème

- **Le batch tourne mais les nouvelles ventes streaming n'apparaissent pas** :
  relancer `build_datalake.py` (ou le flow complet) après le streaming, l'ordre compte.
- **Erreur de connexion PostgreSQL / mot de passe refusé sous Windows** :
  vérifier qu'un autre Postgres ne tourne pas déjà sur le port 5432
  (`netstat -aon | findstr :5432`). Si oui, on utilise le port 5433 (déjà configuré ici).
- **`ModuleNotFoundError` malgré `pip install`** : vérifier que le venv actif
  est bien celui du projet (`python -c "import sys; print(sys.executable)"`),
  et toujours utiliser `python -m pip install ...` plutôt que `pip install ...` seul.