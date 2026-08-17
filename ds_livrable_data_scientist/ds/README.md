# Data Science — Forecasting & Recommandation

Livrable du rôle **Data Scientist** pour la plateforme de Data-Driven
Pricing & Recommandation. Construit sur les données fournies par le Data
Engineer (`data/curated/ventes_enrichies.parquet` + `data/raw/produits.csv`).

**Voir aussi `DOC_CHOIX_MLOPS.md`** — documentation dédiée à l'intégration
des modèles dans l'API (rôle MLOps/Dev).

## Contenu

```
data/                              -> copie des fichiers nécessaires (issus du Data Engineer)
notebooks/
  01_forecasting_demande.ipynb      -> Programme 1 : prévision de la demande
  02_recommandation_produits.ipynb  -> Programme 2 : recommandation de produits
  *.py                               -> mêmes contenus en script (format jupytext "percent",
                                        éditable/exécutable directement, plus facile à versionner)
models/                            -> modèles entraînés + artefacts réutilisables par l'API (MLOps)
outputs/                           -> métriques (JSON) + graphiques (PNG) pour le rapport business
DOC_CHOIX_MLOPS.md                 -> doc d'intégration détaillée pour le MLOps
```

## Programme 1 — Prévision de la demande (mensuelle)

Conforme au sujet : prédire "combien on va vendre le mois prochain" à
partir de `date_vente`, `quantite`, `annee_mois`.

- **Global** (toutes catégories) : **Prophet**, comme suggéré par le prof.
  Donne directement le chiffre du mois prochain avec un intervalle de
  confiance. Résultat : ~1253 unités prévues pour le mois suivant
  l'historique (intervalle 680-1804).
- **Par catégorie** : **LightGBM**, pour affiner par catégorie
  (pricing/réassort). Comparé à une baseline naïve — sur cet historique
  court (12 mois), les baselines restent compétitives, c'est documenté et
  assumé dans le notebook plutôt que masqué.
- **Artefacts** : `models/forecasting_prophet_global.pkl`,
  `models/forecasting_lightgbm_categorie.txt` +
  `models/forecasting_categorie_features.json`.

## Programme 2 — Recommandation de produits

Conforme au sujet : colonnes `id_client`, `id_produit`, `segment`,
`categorie`, approche hybride content-based + collaborative filtering.

- **Collaboratif** (60%) : factorisation SVD sur la matrice client × produit.
- **Contenu** (40%) : similarité par catégorie + prix.
- **Filtre stock** : les produits en rupture (`stock = 0`, colonne staging
  du Data Engineer) sont exclus des recommandations.
- **Évaluation** : precision@k / recall@k (499 clients évaluables), le
  modèle hybride bat la baseline "best-sellers" (+20% de recall relatif à k=5).
- **Artefacts** : `models/reco_similarite_hybride.npy`,
  `models/reco_produits_index.json`.

## Comment relancer

```bash
pip install pandas numpy lightgbm prophet scikit-learn matplotlib pyarrow jupytext nbclient ipykernel
cd notebooks
python 01_forecasting_demande.py     # ou : jupyter notebook 01_forecasting_demande.ipynb
python 02_recommandation_produits.py
```

## Ce qui n'est PAS dans ce livrable (hors périmètre Data Scientist)

- Pipeline ETL / data lake / warehouse → Data Engineer
- API Docker, déploiement, monitoring → MLOps/Dev
- Dashboard, KPIs agrégés, rapport ROI → BI/Business Analyst
- Cadrage, planning, RGPD → Chef de projet
