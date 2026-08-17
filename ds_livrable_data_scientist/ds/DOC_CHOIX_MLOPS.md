# Doc de choix — Data Scientist
### Pour intégration par le rôle MLOps/Dev (API)

Ce document explique **quoi charger, comment appeler les modèles, et
quels formats attendre en entrée/sortie** — pas la démarche d'analyse
(voir les notebooks pour ça).

---

## 1. Modèle de prévision de la demande

### Deux modèles livrés

| Modèle | Fichier | Usage |
|---|---|---|
| Prophet (global) | `models/forecasting_prophet_global.pkl` | Répond à "combien va-t-on vendre le mois prochain ?" (toutes catégories) |
| LightGBM (par catégorie) | `models/forecasting_lightgbm_categorie.txt` + `models/forecasting_categorie_features.json` | Détail par catégorie, pour pricing/stock |

### Pourquoi deux modèles et pas un seul

Le sujet demande une prévision mensuelle globale (colonnes `date_vente`,
`quantite`, `annee_mois`) — Prophet y répond directement. Mais les
décisions de pricing/réassort se prennent par catégorie, d'où un second
modèle plus fin. **Sur cet historique (12 mois), les deux modèles ML sont
proches, voire légèrement en dessous, des baselines naïves** — voir
`outputs/forecasting_metrics.json`. C'est documenté et assumé : ne pas
survendre la précision de ces modèles dans le rapport business.

### Comment charger et appeler Prophet

```python
import pickle
import pandas as pd

with open("models/forecasting_prophet_global.pkl", "rb") as f:
    model = pickle.load(f)

# Prévoir N mois à partir de la dernière date connue à l'entraînement
futur = model.make_future_dataframe(periods=1, freq="MS")
forecast = model.predict(futur)
prochain_mois = forecast.tail(1)

quantite_prevue = max(0, round(prochain_mois["yhat"].values[0]))
borne_basse = max(0, round(prochain_mois["yhat_lower"].values[0]))
borne_haute = round(prochain_mois["yhat_upper"].values[0])
```

**Sortie attendue côté API** : ne pas exposer seulement `quantite_prevue`,
mais aussi l'intervalle `[borne_basse, borne_haute]` — c'est plus honnête
pour l'utilisateur final (dashboard BI) qu'un chiffre unique.

### Comment charger et appeler LightGBM (par catégorie)

```python
import json
import lightgbm as lgb

booster = lgb.Booster(model_file="models/forecasting_lightgbm_categorie.txt")
with open("models/forecasting_categorie_features.json") as f:
    meta = json.load(f)
    feature_cols = meta["feature_cols"]   # ordre exact des colonnes attendu par le modèle
    categories = meta["categories"]

# X doit être un DataFrame avec exactement les colonnes de feature_cols,
# dans cet ordre : mois_num, periode_fetes, lag_1, lag_2, rolling_mean_3,
# puis les one-hot cat_<categorie> pour chaque catégorie de `categories`.
prediction = booster.predict(X)
```

**Point d'attention pour le MLOps** : `lag_1`, `lag_2`, `rolling_mean_3`
doivent être recalculés à chaque appel à partir des 3 derniers mois réels
de ventes de la catégorie — ce ne sont pas des colonnes statiques.

---

## 2. Modèle de recommandation

### Fichiers

| Fichier | Contenu |
|---|---|
| `models/reco_similarite_hybride.npy` | Matrice de similarité produit × produit (150×150) |
| `models/reco_produits_index.json` | Ordre des `id_produit` correspondant aux lignes/colonnes de la matrice + poids `alpha` utilisé |

### Comment l'utiliser côté API

La matrice de similarité est **précalculée** (pas besoin de réentraîner à
chaque appel). L'API n'a besoin que de :
1. L'historique d'achats du client (déjà en base via le Data Warehouse)
2. La matrice de similarité + son index

```python
import json
import numpy as np

sim_hybride = np.load("models/reco_similarite_hybride.npy")
with open("models/reco_produits_index.json") as f:
    meta = json.load(f)
    produits_ids = meta["produits_ids"]  # ordre des lignes/colonnes de sim_hybride

id_to_idx = {pid: i for i, pid in enumerate(produits_ids)}

def recommander_api(id_client, historique_achats, produits_en_stock, k=5):
    """
    historique_achats : set des id_produit déjà achetés par ce client (depuis le DWH)
    produits_en_stock : set des id_produit actuellement en stock (depuis produits.stock > 0)
    """
    if not historique_achats:
        return []  # cold start total -> fallback sur "produits populaires" côté BI/dashboard

    idx_achetes = [id_to_idx[p] for p in historique_achats if p in id_to_idx]
    scores = sim_hybride[idx_achetes].mean(axis=0)

    exclure = {id_to_idx[p] for p in historique_achats if p in id_to_idx}
    exclure |= {i for i, p in enumerate(produits_ids) if p not in produits_en_stock}

    scores_valides = [(produits_ids[i], s) for i, s in enumerate(scores) if i not in exclure]
    scores_valides.sort(key=lambda x: -x[1])
    return [pid for pid, _ in scores_valides[:k]]
```

**Point d'attention pour le MLOps** : filtrer par `produits_en_stock` doit
se faire **à chaque appel** avec le stock à jour du DWH — la matrice
elle-même ne connaît pas le stock (calculée une fois, potentiellement
obsolète en quelques jours).

**Cold start** : un client sans aucun achat renvoie une liste vide dans
cette fonction API — c'est intentionnel (le notebook a une gestion de
repli sur les best-sellers, mais côté API il vaut mieux que le BI/dashboard
décide explicitement de la stratégie de fallback plutôt que le modèle
recommandation).

---

## 3. Fréquence de réentraînement suggérée

- **Forecasting** : à réentraîner chaque mois (nouvelles données
  disponibles = nouveau mois complet).
- **Recommandation** : la matrice de similarité peut être recalculée
  hebdomadairement (comportements d'achat qui évoluent plus vite),
  indépendamment du forecasting.

Aucun des deux modèles n'a de dépendance à un service externe (pas
d'appel API tiers) — l'inférence est locale et rapide (< 100ms par appel),
compatible avec un usage temps réel dans l'API FastAPI.
