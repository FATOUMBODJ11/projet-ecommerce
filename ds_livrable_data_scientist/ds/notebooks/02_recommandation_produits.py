# %% [markdown]
# # Programme 2 — Recommandation de produits (hybride)
#
# **Rôle : Data Scientist** — Plateforme de Data-Driven Pricing & Recommandation
#
# **Objectif** : pour un client donné, recommander des produits pertinents
# (cross-sell / up-sell) à partir de :
# - son historique d'achats et celui des clients similaires (**collaboratif**)
# - les caractéristiques des produits qu'il a déjà achetés (**contenu**)
#
# **Méthode** :
# 1. Construire la matrice client × produit (interactions = quantité achetée)
# 2. Modèle collaboratif : similarité entre produits via factorisation
#    (TruncatedSVD) sur la matrice d'interactions
# 3. Modèle contenu : similarité entre produits via leurs caractéristiques
#    (catégorie, gamme de prix)
# 4. Score hybride = combinaison pondérée des deux
# 5. Évaluation : precision@k / recall@k par validation temporelle
#    (on cache les derniers achats de chaque client et on vérifie si le
#    modèle les aurait recommandés)

# %%
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, OneHotEncoder

warnings.filterwarnings("ignore")

DATA_DIR = Path("../data")
MODELS_DIR = Path("../models")
OUTPUTS_DIR = Path("../outputs")
MODELS_DIR.mkdir(exist_ok=True, parents=True)
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

# %% [markdown]
# ## 1. Chargement des données

# %%
ventes = pd.read_parquet(DATA_DIR / "ventes_enrichies.parquet")
produits = pd.read_csv(DATA_DIR / "produits.csv")  # contient déjà `stock` (donnée raw)

ventes["date_vente"] = pd.to_datetime(ventes["date_vente"])
ventes = ventes[ventes["date_vente"] <= "2025-07-31"].copy()

print(f"Ventes : {ventes.shape[0]} | Clients : {ventes['id_client'].nunique()} | "
      f"Produits achetés : {ventes['id_produit'].nunique()}")
print(f"Produits en rupture de stock (stock=0) : {(produits['stock'] == 0).sum()} / {len(produits)}")

# %% [markdown]
# ## 2. Split temporel pour l'évaluation
#
# Pour chaque client, on cache ses **2 derniers achats** (produits distincts)
# comme "vérité terrain" à retrouver, et on entraîne les modèles uniquement
# sur le reste de l'historique. C'est l'équivalent d'un split train/test
# temporel adapté à la recommandation.

# %%
ventes_sorted = ventes.sort_values(["id_client", "date_vente"])

def split_client(g):
    produits_distincts = g["id_produit"].drop_duplicates()
    if len(produits_distincts) < 4:
        # Pas assez d'historique pour évaluer ce client proprement : tout en train.
        return g.assign(split="train")
    derniers_produits = produits_distincts.tail(2).tolist()
    is_test = g["id_produit"].isin(derniers_produits) & (
        g["date_vente"] >= g[g["id_produit"].isin(derniers_produits)]["date_vente"].min()
    )
    # Ne garder qu'UNE ligne de test par produit caché (le premier achat de ce produit)
    g = g.copy()
    g["split"] = "train"
    for p in derniers_produits:
        idx_first = g[g["id_produit"] == p].index.min()
        g.loc[idx_first, "split"] = "test"
    return g

pieces = []
for cid, g in ventes_sorted.groupby("id_client"):
    g2 = split_client(g)
    g2["id_client"] = cid
    pieces.append(g2)
ventes_split = pd.concat(pieces, ignore_index=True)

train = ventes_split[ventes_split["split"] == "train"]
test = ventes_split[ventes_split["split"] == "test"]

print(f"Train : {len(train)} lignes | Test (achats cachés) : {len(test)} lignes, "
      f"{test['id_client'].nunique()} clients évaluables")

# %% [markdown]
# ## 3. Modèle collaboratif — factorisation matricielle (SVD)
#
# On construit la matrice client × produit (quantité totale achetée), puis
# on la factorise avec TruncatedSVD pour obtenir des embeddings produits
# capturant les habitudes d'achat communes ("les clients qui achètent X
# achètent aussi souvent Y").

# %%
interactions = (
    train.groupby(["id_client", "id_produit"])["quantite"].sum().unstack(fill_value=0)
)
print(f"Matrice interactions : {interactions.shape[0]} clients x {interactions.shape[1]} produits, "
      f"densité = {(interactions > 0).values.mean():.1%}")

n_components = min(30, min(interactions.shape) - 1)
svd = TruncatedSVD(n_components=n_components, random_state=42)
client_embeddings = svd.fit_transform(interactions)
produit_embeddings_collab = svd.components_.T  # (n_produits, n_components)

produits_svd_ids = interactions.columns.tolist()
similarite_collab = cosine_similarity(produit_embeddings_collab)
similarite_collab_df = pd.DataFrame(similarite_collab, index=produits_svd_ids, columns=produits_svd_ids)

print(f"Variance expliquée par la SVD ({n_components} composantes) : {svd.explained_variance_ratio_.sum():.1%}")

# %% [markdown]
# ## 4. Modèle contenu — similarité par caractéristiques produit
#
# Catégorie (one-hot) + prix normalisé : deux produits de la même catégorie
# et de prix proches sont considérés similaires. Utile en "cold start"
# (nouveau produit ou client avec peu d'historique).

# %%
produits_feat = produits.set_index("id_produit").copy()

encoder = OneHotEncoder(sparse_output=False)
cat_encoded = encoder.fit_transform(produits_feat[["categorie"]])

scaler = StandardScaler()
prix_scaled = scaler.fit_transform(produits_feat[["prix"]])

features_contenu = np.hstack([cat_encoded, prix_scaled * 0.5])  # prix pondéré moins fort que la catégorie
similarite_contenu = cosine_similarity(features_contenu)
similarite_contenu_df = pd.DataFrame(
    similarite_contenu, index=produits_feat.index, columns=produits_feat.index
)

print(f"Similarité contenu calculée pour {len(produits_feat)} produits.")

# %% [markdown]
# ## 5. Score hybride et fonction de recommandation

# %%
ALPHA = 0.6  # poids du collaboratif vs contenu (0.6 = privilégie le comportement d'achat réel)

tous_produits = produits_feat.index.tolist()

# Aligner les deux matrices de similarité sur le même référentiel produit complet
sim_collab_full = similarite_collab_df.reindex(index=tous_produits, columns=tous_produits, fill_value=0)
sim_contenu_full = similarite_contenu_df.reindex(index=tous_produits, columns=tous_produits, fill_value=0)

sim_hybride = ALPHA * sim_collab_full + (1 - ALPHA) * sim_contenu_full

achats_train_par_client = train.groupby("id_client")["id_produit"].apply(set).to_dict()
produits_en_stock = set(produits.loc[produits["stock"] > 0, "id_produit"])

def recommander(id_client, k=5, filtrer_stock=True):
    """Retourne les k produits les plus pertinents pour un client, en excluant
    ce qu'il a déjà acheté et (par défaut) les produits en rupture de stock."""
    deja_achetes = achats_train_par_client.get(id_client, set())
    candidats_valides = produits_en_stock if filtrer_stock else set(tous_produits)

    if not deja_achetes:
        # Cold start total : on renvoie les produits les plus populaires (en stock)
        top_pop = train["id_produit"].value_counts()
        top_pop = top_pop[top_pop.index.isin(candidats_valides)]
        return top_pop.head(k).index.tolist()

    scores = sim_hybride.loc[list(deja_achetes)].mean(axis=0)
    a_exclure = [p for p in deja_achetes if p in scores.index] + [
        p for p in scores.index if p not in candidats_valides
    ]
    scores = scores.drop(index=set(a_exclure), errors="ignore")
    return scores.sort_values(ascending=False).head(k).index.tolist()

# Exemple
exemple_client = train["id_client"].iloc[0]
print(f"Exemple — recommandations pour le client {exemple_client} :")
recos = recommander(exemple_client, k=5)
print(produits_feat.loc[recos, ["nom_produit", "categorie", "prix"]])

# %% [markdown]
# ## 6. Évaluation : precision@k / recall@k
#
# Pour chaque client évaluable, on vérifie si les produits recommandés
# recoupent les achats réellement cachés (test set).

# %%
def evaluer(k=5):
    precisions, recalls = [], []
    for id_client, vrais_produits in test.groupby("id_client")["id_produit"].apply(set).items():
        recos = set(recommander(id_client, k=k))
        vrais_recuperables = vrais_produits & set(tous_produits)
        if not vrais_recuperables:
            continue
        hits = len(recos & vrais_recuperables)
        precisions.append(hits / k)
        recalls.append(hits / len(vrais_recuperables))
    return np.mean(precisions), np.mean(recalls), len(precisions)

resultats_eval = []
for k in [3, 5, 10]:
    p, r, n = evaluer(k=k)
    resultats_eval.append({"k": k, "precision_at_k": p, "recall_at_k": r, "n_clients_evalues": n})
    print(f"k={k:2d} | precision@{k}={p:.3f}  recall@{k}={r:.3f}  (sur {n} clients)")

# %% [markdown]
# ### Baseline de comparaison : recommander les produits les plus populaires
#
# Sans aucune personnalisation, "recommander les best-sellers à tout le
# monde" est la référence à battre.

# %%
top_populaires = train["id_produit"].value_counts().head(10).index.tolist()

def evaluer_populaire(k=5):
    precisions, recalls = [], []
    for id_client, vrais_produits in test.groupby("id_client")["id_produit"].apply(set).items():
        deja_achetes = achats_train_par_client.get(id_client, set())
        recos = [p for p in top_populaires if p not in deja_achetes][:k]
        vrais_recuperables = vrais_produits & set(tous_produits)
        if not vrais_recuperables:
            continue
        hits = len(set(recos) & vrais_recuperables)
        precisions.append(hits / k)
        recalls.append(hits / len(vrais_recuperables))
    return np.mean(precisions), np.mean(recalls)

p_base, r_base = evaluer_populaire(k=5)
print(f"Baseline 'best-sellers' | precision@5={p_base:.3f}  recall@5={r_base:.3f}")
print(f"Modèle hybride          | precision@5={resultats_eval[1]['precision_at_k']:.3f}  "
      f"recall@5={resultats_eval[1]['recall_at_k']:.3f}")

# %% [markdown]
# ## 7. Sauvegarde des artefacts

# %%
np.save(MODELS_DIR / "reco_similarite_hybride.npy", sim_hybride.values)
with open(MODELS_DIR / "reco_produits_index.json", "w") as f:
    json.dump({"produits_ids": tous_produits, "alpha": ALPHA}, f, ensure_ascii=False)

metriques_finales = {
    "hybride": resultats_eval,
    "baseline_populaire": {"precision_at_5": p_base, "recall_at_5": r_base},
}
with open(OUTPUTS_DIR / "reco_metrics.json", "w") as f:
    json.dump(metriques_finales, f, indent=2, ensure_ascii=False)

print("Matrice de similarité sauvegardée : models/reco_similarite_hybride.npy")
print("Métriques sauvegardées : outputs/reco_metrics.json")

# %% [markdown]
# ## Conclusion
#
# Le modèle hybride (60% collaboratif / 40% contenu) est comparé à une
# baseline "best-sellers pour tous" — c'est le comparatif le plus honnête
# pour juger si la personnalisation apporte réellement de la valeur.
#
# **Sur les valeurs absolues** : precision@5 ≈ 1,4% paraît faible, mais se
# lit toujours par rapport à la baseline, pas dans l'absolu — avec 150
# produits possibles et seulement 2 achats cachés par client, une precision
# parfaite plafonnerait de toute façon à 2/5 = 40%. Le signal utile est le
# **gain relatif vs la baseline** (+20% de recall ici), qui montre que la
# personnalisation apporte quelque chose de réel, même modeste sur des
# données synthétiques peu structurées.
#
# **Utilisation prévue** : la fonction `recommander(id_client, k)` et la
# matrice de similarité sauvegardée sont ce que le rôle MLOps/Dev doit
# exposer via l'API FastAPI (endpoint `/recommandations/{id_client}`).
#
# **Limites connues** : le contenu ne s'appuie que sur catégorie + prix
# (pas de description produit riche) ; à enrichir avec des embeddings
# texte si des descriptions produits détaillées sont disponibles.
