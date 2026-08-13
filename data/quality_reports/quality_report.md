# Rapport de qualité des données

Généré le : 2026-08-13T23:33:22.110977

**Statut global : OK**


## Table `clients`

| Check | Statut | Détail |
|---|---|---|
| completude_nom | ✅ OK | 0 valeurs manquantes (0.0%) |
| completude_prenom | ✅ OK | 0 valeurs manquantes (0.0%) |
| completude_ville | ✅ OK | 0 valeurs manquantes (0.0%) |
| completude_segment | ✅ OK | 0 valeurs manquantes (0.0%) |
| unicite_id_client | ✅ OK | 0 doublons détectés sur id_client |
| plage_age | ✅ OK | 0 valeurs hors plage attendue (age >= 18 & <= 90) |

## Table `produits`

| Check | Statut | Détail |
|---|---|---|
| completude_nom_produit | ✅ OK | 0 valeurs manquantes (0.0%) |
| completude_categorie | ✅ OK | 0 valeurs manquantes (0.0%) |
| completude_prix | ✅ OK | 0 valeurs manquantes (0.0%) |
| unicite_id_produit | ✅ OK | 0 doublons détectés sur id_produit |
| plage_prix | ✅ OK | 0 valeurs hors plage attendue (prix > 0) |
| plage_stock | ✅ OK | 0 valeurs hors plage attendue (stock >= 0) |

## Table `promotions`

| Check | Statut | Détail |
|---|---|---|
| unicite_id_promo | ✅ OK | 0 doublons détectés sur id_promo |
| coherence_id_produit | ✅ OK | 0 lignes avec id_produit inexistant dans produits |

## Table `ventes`

| Check | Statut | Détail |
|---|---|---|
| completude_id_client | ✅ OK | 0 valeurs manquantes (0.0%) |
| completude_id_produit | ✅ OK | 0 valeurs manquantes (0.0%) |
| completude_date_vente | ✅ OK | 0 valeurs manquantes (0.0%) |
| unicite_id_vente | ✅ OK | 0 doublons détectés sur id_vente |
| coherence_id_client | ✅ OK | 0 lignes avec id_client inexistant dans clients |
| coherence_id_produit | ✅ OK | 0 lignes avec id_produit inexistant dans produits |
| plage_quantite | ✅ OK | 0 valeurs hors plage attendue (quantite > 0 & <= 20) |
| plage_prix_paye | ✅ OK | 0 valeurs hors plage attendue (prix_paye > 0) |
| fraicheur | ✅ OK | Dernière donnée: 2026-08-13 (0 jours d'écart) |