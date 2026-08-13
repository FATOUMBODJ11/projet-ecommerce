"""
Étape 4a - Ingestion BATCH

Simule un job qui tourne périodiquement (ex: un cron nocturne) et qui
relance tout le pipeline raw -> staging -> curated.

En production, ce script serait déclenché par un orchestrateur (Airflow,
Prefect, cron system) toutes les nuits par exemple. Ici on le lance
manuellement pour la démo, mais la logique est la même.

Usage :
    python ingestion_batch.py
"""

import subprocess
import json
import os
import sys
from datetime import datetime

LOG_FILE = "data/batch_runs.json"


def run_batch():
    start = datetime.now()
    print(f"[BATCH] Démarrage du run à {start.isoformat()}")

    # Étape 1 : régénère/actualise les données brutes si besoin
    # (en prod, ce serait un connecteur vers une vraie source : ERP, CRM, etc.
    #  ici, on suppose que raw/ est déjà alimenté et on relance juste le
    #  nettoyage + enrichissement)
    print("[BATCH] Exécution du pipeline raw -> staging -> curated...")
    result = subprocess.run(
        [sys.executable, "build_datalake.py"],
        capture_output=True, text=True
    )

    success = result.returncode == 0
    end = datetime.now()
    duration = (end - start).total_seconds()

    run_info = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "duration_sec": duration,
        "success": success,
        "stdout_tail": result.stdout[-500:] if result.stdout else "",
        "stderr_tail": result.stderr[-500:] if result.stderr else "",
    }

    # historique des runs (append, ne jamais écraser)
    history = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            history = json.load(f)
    history.append(run_info)
    with open(LOG_FILE, "w") as f:
        json.dump(history, f, indent=2)

    status = "OK" if success else "ÉCHEC"
    print(f"[BATCH] Terminé en {duration:.2f}s - statut: {status}")
    if not success:
        print("[BATCH] Erreur détectée :", result.stderr[-500:])

    return success


if __name__ == "__main__":
    run_batch()