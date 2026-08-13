"""
Étape 7 - Orchestration avec Prefect

Enchaîne automatiquement toutes les étapes du pipeline Data Engineer :
  1. Construction du data lake (raw -> staging -> curated)
  2. Data quality checks
  3. Chargement dans le Data Warehouse (PostgreSQL)

Chaque étape est une "task" Prefect : si une étape échoue, les suivantes
ne sont pas exécutées, et Prefect log clairement où ça a cassé.

Prérequis :
    pip install prefect

Usage :
    python orchestration_flow.py              # lance le flow une fois
    prefect server start                       # (optionnel) interface web locale
                                                 # -> http://localhost:4200
"""

import subprocess
import sys
from prefect import flow, task
from prefect.logging import get_run_logger


def run_script(script_name: str):
    """Exécute un script Python avec le même interpréteur que le flow actuel
    (évite les problèmes de venv qu'on a eus avec subprocess + 'python')."""
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} a échoué:\n{result.stderr}")
    return result.stdout


@task(name="build_datalake", retries=1, retry_delay_seconds=5)
def build_datalake_task():
    logger = get_run_logger()
    logger.info("Construction du data lake (raw -> staging -> curated)...")
    output = run_script("build_datalake.py")
    logger.info(output)
    return "OK"


@task(name="data_quality_checks")
def data_quality_task():
    logger = get_run_logger()
    logger.info("Vérification de la qualité des données...")
    output = run_script("data_quality_checks.py")
    logger.info(output)

    if "STATUT GLOBAL : ERROR" in output:
        raise RuntimeError("Data quality check en ERREUR - pipeline arrêté avant chargement en base.")
    if "STATUT GLOBAL : WARNING" in output:
        logger.warning("Data quality check avec WARNING - chargement poursuivi mais à surveiller.")
    return "OK"


@task(name="load_to_warehouse", retries=2, retry_delay_seconds=10)
def load_to_warehouse_task():
    logger = get_run_logger()
    logger.info("Chargement dans le Data Warehouse PostgreSQL...")
    output = run_script("load_to_warehouse.py")
    logger.info(output)
    return "OK"


@flow(name="pipeline_data_engineer_ecommerce")
def pipeline_complet():
    """Flow principal : orchestre tout le pipeline Data Engineer, du data lake
    jusqu'au data warehouse, avec arrêt automatique si la qualité des données
    est mauvaise."""
    build_datalake_task()
    data_quality_task()
    load_to_warehouse_task()


if __name__ == "__main__":
    pipeline_complet()