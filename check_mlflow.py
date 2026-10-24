"""Check that you can reach the MLflow server you are pointed at.

    $env:MLFLOW_TRACKING_URI = "http://<server-ip>:5000"     # PowerShell
    python check_mlflow.py
"""
import os
import sys

# fail fast instead of retrying for minutes when the server is unreachable
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")

import mlflow
from mlflow.tracking import MlflowClient

uri = os.environ.get("MLFLOW_TRACKING_URI")
if not uri:
    print("[i] MLFLOW_TRACKING_URI is not set -> train.py will use your LOCAL mlflow.db.")
    print('    To use the team server (PowerShell):  $env:MLFLOW_TRACKING_URI = "http://<server-ip>:5000"')
    sys.exit(0)

print(f"Connecting to {uri} ...")
try:
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    experiments = client.search_experiments()
    models = client.search_registered_models()
except Exception as e:  # noqa: BLE001
    print(f"\n[x] Could not connect: {type(e).__name__}: {str(e)[:300]}")
    print("    - Is the server running (python start_mlflow_server.py)?")
    print("    - Correct IP and port? Same Wi-Fi / network? Windows Firewall allowing Python?")
    sys.exit(1)

print("\n[ok] Connected.\n\nExperiments:")
for e in experiments:
    n = len(client.search_runs([e.experiment_id], max_results=1000))
    print(f"   {e.name:<36} {n} run(s)")
print("\nRegistered models:")
if not models:
    print("   (none yet)")
for m in models:
    versions = client.search_model_versions(f"name='{m.name}'")
    print(f"   {m.name:<32} {len(versions)} version(s)   aliases: {dict(m.aliases) or '-'}")
