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

from src.env import ENV_PATH, load_env

load_env()  # reads MLFLOW_* from a local .env file if present

uri = os.environ.get("MLFLOW_TRACKING_URI")
if not uri:
    print("[i] MLFLOW_TRACKING_URI is not set -> train.py will use your LOCAL mlflow.db.")
    print('    To use the team server (PowerShell):  $env:MLFLOW_TRACKING_URI = "http://<server-ip>:5000"')
    sys.exit(0)

user = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
token_len = len(os.environ.get("MLFLOW_TRACKING_PASSWORD", ""))
print(f"Using .env file : {'yes' if ENV_PATH.exists() else 'NO (file not found next to train.py)'}")
print(f"Username        : {user or '(not set)'}")
print(f"Token length    : {token_len}   (a DagsHub access token is usually about 40 characters)")
if uri.startswith("https://") and token_len and token_len < 30:
    print("[!] That is too short for a DagsHub access token. Create one at https://dagshub.com/user/settings/tokens")
if ".git.mlflow" in uri:
    print("[!] The address contains '.git.mlflow'. It must end in '.mlflow' only: https://dagshub.com/<user>/<repo>.mlflow")
print(f"Connecting to {uri} ...")
try:
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    experiments = client.search_experiments()
    models = client.search_registered_models()
except Exception as e:  # noqa: BLE001
    msg = str(e)
    print(f"\n[x] Could not connect to {uri}")
    print(f"    {type(e).__name__}: {msg[:160].splitlines()[0]}\n")
    hosted = uri.startswith("https://")
    if "401" in msg:
        print("    401 = DagsHub rejected your login. Check .env:")
        print("      - MLFLOW_TRACKING_USERNAME must be your DagsHub USERNAME (not your email)")
        print("      - MLFLOW_TRACKING_PASSWORD must be a DagsHub ACCESS TOKEN (long string), from")
        print("        https://dagshub.com/user/settings/tokens  (no spaces, quotes or < > brackets)")
        print("      - Signed up with 'Sign in with GitHub'? Then you have NO password: use a token.")
    elif "403" in msg:
        print("    403 = you are logged in but not allowed. Ask Member 1 to add your DagsHub username")
        print("    as a collaborator on the DagsHub repo (Settings -> Collaborators).")
    elif "404" in msg:
        print("    404 = wrong address. It must look like https://dagshub.com/<user>/<repo>.mlflow")
        print("    (ends with .mlflow, NOT .git.mlflow) and the DagsHub repo must exist.")
    elif hosted:
        print("    - Is the address spelled exactly like the DagsHub repo, ending in .mlflow?")
        print("    - Is your internet / VPN working?")
    else:
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
