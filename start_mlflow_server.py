"""Start the team's shared MLflow server on THIS computer.

    python start_mlflow_server.py              # port 5000, reachable from your network
    python start_mlflow_server.py --port 5001
    python start_mlflow_server.py --local-only # only you (127.0.0.1)

Keep this terminal open while teammates are training. Data is stored in ./mlflow_server/
(git-ignored). Stop with Ctrl+C.
"""
import argparse
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def lan_ip() -> str:
    """Best-effort local network address (no traffic is actually sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--local-only", action="store_true", help="do not accept connections from other computers")
    args = ap.parse_args()

    store = ROOT / "mlflow_server"
    (store / "artifacts").mkdir(parents=True, exist_ok=True)
    host = "127.0.0.1" if args.local_only else "0.0.0.0"
    ip = "127.0.0.1" if args.local_only else lan_ip()

    print("=" * 66)
    print(f" MLflow server starting.   Dashboard : http://{ip}:{args.port}")
    print(" Teammates run this in PowerShell (same Wi-Fi / network):")
    print(f'     $env:MLFLOW_TRACKING_URI = "http://{ip}:{args.port}"')
    print("     python check_mlflow.py")
    print(" If they cannot connect: allow Python through Windows Firewall (Private networks).")
    print(" Press Ctrl+C to stop.")
    print("=" * 66, flush=True)

    cmd = [sys.executable, "-m", "mlflow", "server",
           "--backend-store-uri", f"sqlite:///{(store / 'mlflow.db').as_posix()}",
           "--default-artifact-root", "mlflow-artifacts:/",
           "--artifacts-destination", (store / "artifacts").as_uri(),
           "--serve-artifacts", "--host", host, "--port", str(args.port)]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"Server exited with an error (code {e.returncode}). Is port {args.port} already in use? Try --port 5001")


if __name__ == "__main__":
    main()
