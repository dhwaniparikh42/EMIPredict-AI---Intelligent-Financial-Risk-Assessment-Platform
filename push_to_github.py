"""One-command, beginner-safe push to GitHub (Windows / macOS / Linux).

    python push_to_github.py -m "Add model training"            # push to current branch
    python push_to_github.py -m "Add EDA" --branch feature/eda  # push to your own branch (recommended for teams)
    python push_to_github.py --remote https://github.com/<user>/<repo>.git -m "First commit"

What it does: init repo if needed -> set remote -> safety-check file sizes ->
git add -> commit -> pull --rebase (to pick up teammates' work) -> push.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

WARN_MB, BLOCK_MB = 50, 95            # GitHub rejects files > 100 MB
NEVER_COMMIT = ("data/raw/", "data/processed/", "mlflow.db", "mlartifacts/", ".venv/")
# a real (non-placeholder) MLflow/DagsHub token written into ANY file, e.g. pasted into .env.example by mistake
SECRET_RE = re.compile(r"MLFLOW_TRACKING_PASSWORD\s*=\s*[\"']?(?!<)[^\s\"'<]{8,}")


def git(*args, check=True, capture=True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], check=check, text=True,
                          capture_output=capture)


def out(*args) -> str:
    return git(*args, check=False).stdout.strip()


def die(msg: str) -> None:
    print(f"\n[x] {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--message", help="commit message")
    ap.add_argument("--branch", help="branch to commit/push to (created if missing)")
    ap.add_argument("--remote", help="GitHub repo URL (only needed the first time)")
    ap.add_argument("--no-pull", action="store_true", help="skip 'git pull --rebase'")
    args = ap.parse_args()

    # 0. git installed?
    try:
        git("--version")
    except (FileNotFoundError, subprocess.CalledProcessError):
        die("Git is not installed. Get it from https://git-scm.com/downloads and restart VS Code.")

    # 1. repo + identity
    if not os.path.isdir(".git"):
        print("[i] No git repo yet -> running 'git init'")
        git("init")
        git("branch", "-M", "main")
    if not out("config", "user.name") or not out("config", "user.email"):
        die('Set your identity once:\n    git config --global user.name "Your Name"\n'
            '    git config --global user.email "you@example.com"')

    # 2. remote
    remote = out("remote", "get-url", "origin")
    if not remote:
        url = args.remote or input("Paste your GitHub repo URL (https://github.com/<user>/<repo>.git): ").strip()
        if not url:
            die("A GitHub URL is required the first time.")
        git("remote", "add", "origin", url)
        remote = url
    print(f"[i] remote: {remote}")

    # 3. branch
    if args.branch:
        if git("rev-parse", "--verify", args.branch, check=False).returncode == 0:
            git("checkout", args.branch)
        else:
            git("checkout", "-b", args.branch)
    branch = out("symbolic-ref", "--short", "HEAD") or "main"
    print(f"[i] branch: {branch}")

    # 4. stage + safety checks
    git("add", "-A")
    staged = [f for f in out("diff", "--cached", "--name-only").splitlines() if f]
    if not staged:
        print("[i] Nothing new to commit.")
    else:
        bad = [f for f in staged if f.startswith(NEVER_COMMIT) or os.path.basename(f) == ".env"]
        if bad:
            git("reset", "-q")
            die("These should never be committed (data / local MLflow files):\n  " + "\n  ".join(bad[:10])
                + "\nCheck your .gitignore, then run again.")
        leaks = []
        for f in staged:
            if os.path.isfile(f) and os.path.getsize(f) < 1_000_000:
                with open(f, encoding="utf-8", errors="ignore") as fh:
                    if SECRET_RE.search(fh.read()):
                        leaks.append(f)
        if leaks:
            git("reset", "-q")
            die("A real MLflow/DagsHub token seems to be written in:\n  " + "\n  ".join(leaks)
                + "\nPut real values ONLY in .env (git-ignored). In files like .env.example keep the "
                  "<placeholders>. Fix the file, then run again.")
        big = [(f, os.path.getsize(f) / 1e6) for f in staged if os.path.exists(f)]
        too_big = [(f, s) for f, s in big if s > BLOCK_MB]
        if too_big:
            git("reset", "-q")
            die("GitHub blocks files over 100 MB. Too large:\n  "
                + "\n  ".join(f"{f}  ({s:.0f} MB)" for f, s in too_big)
                + "\nAdd them to .gitignore or use Git LFS.")
        for f, s in big:
            if s > WARN_MB:
                print(f"[!] large file: {f} ({s:.0f} MB)")

        print(f"[i] committing {len(staged)} file(s)")
        msg = args.message or input("Commit message: ").strip() or "Update project"
        git("commit", "-m", msg)

    # 5. pick up teammates' work, then push
    has_remote_branch = bool(out("ls-remote", "--heads", "origin", branch))
    if has_remote_branch and not args.no_pull:
        print("[i] pulling latest changes (rebase) ...")
        r = git("pull", "--rebase", "origin", branch, check=False)
        if r.returncode != 0:
            print(r.stdout, r.stderr)
            die("Merge conflict while pulling. Fix the conflicted files in VS Code, then run:\n"
                "    git add -A && git rebase --continue\n    python push_to_github.py")

    print("[i] pushing ...")
    r = git("push", "-u", "origin", branch, check=False)
    if r.returncode != 0:
        print(r.stderr)
        die("Push failed. Common causes: wrong repo URL, not logged in to GitHub "
            "(use a Personal Access Token as the password, or install GitHub CLI: 'gh auth login'), "
            "or you lack write access to the repo.")
    print(f"\n[ok] Pushed to {remote} ({branch})")
    if "github.com" in remote and branch != "main":
        repo = remote.removesuffix(".git")
        print(f"     Open a Pull Request: {repo}/pull/new/{branch}")


if __name__ == "__main__":
    main()
