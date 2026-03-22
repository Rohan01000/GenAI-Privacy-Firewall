"""
setup.py  —  GenAI Privacy Firewall setup & launcher
=====================================================
Usage:
    python setup.py setup     Install deps + generate data + train model
    python setup.py data      Generate PII dataset only
    python setup.py train     Train BiLSTM-CRF model only
    python setup.py server    Start the FastAPI server (default)
"""

import os
import subprocess
import sys

# Always run from project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))

CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
RESET = "\033[0m"


def banner(msg):
    print(f"\n{CYAN}{'=' * 50}")
    print(f"  {msg}")
    print(f"{'=' * 50}{RESET}\n")


def run(cmd, desc=None):
    if desc:
        print(f"  {YELLOW}> {desc}{RESET}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n  {RED}Command failed: {cmd}{RESET}")
        sys.exit(1)


def install_deps():
    banner("1/3 — Installing dependencies")
    run(f"{sys.executable} -m pip install -r requirements.txt", "pip install -r requirements.txt")


def generate_data():
    banner("2/3 — Generating PII dataset (5000 sentences)")
    run(f"{sys.executable} data/generate_dataset.py", "python data/generate_dataset.py")


def train_model():
    banner("3/3 — Training BiLSTM-CRF model")
    run(f"{sys.executable} model/train.py", "python model/train.py")


def start_server():
    banner("Starting GenAI Privacy Firewall")
    print(f"  {YELLOW}Chat      : http://localhost:8000/chat.html{RESET}")
    print(f"  {YELLOW}Dashboard : http://localhost:8000/dashboard.html{RESET}")
    print(f"  {YELLOW}Audit     : http://localhost:8000/audit.html{RESET}")
    print(f"  {YELLOW}API       : http://localhost:8000/chat{RESET}")
    print(f"  {YELLOW}Health    : http://localhost:8000/health{RESET}")
    print()
    run(f"{sys.executable} -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload")


def full_setup():
    install_deps()
    generate_data()
    train_model()
    print(f"\n  {GREEN}Setup complete! Run: python setup.py server{RESET}\n")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "server"

    commands = {
        "setup":  full_setup,
        "data":   generate_data,
        "train":  train_model,
        "server": start_server,
    }

    if cmd in commands:
        commands[cmd]()
    else:
        print(f"{RED}Unknown command: {cmd}{RESET}")
        print(f"Usage: python setup.py [setup|data|train|server]")
        sys.exit(1)
