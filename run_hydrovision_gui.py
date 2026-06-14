from __future__ import annotations

import argparse
import importlib.util
import socket
import subprocess
import sys
from pathlib import Path


def ensure_streamlit() -> None:
    missing = [name for name in ("streamlit", "numpy", "tifffile") if importlib.util.find_spec(name) is None]
    if not missing:
        return
    req_file = Path(__file__).with_name("requirements.txt")
    if req_file.exists():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "numpy", "tifffile"])


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def pick_port(preferred: int, search_range: int = 50) -> int:
    for offset in range(search_range + 1):
        port = preferred + offset
        if is_port_available(port):
            return port
    raise RuntimeError(f"No free port found from {preferred} to {preferred + search_range}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch HYDROVISION Streamlit GUI.")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--port-search-range", type=int, default=50)
    args = parser.parse_args()

    app_path = Path(__file__).with_name("hydrovision_gui.py")
    if not app_path.exists():
        raise FileNotFoundError(f"HYDROVISION GUI file not found: {app_path}")

    ensure_streamlit()
    port = pick_port(args.port, args.port_search_range)
    if port != args.port:
        print(f"[run] requested port {args.port} is busy, using {port} instead.")

    print(f"[run] starting HYDROVISION GUI on http://localhost:{port}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            str(port),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
