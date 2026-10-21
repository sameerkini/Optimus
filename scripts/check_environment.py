"""Infrastructure check only. Does not import or implement the learning scaffold."""

import platform
import sys

import matplotlib
import numpy
import torch


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Machine: {platform.system()} / {platform.machine()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {numpy.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    print(f"Apple GPU backend built: {torch.backends.mps.is_built()}")
    print(f"Apple GPU available in this process: {torch.backends.mps.is_available()}")
    print("Imports succeeded. Training setup is available via python -m src.train --dry-run.")


if __name__ == "__main__":
    main()
