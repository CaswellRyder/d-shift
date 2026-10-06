"""Run with: python webcam_app.py --allow-unvalidated"""

import os
from pathlib import Path

from dtr.web import main

if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
