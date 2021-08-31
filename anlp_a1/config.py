import os
from pathlib import Path

# this comes handy in many places
DATA_ROOT = os.getenv("ANLP_A1_DATA_ROOT", Path(__file__).parent.parent)
