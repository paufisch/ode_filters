import sys
from pathlib import Path

# Add the parent directory to the path so pytest can find ode_filters
sys.path.insert(0, str(Path(__file__).parent.parent))
