"""Build / refresh the role blueprint JSON artifact."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from redrob.blueprint import save_blueprint

if __name__ == "__main__":
    bp = save_blueprint()
    print(f"blueprint saved: {len(bp['core_competencies'])} core skills")
