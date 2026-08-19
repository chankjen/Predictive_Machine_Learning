import json
from pathlib import Path


notebook_path = Path("Week_9_Predictive_Maintenance_Lab.ipynb")
nb = json.loads(notebook_path.read_text(encoding="utf-8"))

setup_cell = nb["cells"][2]
source = "".join(setup_cell["source"])

if "import sys" not in source:
    source = source.replace(
        "from pathlib import Path\n",
        "from pathlib import Path\nimport sys\n",
        1,
    )
if "NOTEBOOK_DEPS = Path.cwd() / \"notebook_deps\"" not in source:
    source = source.replace(
        "import warnings\n",
        """import warnings

# Notebook environment guard:
# Put workspace-local packages first. This fixes kernels that can see
# matplotlib but lost numpy in the user-site package directory.
NOTEBOOK_DEPS = Path.cwd() / "notebook_deps"
if NOTEBOOK_DEPS.exists():
    sys.path.insert(0, str(NOTEBOOK_DEPS))
""",
        1,
    )

setup_cell["source"] = source.splitlines(keepends=True)
notebook_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Patched {notebook_path.resolve()}")
