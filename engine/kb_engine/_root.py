import os
from pathlib import Path

# Corpus + index/artifact root. Defaults to the current working directory so the
# engine operates on the *consuming agent's* directory; the agent's shim sets
# KB_ROOT to its own location. Evaluated at import — the shim sets the env var
# before importing kb_engine.
KB_ROOT = Path(os.environ.get("KB_ROOT") or Path.cwd()).resolve()
