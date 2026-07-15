"""Point DATA_DIR at a throwaway directory BEFORE any autovid import.

autovid.config resolves DATA_DIR at import time, so this must happen at
conftest import (pytest loads conftest before test modules). Tests therefore
never touch the developer's real channels/ and projects/.
"""

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="autovid-tests-")
os.environ["DATA_DIR"] = _TMP
os.environ.pop("DATABASE_URL", None)  # default backend under test is the filesystem
