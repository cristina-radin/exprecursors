"""Importing tests.synthetic sets MHW_DATA_FILE / MHW_CLIM_FILE to synthetic
files before any src module is imported (src.utils.paths reads them at import).
This overrides real values for the pytest process only."""

from tests import synthetic  # noqa: F401
