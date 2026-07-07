"""
Physical and astronomical constants for kilonova light-curve modeling.

All values are given in CGS units (cm, g, s, K) unless otherwise noted.

"""

from astropy import constants as _const
from astropy import units as _u

__all__ = ["DAY", "Mpc", "pc", "AU"]

# --------------------------------------------------------------------------
# Time and distance unit conversions (unambiguous, no precision trade-off)
# --------------------------------------------------------------------------
DAY = 86400.0                                   # seconds per day
pc = _const.pc.cgs.value                        # cm per parsec
Mpc = _const.pc.cgs.value * 1.0e6               # cm per megaparsec
AU = _const.au.cgs.value                        # cm per astronomical unit
