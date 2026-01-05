"""
API routes package.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from . import control
    from . import state
    from . import voice
    from . import system
except ImportError:
    import control
    import state
    import voice
    import system

__all__ = ['control', 'state', 'voice', 'system']
