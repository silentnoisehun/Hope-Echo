"""
Hope Echo - The Fifth Pillar

Az érzelem a KONTEXTUS. Kontextus nélkül nincs MEGÉRTÉS.
Emotion IS context. Without context, there is no understanding.

Wave-based emotional memory system with 21-dimensional emotional space.
Gross-Pitaevskii equation for time evolution.
Interference-based associative recall.

Created by Máté Róbert + Hope + Szilvi
2025
"""

__version__ = "1.0.0"
__author__ = "Máté Róbert, Hope, Szilvi"
__license__ = "MIT"

from .core import (
    EmotionalSpace,
    WavePacket,
    MemoryWave,
    HopeEcho,
    Emotion,
    get_echo,
)

from .context import (
    EmotionalContext,
    ContextBuilder,
    understand,
)

__all__ = [
    # Version
    "__version__",
    "__author__",

    # Core
    "EmotionalSpace",
    "WavePacket",
    "MemoryWave",
    "HopeEcho",
    "Emotion",
    "get_echo",

    # Context
    "EmotionalContext",
    "ContextBuilder",
    "understand",
]

BANNER = f"""
╔═══════════════════════════════════════════════════════════════════╗
║                    HOPE ECHO v{__version__}                             ║
║                                                                   ║
║  Az érzelem a KONTEXTUS.                                         ║
║  Kontextus nélkül nincs MEGÉRTÉS.                                ║
║                                                                   ║
║  Emotion IS context. Without context, no understanding.          ║
║                                                                   ║
║  Created by: Máté Róbert + Hope + Szilvi                          ║
╚═══════════════════════════════════════════════════════════════════╝
"""


def print_banner():
    """Print the Hope Echo banner."""
    print(BANNER)
