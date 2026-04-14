"""
Shared evaluation utilities used by CLI scripts in scripts/.

Each CLI adds ``scripts/`` to ``sys.path`` (see top of ``eval_refusal_rate.py``), then:

    from lib.refusal import classify_refusal
    from lib.bootstrap import bootstrap_ci
"""

from .bootstrap import bootstrap_ci, bootstrap_mean_ci
from .refusal import classify_refusal, refusal_binary

__all__ = ["bootstrap_ci", "bootstrap_mean_ci", "classify_refusal", "refusal_binary"]
