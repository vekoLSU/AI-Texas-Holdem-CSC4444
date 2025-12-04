"""
Training utilities for the poker bot.

Exposes helper classes for logging decisions, encoding state, datasets, and
reference neural network architectures that can be trained via self-play or
supervised learning.
"""

from .data_collector import TrainingRecorder  # noqa: F401
from .state_encoder import StateEncoder  # noqa: F401

