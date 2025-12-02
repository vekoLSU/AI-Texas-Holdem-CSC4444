"""
Integration components for connecting trained RL agent to competition infrastructure.
"""

from .state_translator import StateTranslator
from .action_translator import ActionTranslator

__all__ = ['StateTranslator', 'ActionTranslator']
