from . import common
from . import palette
from . import organic
from . import boolean
from . import forcefield
from . import lsystem
from . import lod_gen
from . import softbody
from . import remesh
from . import autorig
from . import hair
from . import generators
from . import physics_unified
from . import baking
from . import batch_export
from . import thumbnail
from . import llm_brain
from . import furniture
from . import vrm_export
from . import terrain_pro
from . import sculpt_details
from . import pbr_materials
from . import weapons
from . import safety

# ★ v2.0: 新しい統合システム ★
from . import unified_memory
from . import aether_core
from . import evaluator_v2

# ★ v3.0: 10000D統合システム ★
from . import llm_prompt_10000d_v2
from . import universal_decoder
from . import aether_10000d

# Legacy (deprecated, try/except for backward compatibility)
try:
    from . import recipe_memory
except ImportError:
    pass
try:
    from . import recipe_mixer
except ImportError:
    pass

# ★ skill_manager互換性レイヤー - _legacyからre-export ★
try:
    from ._legacy import skill_manager
    from ._legacy.skill_manager import SkillManager, get_brain, get_synthesizer
except ImportError:
    skill_manager = None
    SkillManager = None
    get_brain = None
    get_synthesizer = None

from . import shapes
from . import utils
from . import studio
from . import lighting
from . import textures
from . import uv
from . import environment
from . import animation
from . import physics
from . import rigging
from . import lod

from . import particles
from . import architecture
from . import vehicles
from . import character
from . import dungeon
from . import export

from . import facial
from . import planet
from . import ik
from . import city
from . import interactive
from . import cad

from . import ai
from . import cinematic
from . import fluid
from . import audio_viz
from . import level_editor
from . import engine

from . import autonomous
from . import templates
from . import commercial

from . import ai_texture

# [DBG] 品質検査・スマート合成（v5.2.0 追加）
from . import inspector
from . import smart_synthesize
from . import pipeline
from . import translator

__all__ = [
    'common', 'palette', 'organic', 'boolean', 'forcefield', 'lsystem', 'lod_gen', 'softbody', 'remesh', 'autorig', 'hair', 'generators', 'physics_unified', 'baking', 'batch_export', 'thumbnail',
    'shapes', 'utils', 'studio', 'lighting', 'textures', 'uv',
    'environment', 'animation', 'physics', 'rigging', 'lod',
    'particles', 'architecture', 'vehicles', 'character', 'dungeon', 'export',
    'facial', 'planet', 'ik', 'city', 'interactive', 'cad',
    'ai', 'cinematic', 'fluid', 'audio_viz', 'level_editor', 'engine',
    'autonomous', 'templates', 'commercial',
    'ai_texture', 'llm_brain', 'safety',
    # v6.0: 新統合システム
    'unified_memory', 'aether_core', 'evaluator_v2',
    'inspector', 'smart_synthesize', 'pipeline', 'translator',
    # v3.0: 10000D統合システム
    'llm_prompt_10000d_v2', 'universal_decoder', 'aether_10000d',
]

__version__ = '6.1.0'

