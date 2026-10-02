from .base_sim import BaseSim

# Import simulations with error handling for optional dependencies
try:
    from .libero_sim import MultiTaskSim
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import MultiTaskSim: {e}")
    MultiTaskSim = None

try:
    from .bidex_sim import BidexSim
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import BidexSim: {e}")
    BidexSim = None

try:
    from .robocasa_sim import RoboCasaSim
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import RoboCasaSim: {e}")
    RoboCasaSim = None

RobocasaSim = RoboCasaSim

try:
    from .robocasa_pc_sim import RobocasaPCSim
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import RobocasaPCSim: {e}")
    RobocasaPCSim = None

try:
    from .robocasa_pc_img_sim import RobocasaPCImgSim
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import RobocasaPCImgSim: {e}")
    RobocasaPCImgSim = None

try:
    from .noop_sim import NoOpSim
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import NoOpSim: {e}")
    NoOpSim = None

NoopSim = NoOpSim

__all__ = [
    'BaseSim',
    'MultiTaskSim',
    'BidexSim',
    'RoboCasaSim',
    'RobocasaSim',
    'RobocasaPCSim',
    'RobocasaPCImgSim',
    'NoOpSim',
    'NoopSim',
]
