import numpy as np
mgc_dim=59
frame_period=5
f0_floor=71.0
f0_ceil=700
use_harvest=True  # If False, use dio and stonemask
windows=[(0, 0, np.array([1.0])),(1, 1, np.array([-0.5, 0.0, 0.5])),(1, 1, np.array([1.0, -2.0, 1.0]))]
f0_interpolation_kind="quadratic"
mod_spec_smoothing=True
mod_spec_smoothing_cutoff=50
recompute_delta_features=False
stream_sizes=[180, 3, 1, 3]
