"""Stereo HOI object pose estimation pipeline.

Submodules:
    depth      —  stereo depth estimation (Fast-FoundationStereo)
    tracking   —  single-view 6DoF tracking (FoundationPose)
    fusion     —  multi-view pose fusion + temporal smoothing
    hand       —  hand mesh inference (WiLoR) + metric alignment
    vis        —  2D/3D visualisation, web export, comparison videos
"""

__version__ = "0.1.0"
