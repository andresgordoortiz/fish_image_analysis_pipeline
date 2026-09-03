"""Modular SPIM preprocessing primitives.

Each script in this package is a single, focused preprocessing step that
reads a 3D volume from a TIFF (with ImageJ metadata), applies one
correction, and writes the result back to a TIFF preserving voxel sizes.

This is the lean replacement for the previous monolithic
``spim_pipeline_fixed.py`` (deconvolution + CLAHE + shading + Z-correction +
WBNS + isotropic reslice + post-processing all in one file). The new design
makes every correction:

* independently testable (one CLI per step),
* independently skippable (just don't run the process in the workflow),
* and self-documenting (the docstring of each module is the spec).

The math is ported from the AIAF-32 modular scripts
(``planar_intensity_correction.py`` and ``depth_intensity_correction.py``)
and adapted to read/write plain TIFFs with ImageJ metadata instead of
OME-TIFF, so the rest of the SPIM pipeline (Cellpose, ultrack, viewer)
does not need to change.
"""
