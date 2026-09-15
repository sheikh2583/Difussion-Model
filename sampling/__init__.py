"""
sampling — generic sampling engine.

This package contains the Sampler class, which drives image generation
for evaluation and visualisation purposes.  Like the Trainer, the
Sampler is algorithm-agnostic: it delegates the actual image generation
to BaseAlgorithm.sample(n_samples, nfe, device) and only handles the
surrounding concerns:

  - iterating over each NFE value in the configured nfe_values list;
  - seeding the RNG identically for every algorithm so generated images
    are reproducibly comparable;
  - wall-clock timing with CUDA synchronization;
  - peak GPU memory measurement;
  - writing a ResultRecord row (sampling record type) to the JSONL log;
  - optionally saving a grid image to disk for visual inspection.
"""

from sampling.sampler import Sampler

__all__ = ["Sampler"]
