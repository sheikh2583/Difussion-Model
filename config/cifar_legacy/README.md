# CIFAR-10 legacy-backbone presets

These presets rerun the completed CIFAR-10 comparison with the frozen
`legacy_cifar_unet` implementation in `models/legacy_cifar_backbone.py`.
They intentionally write to separate `*_legacy_backbone_cifar10` result
directories and use a separate Reflow-pair artifact.

The ordinary presets under `config/` keep their original `simple_unet` name so
existing checkpoint provenance and evaluation commands remain compatible.

