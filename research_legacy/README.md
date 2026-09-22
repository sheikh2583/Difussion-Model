# Legacy research implementation archive

This directory preserves Git-exact source states associated with the project's
existing research checkpoints. It does **not** replace or duplicate the large
checkpoint tensors under `results/`; `checkpoint_provenance.json` maps those
artifacts to the source evidence retained here.

## Backbone finding

`models/backbone.py` was introduced in commit
`5520fbe8b26ca6f71714503efaba03c9025b362a` and every checkpoint-backed commit
audited here resolves to the same Git blob:

```text
d1b02c84feba4f1f4360a7adf54c3a98729824c8
```

There are therefore multiple historical *algorithm/configuration* states, but
only one genuine legacy backbone implementation. Each ZIP contains that exact
backbone together with the corresponding committed `algorithms/` and `config/`
trees so old checkpoint behavior can be studied without mixing it into the
current latent implementation.

For executable reruns, that architecture is also frozen as
`models/legacy_cifar_backbone.py` and selected with the
`legacy_cifar_unet` backbone name. The presets under `config/cifar_legacy/`
write to isolated result directories; the ZIP files here remain the immutable
source-history evidence.

## Reproducibility levels

- `8618531_clean_mf_probe.zip` and `2ea2a5e_clean_research_runs.zip` are exact
  clean Git snapshots recorded by run manifests.
- `c1f065a_base_for_dirty_runs.zip` and `e04529a_base_for_dirty_run.zip` are
  exact base commits, but the associated run manifests explicitly record dirty
  fingerprints. The uncommitted patch was not embedded in the checkpoints, so
  these bundles must not be presented as exact reconstruction of those dirty
  runs.
- `results/exports/thesis_context.zip` is retained in place as the surviving
  later source capture. It is evidence, not proof that it equals either dirty
  runtime state.

All archives were produced with `git archive`; no historical branch was
checked out and the current worktree was not overwritten.
