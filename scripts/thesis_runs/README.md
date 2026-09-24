# Three-terminal thesis training

Each launcher defaults to `--mode fresh`. Existing results are preserved and
the next `checkpoints/run_N/` series is selected automatically. Jobs within one
suite remain serialized because of teacher and Reflow dependencies.

This PC has one RTX 3090. Start the three scripts in numbered order. Their
shared queue lock permits only one stage to run; later terminals wait and then
report their own start, finish, elapsed seconds, and exit status:

```bash
./scripts/linux/01_train_cifar10.sh
./scripts/linux/02_train_celeba_pixel.sh
./scripts/linux/03_train_celeba_latent.sh
```

The numbered wrappers queue on `results/.locks/thesis_stages.lock`; the normal
GPU workflow still uses `results/.lock`. Start them in numerical order so the
waiting order represents the thesis progression. Pressing Ctrl-C cancels a
waiting terminal without interrupting the stage that currently owns the GPU.
Completed stage timings are appended to
`training_logs/thesis_stages/stage_timings.jsonl`; dry-runs do not write it.

Preview without writing artifacts or starting a model:

```bash
./scripts/linux/01_train_cifar10.sh --dry-run
./scripts/linux/02_train_celeba_pixel.sh --dry-run
./scripts/linux/03_train_celeba_latent.sh --dry-run
```

Use `--mode continue` to resume instead of allocating a new run series. For a
historical CIFAR rerun, append `--cifar-backbone legacy` to the first command.

Every job records GPU identity, Git commit, dirty-diff digest, frozen source/config identity,
checkpoint series, timestamps, exit status, transcript hash, and command. The
launchers never stage, commit, or push files.

Run numbering is independent per experiment. Fresh mode preserves all earlier
series and selects `max(existing run_N) + 1`; it does not force unrelated
algorithms to share an artificial number. The selected `run_N` appears in the
checkpoint directory, transcript filename, transcript header, and `.meta.json`
sidecar.
