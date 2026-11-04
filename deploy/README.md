# Serving and benchmarks

`models.py` is the shared model registry. It holds the pinned checkpoints, the card count and
the serving limits of each preset, and builds the runner and the chat recipe. `serving.py` is the
server that the `wright` command and `scripts/serve_local.py` start.

The presets run on your own host. Use the `wright` command to serve a model, and `scripts/bench.py`
to measure one. The [README](../README.md#how-to-run-it) gives the commands and flags, and
[the install list](../docs/install.md) gives the packages.

To run the server as a service, use [`systemd/wright.service`](systemd/wright.service). Set `User`, the
paths, and the `ExecStart` arguments, then copy the file to `/etc/systemd/system/`. `gaudi_cli.py` holds
the flags that the server and the benchmark share.
