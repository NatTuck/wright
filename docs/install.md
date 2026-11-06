# Install Wright

This page gives the exact items that Wright needs on a Gaudi 2 host, and the items that it does not
need. The list comes from a trace of a cold start by a new user account on Debian 13, Synapse 1.24.1.

## Required

| Item | Source | Use |
|---|---|---|
| Driver and firmware | `habanalabs-dkms`, `habanalabs-firmware` | The cards |
| Synapse, HCCL, stock kernels, headers, engine firmware | `habanalabs-graph` | `libSynapse.so`, `libhcl.so`, `libtpc_kernels.so`, `synapse_api.h`, `/opt/habanalabs/engines_fw` |
| Thunk and RDMA libraries | `habanalabs-thunk`, `habanalabs-rdma-core` | `habanalabs-graph` pulls these in |
| TPC-C compiler | `habanalabs-tools` | `tpc-clang`, which builds the kernels. The tests used version 2.7.33 |
| C and C++ compilers | `gcc`, `g++` | The native libraries |
| Rust | [`rustup`](https://rustup.rs) | Builds Resin. `rustup` installs the version from `rust-toolchain.toml`. A distribution `rustc` can be too old |
| Python 3.11 or later, and `uv` | Your distribution, [`uv`](https://docs.astral.sh/uv/) | The Python environment |

Wright keeps Resin in the repository. `uv` builds it from source, so you do not install Resin
separately.

## Not required

- **Intel Gaudi PyTorch.** Wright does not import `habana_frameworks`.
- **A GPU build of PyTorch.** Wright installs CPU-only `torch`, which the ranks use only for their Gloo
  control plane. The model does not run on `torch`.
- **`habanalabs-qual`, `habanalabs-perf-test`, `habanalabs-aeon`, `habanalabs-hypervisor-utils` and
  OpenMPI.** The trace shows no use of these packages.
- **`habanatools`.** The server does not use these diagnostic tools.

`habanalabs-firmware-tools` gives `hl-smi`. The server does not use it, but `hl-smi` shows the module IDs
for `--device`, so we recommend it.

## Host settings

| Setting | Value | Why |
|---|---|---|
| Card access | `/dev/accel/*` readable and writable by your user | The `habanalabs` packages install a rule for this |
| `HABANA_LOGS` | A directory that your user can write | Synapse stops the process if it cannot write its logs. When the current directory has files of another user, Wright uses a directory in its cache and shows a message |
| `HABANA_SCAL_BIN_PATH` | `/opt/habanalabs/engines_fw` | A login shell gets this from `/etc/profile.d`. A systemd service does not, so `deploy/systemd/wright.service` sets it |

## Disk

| Item | Size |
|---|---|
| Virtual environment | Approximately 900 MB |
| Rust toolchain and packages | Approximately 450 MB |
| `uv` package cache | Approximately 420 MB |
| Qwen3.8-27B weights | 29 GB |
| GLM-5.3-Flash weights | 306 GiB |
| Mistral-Large-4 weights | 564 GiB |
| DeepSeek-V4.1-Flash weights | 475 GiB |

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none --profile minimal
git clone https://github.com/abotsis/wright.git
cd wright
uv sync --locked --no-default-groups --group gaudi
```

To run the tests, use the `gaudi-dev` group in place of the `gaudi` group.

## Run as a service

`deploy/systemd/wright.service` starts the server from `.venv/bin/python`. Copy the file, then set
`User`, the paths and the arguments. When the service stops, the server stops new connections and
gives requests in flight the time from `--drain-timeout`.

## Other distributions

Wright finds the Synapse version with `dpkg-query`. If this command is not available, Wright uses a
hash of `libSynapse.so` in place of the version, and it cannot warn you about a version that is not
tested. Only Debian 13 has been tested.
