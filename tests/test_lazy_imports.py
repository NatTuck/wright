from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


class TestImport:
    def test_import_resin_does_not_import_torch(self) -> None:
        code = (
            "import sys; sys.path[:0] = ['models/torch']; "
            "import resin; import resin.backend; "
            "from resin import Parallelism; Parallelism(world_size=4, attn_dp=4); "
            "import resin_models_torch; "
            "print(int('torch' in sys.modules))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], check=True, capture_output=True, text=True
        )
        assert result.stdout.strip() == "0"

    def test_torch_comes_from_runners_and_the_cpu_index(self) -> None:
        workspace = tomllib.loads(Path("pyproject.toml").read_text())
        project = workspace["project"]
        assert project["dependencies"] == [
            "blake3>=1.0,<2",
            "numpy>=1.26,<3",
        ]
        assert "optional-dependencies" not in project

        dev = workspace["dependency-groups"]["dev"]
        assert "resin-models-torch" in dev
        assert any(item.startswith("tokenizers") for item in dev)
        assert not any(item.startswith("hf-xet") for item in dev)
        assert workspace["tool"]["uv"]["sources"]["torch"] == {"index": "pytorch-cpu"}

        torch_models = tomllib.loads(Path("models/torch/pyproject.toml").read_text())[
            "project"
        ]
        assert "resin==0.1.0" in torch_models["dependencies"]
        assert any(item.startswith("torch>=") for item in torch_models["dependencies"])
        assert any(
            item.startswith("huggingface-hub>=")
            for item in torch_models["dependencies"]
        )
        assert any(
            item.startswith("safetensors>=") for item in torch_models["dependencies"]
        )
