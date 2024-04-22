from pathlib import Path

import wandb


def upload_code_to_wandb(code_dir: Path | str):
    if isinstance(code_dir, str):
        code_dir = Path(code_dir)

    code = wandb.Artifact("project-source", type="code")

    for path in code_dir.resolve().rglob("*.py"):
        code.add_file(str(path), name=str(path.relative_to(code_dir)))

    wandb.log_artifact(code)
