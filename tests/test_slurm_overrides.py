import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def dedupe(args: list[str]) -> list[str]:
    script = (
        f'source "{ROOT}/slurm/common/overrides.sh"; '
        'HYDRA_ARGS=("$@"); rlx_dedupe_overrides; printf "%s\\n" "${HYDRA_ARGS[@]}"'
    )
    out = subprocess.run(["bash", "-c", script, "_", *args], capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def test_plain_override_replaces_in_place():
    assert dedupe(["a.b=1", "c=2", "a.b=3"]) == ["a.b=3", "c=2"]


def test_plus_then_plus_becomes_force_add():
    assert dedupe(["+sd.n=4", "x=1", "+sd.n=2"]) == ["++sd.n=2", "x=1"]


def test_plus_then_plain_keeps_working():
    assert dedupe(["+sd.n=4", "sd.n=2"]) == ["++sd.n=2"]


def test_values_with_equals_and_brackets_survive():
    assert dedupe(["trainer.logger=[\"console\",\"wandb\"]", "k=a=b"]) == ['trainer.logger=["console","wandb"]', "k=a=b"]


def test_deletion_overrides_pass_through():
    assert dedupe(["~a.b", "c=1"]) == ["~a.b", "c=1"]
