"""Validate benchmark evidence and append a machine run under its test title."""

import math
from pathlib import Path
import sys
import tempfile


def read_measurements(path):
    measurements = {}
    for line in path.read_text().splitlines():
        key, text = line.split()
        value = float(text)
        if not math.isfinite(value):
            raise ValueError(f"{path.name}: non-finite {key}")
        measurements.setdefault(key, []).append(value)
    return measurements


def validate(directory, rounds, repeats):
    before = read_measurements(directory / "before.txt")
    after = read_measurements(directory / "after.txt")
    metrics = {
        "matmul100_max_error", "matmul_layer_max_error", "matmul100_ms",
        "matmul100_gflops", "matmul_layer_ms", "matmul_layer_gflops",
        "matmul100_allocs", "matmul100_kb", "mlp_small_epoch_ms",
        "mlp_small_step_ms", "mlp_small_allocs_per_step", "mlp_small_loss",
    }
    samples = {"sample_matmul100_ms", "sample_matmul_layer_20_ms", "sample_mlp_small_epoch_ms"}
    parameters = {f"check_mlp_parameter_{i}" for i in range(87)}
    expected = metrics | samples | parameters
    for side, values in (("before", before), ("after", after)):
        if values.keys() != expected:
            raise ValueError(f"{side}: missing or unexpected measurements")
        for key, numbers in values.items():
            count = rounds * repeats if key in samples else rounds
            if len(numbers) != count:
                raise ValueError(f"{side}: incomplete {key} ({len(numbers)} of {count})")
            if key.endswith("_max_error") and any(not 0 <= x <= 1e-9 for x in numbers):
                raise ValueError(f"{side}: matrix correctness check failed")
            if (key.endswith("_ms") or key.endswith("_gflops")) and any(x <= 0 for x in numbers):
                raise ValueError(f"{side}: nonpositive timing or throughput")
    for key in parameters | {"mlp_small_loss"}:
        for b, a in zip(before[key], after[key]):
            if not math.isclose(b, a, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"{key} changed: {b} before, {a} after")


def append_run(report, title, entry):
    if not title.strip() or "\n" in title or "\r" in title:
        raise ValueError("The test title must be one nonempty line")
    report.parent.mkdir(parents=True, exist_ok=True)
    # Refuse overlapping writers instead of silently losing someone else's run.
    lock = report.parent / ".performance.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError("Another report update holds info/.performance.lock; raw results are retained")
    temporary = None
    try:
        text = report.read_text() if report.exists() else "# Performance\n\n"
        heading = f"## {title}\n"
        lines = text.splitlines(keepends=True)
        matches = [i for i, line in enumerate(lines) if line == heading]
        if len(matches) > 1:
            raise ValueError("Duplicate test headings; combine them before appending")
        run = entry.read_text().strip() + "\n\n"
        if matches:
            end = next((i for i in range(matches[0] + 1, len(lines))
                        if lines[i].startswith("## ")), len(lines))
            updated = "".join(lines[:end]).rstrip() + "\n\n" + run + "".join(lines[end:])
        else:
            updated = text.rstrip() + "\n\n" + heading + "\n" + run
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=report.parent,
                                         prefix=".performance-", delete=False) as output:
            temporary = Path(output.name)
            output.write(updated)
        temporary.chmod(report.stat().st_mode & 0o777 if report.exists() else 0o644)
        temporary.replace(report)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock.rmdir()


if __name__ == "__main__":
    try:
        if len(sys.argv) == 5 and sys.argv[1] == "validate":
            validate(Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
        elif len(sys.argv) == 5 and sys.argv[1] == "append":
            append_run(Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
        else:
            raise ValueError("Use validate RUN_DIR ROUNDS REPEATS or append REPORT TITLE ENTRY")
    except (OSError, ValueError) as error:
        sys.exit(f"Benchmark report: {error}")
