#!/usr/bin/env bash
#
# Compares two versions of nnet and appends a run to info/PERFORMANCE.md.
#
# Both versions are built with the same compiler and flags, and run the same
# benchmark program (benchmarks/bench_suite.cpp from your current files), so
# the only difference between them is the library code. The runs alternate
# before/after, so if the computer speeds up or slows down partway through,
# both sides are affected equally.
#
# Usage:
#   scripts/benchmarks.sh <before> [after] [options]
#
#   before   commit, branch or tag to compare against: HEAD, HEAD~1, fef6a15, main
#   after    commit to compare it with. Leave it out to use your files as they
#            are right now, including changes you haven't committed yet.
#
# Options:
#   -t "title"     title of the entry           (default: "Untitled optimization")
#   -m "label"     machine label                (default: "Machine 1: <your OS>")
#   -r rounds      runs of each side            (default: 3)
#   -n repeats     repeats inside each run      (default: 5)
#
# Examples:
#   # you just made a change and haven't committed it: compare against HEAD
#   scripts/benchmarks.sh HEAD -t "std::fill for gradient resets"
#
#   # redo an old comparison between two commits
#   scripts/benchmarks.sh fef6a15 b7eff87 -t "i-k-j matrix multiplication"
#
# The same title groups runs from different machines under one test section.
# Raw samples and source evidence stay in info/benchmark-runs/ (also on failure).
# Requires Python 3 for validation and updating the Markdown log.
# Set CXX to choose the compiler (default: clang++ if installed, else c++).

set -euo pipefail

usage() {
    sed -n '3,33p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

# ---------------------------------------------------------------- arguments

title="Untitled optimization"
machine_label=""
rounds=3
repeats=5
before_ref=""
after_ref=""

while [ $# -gt 0 ]; do
    case "$1" in
        -t|-m|-r|-n)
            if [ $# -lt 2 ]; then echo "Missing value for $1" >&2; exit 1; fi ;;
    esac
    case "$1" in
        -t) title="$2"; shift 2 ;;
        -m) machine_label="$2"; shift 2 ;;
        -r) rounds="$2"; shift 2 ;;
        -n) repeats="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        -*) echo "Unknown option: $1" >&2; usage 1 ;;
        *)
            if [ -z "$before_ref" ]; then before_ref="$1"
            elif [ -z "$after_ref" ]; then after_ref="$1"
            else echo "Too many arguments: $1" >&2; usage 1
            fi
            shift ;;
    esac
done

if [ -z "$before_ref" ]; then
    echo "Say which commit to compare against, e.g.: scripts/benchmarks.sh HEAD" >&2
    echo >&2
    usage 1
fi

for count in "$rounds" "$repeats"; do
    if ! [[ "$count" =~ ^[1-9][0-9]*$ ]] || [ "${#count}" -gt 6 ]; then
        echo "Rounds and repeats must be positive integers (at most six digits)." >&2
        exit 1
    fi
done
if [ -z "$title" ] || [[ "$title" == *$'\n'* || "$title" == *$'\r'* ]]; then
    echo "The test title must be one nonempty line." >&2
    exit 1
fi
command -v python3 > /dev/null || { echo "Python 3 is required." >&2; exit 1; }

repo=$(git rev-parse --show-toplevel)
cd "$repo"
suite="$repo/benchmarks/bench_suite.cpp"
if [ ! -f "$suite" ]; then
    echo "Can't find $suite" >&2
    exit 1
fi

for ref in "$before_ref" ${after_ref:+"$after_ref"}; do
    if ! git rev-parse --verify --quiet "$ref^{commit}" > /dev/null; then
        echo "'$ref' isn't a commit, branch or tag in this repository." >&2
        exit 1
    fi
done

if [ -z "${CXX:-}" ]; then
    if command -v clang++ > /dev/null 2>&1; then CXX=clang++; else CXX=c++; fi
fi
flags="-std=c++17 -O2 -DNDEBUG -Wall -Wextra -Wpedantic"

mkdir -p "$repo/info/benchmark-runs"
results=$(mktemp -d "$repo/info/benchmark-runs/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
run_id=$(basename "$results")
cp "$suite" "$results/suite.cpp"
suite="$results/suite.cpp"
git diff --binary HEAD -- src include > "$results/source.patch"
git ls-files --others --exclude-standard -- src include > "$results/untracked-sources.txt"
if [ -s "$results/untracked-sources.txt" ]; then
    echo "Untracked library sources cannot be reproduced by source.patch; record them before benchmarking." >&2
    exit 1
fi
{
    printf 'Before commit: %s\n' "$(git rev-parse "$before_ref^{commit}")"
    printf 'After commit: %s\n' "$(git rev-parse "${after_ref:-HEAD}^{commit}")"
    printf 'After uses live files: %s\n' "$([ -z "$after_ref" ] && echo yes || echo no)"
    printf 'Suite Git blob hash: %s\n' "$(git hash-object "$suite")"
    printf 'Compiler: %s\nFlags: %s\nRounds: %s\nRepeats: %s\n' "$CXX" "$flags" "$rounds" "$repeats"
    printf 'Source patch applies to HEAD only when after uses live files.\n'
} > "$results/metadata.txt"
echo "Raw evidence: $results" >&2
echo "incomplete" > "$results/status.txt"

work=$(mktemp -d "${TMPDIR:-/tmp}/nnet-bench.XXXXXX")
cleanup() {
    rm -rf "$work"
}
trap cleanup EXIT
# Ctrl+C: clean up too, instead of leaving a copy of the old commit behind.
trap 'exit 130' INT TERM

# ---------------------------------------------------------------- versions

# Describes a side for the entry: `hash` subject, or the working tree.
describe() {
    if [ -z "$1" ]; then
        local head
        head=$(git rev-parse --short HEAD)
        if [ -n "$(git status --porcelain --untracked-files=no -- src include)" ]; then
            echo "your current files (\`$head\` + uncommitted changes)"
        else
            echo "\`$head\` $(git log -1 --format=%s HEAD)"
        fi
    else
        echo "\`$(git rev-parse --short "$1")\` $(git log -1 --format=%s "$1")"
    fi
}

# Builds bench_suite against one version of the library.
# $1 = before|after, $2 = commit (empty for the current files).
build_side() {
    local side="$1" ref="$2" dir
    if [ -z "$ref" ]; then
        dir="$repo"
    else
        dir="$work/$side"
        mkdir -p "$dir"
        git archive "$ref" | tar -x -C "$dir"
    fi

    local sources=()
    local file
    for file in "$dir"/src/core/tensor.cpp "$dir"/src/core/tensor_ops.cpp \
                "$dir"/src/autograd/*.cpp "$dir"/src/train/*.cpp "$dir"/src/optim/*.cpp; do
        if [ -f "$file" ]; then sources+=("$file"); fi
    done

    echo "Building $side: $(describe "$ref")" >&2
    # shellcheck disable=SC2086  # $flags is meant to split into several flags
    if ! "$CXX" $flags -I"$dir/include" "$suite" "${sources[@]}" -o "$work/bench_$side" \
            > "$results/build_$side.log" 2>&1; then
        echo >&2
        echo "Building the $side version failed:" >&2
        tail -n 20 "$results/build_$side.log" >&2
        echo >&2
        echo "If this is an old commit, bench_suite.cpp may use something it doesn't have yet." >&2
        exit 1
    fi
}

if [ -z "$after_ref" ] && [ -z "$(git status --porcelain --untracked-files=no -- src include)" ] &&
   [ "$(git rev-parse "$before_ref^{commit}")" = "$(git rev-parse HEAD)" ]; then
    echo "Note: you have no uncommitted changes in src/ or include/, so before and after" >&2
    echo "are the same code. Expect \"same\" or \"within noise\" everywhere." >&2
    echo "To compare your last commit with the one before it: scripts/benchmarks.sh HEAD~1 HEAD" >&2
    echo >&2
fi

build_side before "$before_ref"
build_side after "$after_ref"

# ---------------------------------------------------------------- running

: > "$results/before.txt"
: > "$results/after.txt"
round=1
while [ "$round" -le "$rounds" ]; do
    # Swap the order every round, so neither side always goes first.
    if [ $((round % 2)) -eq 1 ]; then order="before after"; else order="after before"; fi
    for side in $order; do
        echo "Round $round/$rounds: running $side..." >&2
        printf '%s %s\n' "$round" "$side" >> "$results/order.txt"
        "$work/bench_$side" "$repeats" >> "$results/$side.txt" 2>> "$results/$side-errors.log"
    done
    round=$((round + 1))
done

# "median min max" of one measurement across all rounds of one side.
stats() {
    awk -v key="$2" '$1 == key { print $2 }' "$results/$1.txt" | sort -g | awk '
        BEGIN { OFMT = "%.12g"; CONVFMT = "%.12g" }
        { values[NR] = $1 }
        END {
            if (NR == 0) { print "nan nan nan"; exit }
            if (NR % 2 == 1) median = values[(NR + 1) / 2]
            else median = (values[NR / 2] + values[NR / 2 + 1]) / 2
            print median, values[1], values[NR]
        }'
}

# A number at a sensible precision for a table. $2 = count shows whole
# numbers (allocation counts) without decimals.
format() {
    awk -v v="$1" -v kind="${2:-}" 'BEGIN {
        a = v < 0 ? -v : v
        whole = int(a + 0.5)
        if (kind == "count" && a >= 1 && a - whole < 0.05 && whole - a < 0.05) printf "%.0f", v
        else if (a >= 1000) printf "%.0f", v
        else if (a >= 100) printf "%.1f", v
        else if (a >= 1) printf "%.2f", v
        else printf "%.3g", v
    }'
}

best_label=""
best_ratio=1
noise_any=0

# One table row. $1 = key, $2 = label, $3 = time | rate | count
row() {
    local key="$1" label="$2" kind="$3"
    local b bmin bmax a amin amax change
    read -r b bmin bmax <<< "$(stats before "$key")"
    read -r a amin amax <<< "$(stats after "$key")"

    change=$(awk -v b="$b" -v a="$a" -v bmin="$bmin" -v bmax="$bmax" \
                 -v amin="$amin" -v amax="$amax" -v kind="$kind" 'BEGIN {
        if (a == b) { print "same"; exit }
        if (b == 0 || a == 0) { print "n/a"; exit }
        if (kind == "count") {
            printf "%+.0f%%", (a - b) / b * 100
            exit
        }
        ratio = (kind == "time") ? b / a : a / b
        if (ratio >= 1) text = sprintf("%.2f× %s", ratio, kind == "time" ? "faster" : "higher")
        else text = sprintf("%.2f× %s", 1 / ratio, kind == "time" ? "slower" : "lower")
        # If the before and after ranges overlap, the difference could be noise.
        if (amin <= bmax && bmin <= amax) text = text " (within noise)"
        print text
    }')

    printf '| %s | %s | %s | %s |\n' "$label" "$(format "$b" "$kind")" "$(format "$a" "$kind")" "$change"

    case "$change" in *"within noise"*) noise_any=1 ;; esac
    if [ "$kind" = "time" ]; then
        case "$change" in
            *faster|*slower)
                if awk -v b="$b" -v a="$a" -v best="$best_ratio" \
                       'BEGIN { r = b / a; if (r < 1) r = 1 / r; best0 = best < 1 ? 1 / best : best; exit !(r > best0) }'; then
                    best_ratio=$(awk -v b="$b" -v a="$a" 'BEGIN { print b / a }')
                    best_label="$label"
                fi ;;
        esac
    fi
}

# ---------------------------------------------------------------- machine

os_name=$(uname -s)
case "$os_name" in
    Darwin)
        os_label="macOS"
        os_version="macOS $(sw_vers -productVersion)"
        cpu=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || sysctl -n hw.model)
        cores="$(sysctl -n hw.physicalcpu) physical, $(sysctl -n hw.logicalcpu) logical"
        ram="$(awk -v b="$(sysctl -n hw.memsize)" 'BEGIN { printf "%.0f", b / 1073741824 }') GB"
        ;;
    Linux)
        os_label="Linux"
        # shellcheck disable=SC1091  # the file exists on any modern Linux
        os_version=$( (. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") || uname -sr)
        cpu=$(awk -F: '/model name/ { sub(/^ +/, "", $2); print $2; exit }' /proc/cpuinfo)
        cores="$(getconf _NPROCESSORS_ONLN) logical"
        ram="$(awk '/MemTotal/ { printf "%.0f", $2 / 1048576 }' /proc/meminfo) GB"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        os_label="Windows"
        os_version="Windows ($os_name)"
        cpu=$(awk -F: '/model name/ { sub(/^ +/, "", $2); print $2; exit }' /proc/cpuinfo 2>/dev/null || echo "unknown")
        cores="${NUMBER_OF_PROCESSORS:-?} logical"
        ram="(check in Task Manager)"
        ;;
    *)
        os_label="$os_name"; os_version="$os_name"; cpu="unknown"; cores="?"; ram="?"
        ;;
esac
: "${cpu:=unknown}"
compiler=$("$CXX" --version 2>/dev/null | head -n 1)
if [ -z "$machine_label" ]; then machine_label="Machine 1: $os_label"; fi
machine_label=$(printf '%s' "$machine_label" | sed 's/\&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')
printf 'CPU: %s\nCores: %s\nRAM: %s\nOS: %s\nKernel: %s\nCompiler version: %s\n' \
    "$cpu" "$cores" "$ram" "$os_version" "$(uname -sr)" "$compiler" >> "$results/metadata.txt"

# ---------------------------------------------------------------- checks

python3 "$repo/scripts/benchmark_report.py" validate "$results" "$rounds" "$repeats"

# ---------------------------------------------------------------- entry

table=$(
    echo "  | Metric | Before | After | Change |"
    echo "  | :--- | ---: | ---: | :--- |"
    row matmul100_ms "Matmul 100×100, 1000 multiplies (ms)" time
    row matmul100_gflops "Matmul 100×100 throughput (GFLOP/s)" rate
    row matmul_layer_ms "Matmul 64×784 · 784×128, one multiply (ms)" time
    row matmul_layer_gflops "Matmul 64×784 · 784×128 throughput (GFLOP/s)" rate
    row matmul100_allocs "Matmul 100×100, allocations per multiply" count
    row matmul100_kb "Matmul 100×100, memory allocated per multiply (KiB)" count
    row mlp_small_epoch_ms "Small MLP (tensor_mlp network), one epoch (ms)" time
    row mlp_small_allocs_per_step "Small MLP, allocations per training step" count
    echo "BEST|$best_label|$best_ratio|$noise_any"
)
summary_line=$(printf '%s\n' "$table" | tail -n 1)
table=$(printf '%s\n' "$table" | sed '$d' | sed 's/^| /  | /')
best_label=$(echo "$summary_line" | cut -d'|' -f2)
best_ratio=$(echo "$summary_line" | cut -d'|' -f3)
noise_any=$(echo "$summary_line" | cut -d'|' -f4)

if [ -n "$best_label" ]; then
    best_label="${best_label% (*}"   # drop the "(ms)"
    headline=$(awk -v r="$best_ratio" -v l="$best_label" 'BEGIN {
        if (r >= 1) printf "**%.2f× faster**: %s", r, l
        else printf "**%.2f× slower**: %s", 1 / r, l
    }')
else
    headline="**No non-overlapping timing improvement or regression observed.**"
fi

noise_note=""
if [ "$noise_any" = "1" ]; then
    noise_note="
  \"Within noise\" means the before and after runs overlapped, so that difference may not be real."
fi

# Keep the report fragment with its evidence, then insert beneath its test title.
entry_file="$results/entry.md"
cat > "$entry_file" <<EOF
<details>
<summary><b>$machine_label — $(date -u '+%Y-%m-%d %H:%M:%S UTC')</b></summary>

### Setup

- CPU: $cpu
- Cores: $cores
- RAM: $ram
- OS: $os_version
- Kernel: $(uname -sr)
- Compiler: $compiler, \`$flags\`
- Before: $(describe "$before_ref")
- After: $(describe "$after_ref")
- Timing numbers are medians of $rounds rounds, each the median of $repeats repeats, with one untimed warmup per case. Run order alternates.
- Both matrix shapes pass the plain dot-product reference (absolute tolerance 1e-9). First-epoch loss and all 87 trained MLP parameters match across versions (absolute/relative tolerance 1e-9).
- No CPU affinity or power controls applied; results describe this machine and workload only.
- Evidence: [before samples](benchmark-runs/$run_id/before.txt), [after samples](benchmark-runs/$run_id/after.txt), [metadata](benchmark-runs/$run_id/metadata.txt), [suite source](benchmark-runs/$run_id/suite.cpp), [uncommitted library patch](benchmark-runs/$run_id/source.patch).

### Results

  $headline

$table
$noise_note

</details>
EOF

python3 "$repo/scripts/benchmark_report.py" append "$repo/info/PERFORMANCE.md" "$title" "$entry_file"
echo "passed; appended to info/PERFORMANCE.md" > "$results/status.txt"
echo >&2
echo "Appended under '$title' in info/PERFORMANCE.md" >&2
echo >&2
cat "$entry_file"
