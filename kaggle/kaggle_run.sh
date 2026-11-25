#!/usr/bin/env bash
# Run a notebook on Kaggle from the Mac and fetch its outputs.
#
#   kaggle/kaggle_run.sh push   [notebook]   # default kaggle/soup_pilot.ipynb, runs on T4
#   kaggle/kaggle_run.sh status [notebook]
#   kaggle/kaggle_run.sh fetch  [notebook]   # outputs -> kaggle/out/<slug>/
#
# Needs a Kaggle API token installed by the user (never pasted in chat):
#   ~/.kaggle/kaggle.json (chmod 600)  or  ~/.kaggle/access_token
set -euo pipefail
cd "$(dirname "$0")/.."
KAGGLE="${KAGGLE:-$HOME/.local/bin/kaggle}"
cmd="${1:?push|status|fetch}"
nb="${2:-kaggle/soup_pilot.ipynb}"
slug="$(basename "$nb" .ipynb | tr '_' '-')"

user="${KAGGLE_USERNAME:-}"
if [[ -z "$user" && -f "$HOME/.kaggle/kaggle.json" ]]; then
  user="$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["username"])')"
fi
if [[ -z "$user" ]]; then
  user="$("$KAGGLE" config view 2>/dev/null | awk -F': ' '/username/{print $2}' | tr -d ' ' || true)"
fi
[[ -n "$user" ]] || { echo "no Kaggle username: install the API token first" >&2; exit 2; }
ref="$user/$slug"

case "$cmd" in
  push)
    dir="kaggle/push/$slug"
    rm -rf "$dir"; mkdir -p "$dir"
    cp "$nb" "$dir/"
    cat > "$dir/kernel-metadata.json" <<EOF
{
  "id": "$ref",
  "title": "$slug",
  "code_file": "$(basename "$nb")",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
EOF
    "$KAGGLE" kernels push -p "$dir" --accelerator NvidiaTeslaT4
    echo "pushed $ref"
    ;;
  status)
    "$KAGGLE" kernels status "$ref"
    ;;
  fetch)
    out="kaggle/out/$slug"
    mkdir -p "$out"
    "$KAGGLE" kernels output "$ref" -p "$out"
    ls -la "$out"
    ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
