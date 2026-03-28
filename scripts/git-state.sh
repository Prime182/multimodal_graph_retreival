#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

echo "== Tracked files =="
git ls-files

echo
echo "== Modified tracked files =="
git diff --name-only
git diff --cached --name-only

echo
echo "== Untracked files =="
git status --porcelain=v1 --untracked-files=all | awk '$1 == "??" { $1=""; sub(/^ /, ""); print }'
