#!/bin/zsh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

conda run --no-capture-output -n project1-grading \
  python -m pytest grading_tests --github_link . -q \
  -k "not github_link_format" "$@"