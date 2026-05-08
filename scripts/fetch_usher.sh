#!/usr/bin/env bash
# Clone or update https://github.com/yatisht/usher into scripts/usher for
# BUILD_GET_MUTATIONS (CMake defaults USHER_ROOT to that path).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${SCRIPT_DIR}/usher"
REPO="${USHER_GIT_URL:-https://github.com/yatisht/usher.git}"

mkdir -p "${SCRIPT_DIR}"

if [[ -d "${TARGET}/.git" ]]; then
  git -C "${TARGET}" pull --ff-only
else
  git clone --depth 1 "${REPO}" "${TARGET}"
fi

if [[ ! -f "${TARGET}/parsimony.proto" ]]; then
  echo "fetch_usher.sh: expected parsimony.proto under ${TARGET}" >&2
  exit 1
fi

echo "UShER ready at ${TARGET}"
