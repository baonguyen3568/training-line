#!/usr/bin/env bash
# Run a pinned, checksum-verified gitleaks binary (no Go toolchain needed). Usage: scripts/tools/gitleaks.sh <args>
# The release tarball is verified on download and the extracted binary is re-verified before EVERY run.
# Bump: change VERSION + the tarball SHA256 (from the release's gitleaks_<ver>_checksums.txt) + the binary SHA256
# (sha256 of the `gitleaks` file inside that verified tarball), in a dedicated commit.
set -euo pipefail

VERSION="8.30.1"
case "$(uname -m)" in
  aarch64 | arm64)
    ARCH="arm64"
    TARBALL_SHA256="e4a487ee7ccd7d3a7f7ec08657610aa3606637dab924210b3aee62570fb4b080"
    BINARY_SHA256="00e91bbe655bd7c47753e8cfe61cb76ea1a5d7e7702fe161ee40102b46b3823b"
    ;;
  x86_64)
    ARCH="x64"
    TARBALL_SHA256="551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb"
    BINARY_SHA256="88f91962aa2f93ac6ab281d553b9e125f5197bbbce38f9f2437f7299c32e5509"
    ;;
  *) echo "gitleaks.sh: unsupported arch $(uname -m)" >&2; exit 2 ;;
esac

ROOT="$(git rev-parse --show-toplevel)"
DIR="$ROOT/.cache/tools/gitleaks-$VERSION"
BIN="$DIR/gitleaks"

binary_ok() { [[ -x "$BIN" ]] && echo "$BINARY_SHA256  $BIN" | sha256sum --check --quiet --status; }

if ! binary_ok; then
  rm -f "$BIN"
  mkdir -p "$DIR"
  TARBALL="$DIR/gitleaks.tar.gz"
  curl -sSfL -o "$TARBALL" \
    "https://github.com/gitleaks/gitleaks/releases/download/v$VERSION/gitleaks_${VERSION}_linux_${ARCH}.tar.gz"
  echo "$TARBALL_SHA256  $TARBALL" | sha256sum --check --quiet
  tar -xzf "$TARBALL" -C "$DIR" gitleaks
  rm -f "$TARBALL"
  binary_ok || { echo "gitleaks.sh: extracted binary checksum mismatch" >&2; exit 1; }
fi

exec "$BIN" "$@"
