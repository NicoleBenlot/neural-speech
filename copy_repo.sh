#!/usr/bin/env bash
#
# copy_repo.sh - sync this repo with your USB/flash drive (both directions),
# skipping .venv and Python caches. The drive is found by its exact volume
# label, or you can give a destination/source directory manually.
#
# Usage:
#   bash copy_repo.sh                       # PUSH  repo -> <flash>/projects/neural-speech
#   bash copy_repo.sh -r                    # PULL  <flash>/projects/neural-speech -> this PC (overwrite repo files, keeps .venv)
#   bash copy_repo.sh -p myfolder           # use folder <flash>/myfolder instead of "projects"
#   bash copy_repo.sh -d "E:\backups" -r    # manual base dir in either direction
#   bash copy_repo.sh -d "E:\backups" -f    # push to a manual dir, overwrite if exists
#
# Flags:
#   -r              reverse (pull from flash -> PC); default is push (PC -> flash)
#   -d <BASE DIR>   use this directory as the base instead of the detected flash drive
#   -p <FOLDER>     name of the folder on the base (default "projects")
#   -f              overwrite the destination even if it already exists
#
# Defaults (edit these as you like):
FLASH_LABEL="${FLASH_LABEL:-KINGSTONE E}"          # exact volume label of the flash drive
DEST_PARENT_NAME="${DEST_PARENT_NAME:-projects}"   # folder on the flash/base dir
DEST_FOLDER_NAME="${DEST_FOLDER_NAME:-neural-speech}"

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REVERSE=0
FORCE=0
MANUAL=""
PARENT="$DEST_PARENT_NAME"

usage() {
  sed -n '2,21p' "${BASH_SOURCE[0]}"
  exit "${1:-0}"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -r) REVERSE=1; shift ;;
    -f) FORCE=1; shift ;;
    -d) MANUAL="${2:-}"; shift 2 ;;
    -p) PARENT="${2:-}"; shift 2 ;;
    *) usage 1 ;;
  esac
done

detect_flash() {
  powershell -NoProfile -Command "Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=2' | Where-Object { \$_.VolumeName -eq '$FLASH_LABEL' } | Select-Object -ExpandProperty DeviceID" \
    | tr -d '\r' | sed '/^$/d'
}

if [ -n "$MANUAL" ]; then
  BASE="$MANUAL"
else
  DRIVE="$(detect_flash)"
  if [ -z "$DRIVE" ]; then
    echo "ERROR: no removable drive labelled '$FLASH_LABEL' was found." >&2
    echo "Removable drives present:" >&2
    powershell -NoProfile -Command "Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=2' | Select-Object DeviceID,VolumeName,Size | Format-Table -AutoSize" >&2
    exit 1
  fi
  BASE="${DRIVE}/${PARENT}"
fi

unixpath() {
  cygpath -u "$1" 2>/dev/null || echo "$1"
}

EXCLUDES=(
  --exclude='.venv'
  --exclude='.pytest_cache'
  --exclude='.mypy_cache'
  --exclude='.ruff_cache'
  --exclude='*__pycache__*'
  --exclude='*.py[cod]'
)

BASE_UNIX="$(unixpath "$BASE")"
REMOTE_UNIX="$BASE_UNIX/$DEST_FOLDER_NAME"

if [ "$REVERSE" -eq 0 ]; then
  TARGET="$REMOTE_UNIX"
  echo "PUSH: repo -> $BASE/$DEST_FOLDER_NAME"
  [ -e "$TARGET" ] && [ "$FORCE" -eq 0 ] && {
    echo "ERROR: '$BASE/$DEST_FOLDER_NAME' already exists - re-run with -f to overwrite." >&2
    exit 1
  }
  mkdir -p "$BASE_UNIX"
  rm -rf "$TARGET"
  mkdir -p "$TARGET"
  tar -C "$REPO" "${EXCLUDES[@]}" -cf - . | tar -C "$TARGET" -xf -
else
  TARGET="$REPO"
  echo "PULL: $BASE/$DEST_FOLDER_NAME -> repo ($REPO)"
  [ -d "$REMOTE_UNIX" ] || {
    echo "ERROR: '$BASE/$DEST_FOLDER_NAME' does not exist on the flash drive." >&2
    exit 1
  }
  tar -C "$REMOTE_UNIX" "${EXCLUDES[@]}" -cf - . | tar -C "$TARGET" -xf -
fi

echo "Verifying against $TARGET ..."
ok=1
for p in ns.py commands.txt checkpoints/mms/v026 checkpoints/mms/v030; do
  if [ -e "$TARGET/$p" ]; then
    echo "  OK   $p"
  else
    echo "  MISSING  $p" >&2
    ok=0
  fi
done
if [ "$REVERSE" -eq 0 ]; then
  if [ -e "$TARGET/.venv" ]; then
    echo "  FAIL  .venv present in copy" >&2
    ok=0
  else
    echo "  OK   .venv excluded"
  fi
else
  echo "  OK   local .venv kept untouched"
fi
if [ "$ok" -eq 0 ]; then
  exit 1
fi

if [ "$REVERSE" -eq 0 ]; then
  echo "Done. Copied size: $(du -sh "$TARGET" | cut -f1)"
  echo "To pull back later: bash copy_repo.sh -r"
else
  echo "Done. Restored from flash (local .venv kept)."
fi