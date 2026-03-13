#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${1:-$repo_root/data}"
archive_path="$data_root/lra_release.gz"
extract_root="$data_root/lra_release"
pathfinder_root="$data_root/pathfinder"
archive_url="https://storage.googleapis.com/long-range-arena/lra_release.gz"
mirror_url="https://connectomics.clps.brown.edu/tf_records/PathFinder/pathfinder128/curv_contour_length_14/"

mkdir -p "$data_root"

if [[ -d "$pathfinder_root/pathfinder128" ]]; then
  echo "Pathfinder-128 already present at $pathfinder_root/pathfinder128"
  exit 0
fi

download_archive() {
  rm -f "$archive_path"
  echo "Downloading LRA archive to $archive_path"
  if command -v wget >/dev/null 2>&1; then
    wget -O "$archive_path" "$archive_url"
  else
    curl -L "$archive_url" -o "$archive_path"
  fi
}

download_pathx_mirror() {
  if ! command -v wget >/dev/null 2>&1; then
    echo "wget is required for recursive mirror download fallback" >&2
    exit 1
  fi
  mkdir -p "$pathfinder_root"
  echo "Downloading Pathfinder-128 hard split from mirror into $pathfinder_root"
  wget \
    --recursive \
    --no-parent \
    --no-host-directories \
    --cut-dirs=2 \
    --reject "index.html*" \
    --continue \
    --directory-prefix "$pathfinder_root" \
    "$mirror_url"
}

extract_archive() {
  rm -rf "$extract_root"
  mkdir -p "$extract_root"
  echo "Extracting LRA archive into $extract_root"
  tar -xvf "$archive_path" -C "$extract_root"
}

prepare_from_archive() {
  mkdir -p "$pathfinder_root"
  release_root=""
  for candidate in \
    "$extract_root/lra_release" \
    "$extract_root/lra_release/lra_release" \
    "$extract_root"; do
    if [[ -d "$candidate/pathfinder128" ]]; then
      release_root="$candidate"
      break
    fi
  done

  if [[ -z "$release_root" ]]; then
    return 1
  fi

  for resolution in 32 64 128 256; do
    source_dir="$release_root/pathfinder${resolution}"
    target_dir="$pathfinder_root/pathfinder${resolution}"
    if [[ -d "$source_dir" && ! -e "$target_dir" ]]; then
      mv "$source_dir" "$target_dir"
    fi
  done

  if [[ -d "$release_root/listops-1000" && ! -e "$data_root/listops" ]]; then
    mv "$release_root/listops-1000" "$data_root/listops"
  fi

  if [[ -d "$release_root/tsv_data" && ! -e "$data_root/aan" ]]; then
    mv "$release_root/tsv_data" "$data_root/aan"
  fi
}

if [[ ! -s "$archive_path" ]]; then
  if ! download_archive; then
    rm -f "$archive_path"
  fi
fi

if [[ -s "$archive_path" ]]; then
  if extract_archive && prepare_from_archive; then
    echo "Prepared datasets under $data_root"
    echo "Path-X data root is $pathfinder_root/pathfinder128"
    exit 0
  fi
  echo "Archive was unavailable or did not contain usable Pathfinder data; falling back to mirror"
fi

download_pathx_mirror

echo "Prepared datasets under $data_root"
echo "Path-X data root is $pathfinder_root/pathfinder128"
