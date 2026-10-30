#!/usr/bin/env bash
# Build the serial DHSVM executable from pnnl/DHSVM-PNNL at a pinned commit.
#
# Usage: scripts/build_dhsvm.sh [PREFIX]
#
#   PREFIX defaults to ${WADER_CACHE:-$HOME/.cache/wader}/dhsvm/<commit[:7]>.
#   The executable is installed as PREFIX/bin/DHSVM; its path is printed as the
#   last line of stdout. If it already exists, nothing is rebuilt.
#
# Only the source tree is fetched (partial clone + sparse checkout); the
# TestCase data and the PDF tutorials are skipped. Build flags follow the
# project brief: Release, no X11, no NetCDF. GCC 13 needs no extra flags;
# for GCC >= 14 you may need CFLAGS="-std=gnu17 -Wno-implicit-function-declaration".
set -euo pipefail

DHSVM_REPO="${DHSVM_REPO:-https://github.com/pnnl/DHSVM-PNNL.git}"
DHSVM_COMMIT="${DHSVM_COMMIT:-e6f333d0e70ed72345d52f1ab83e35317fa92458}"
PREFIX="${1:-${WADER_CACHE:-$HOME/.cache/wader}/dhsvm/${DHSVM_COMMIT:0:7}}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 2)}"

if [[ -x "$PREFIX/bin/DHSVM" ]]; then
  echo "DHSVM already built at $PREFIX/bin/DHSVM" >&2
  echo "$PREFIX/bin/DHSVM"
  exit 0
fi

src="$PREFIX/src"
build="$PREFIX/build"
rm -rf "$src" "$build"
mkdir -p "$src"

echo "Fetching DHSVM-PNNL source at $DHSVM_COMMIT ..." >&2
git -C "$src" init -q
git -C "$src" remote add origin "$DHSVM_REPO"
git -C "$src" config core.sparseCheckout true
printf '/*\n!/TestCase/\n!/*.pdf\n' > "$src/.git/info/sparse-checkout"
git -C "$src" fetch -q --depth 1 --filter=blob:none origin "$DHSVM_COMMIT"
git -C "$src" checkout -q --detach FETCH_HEAD
test "$(git -C "$src" rev-parse HEAD)" = "$DHSVM_COMMIT"

echo "Building (log: $PREFIX/build.log) ..." >&2
if ! { cmake -S "$src" -B "$build" \
         -D CMAKE_BUILD_TYPE=Release \
         -D DHSVM_USE_X11=OFF \
         -D DHSVM_USE_NETCDF=OFF &&
       cmake --build "$build" -j "$JOBS"; } > "$PREFIX/build.log" 2>&1; then
  tail -n 60 "$PREFIX/build.log" >&2
  echo "DHSVM build failed" >&2
  exit 1
fi

mkdir -p "$PREFIX/bin"
cp "$build/DHSVM/sourcecode/DHSVM" "$PREFIX/bin/DHSVM"
echo "$PREFIX/bin/DHSVM"
