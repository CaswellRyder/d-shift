#!/bin/bash
# Isolated research runtime; never installs to /usr or changes the Pi runtime.
set -euo pipefail
task_root=${1:?Pass the absolute external build directory}
script_root=$(cd "$(dirname "$0")" && pwd)
expected_revision=72fbba3d20f4616d7312b5e2b7f79daf6e82f2fa
test "$(git -C "$task_root/tensorflow" rev-parse HEAD)" = "$expected_revision"
for patch_name in xnnpack-disabled fp16-cpu-headers; do
  if git -C "$task_root/tensorflow" apply --check "$script_root/$patch_name.patch" 2>/dev/null; then
    git -C "$task_root/tensorflow" apply "$script_root/$patch_name.patch"
  else
    git -C "$task_root/tensorflow" apply --reverse --check "$script_root/$patch_name.patch"
  fi
done
export PATH="$task_root/tools/bin:$PATH"
export DTR_ZIG="$task_root/tools/lib/python3.11/site-packages/ziglang/zig"
export ZIG_GLOBAL_CACHE_DIR="$task_root/zig-cache"
export ZIG_LOCAL_CACHE_DIR="$task_root/zig-local-cache"
test "$("$DTR_ZIG" version)" = 0.14.1
test -x "$task_root/host-tools/bin/flatc"
cmake -G Ninja -S "$task_root/tensorflow/tensorflow/lite/c" \
  -B "$task_root/build-armv6" \
  -DCMAKE_TOOLCHAIN_FILE="$script_root/armv6-zig.cmake" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DTFLITE_HOST_TOOLS_DIR="$task_root/host-tools/bin" \
  -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_RUY=OFF \
  -DTFLITE_ENABLE_NNAPI=OFF -DTFLITE_ENABLE_GPU=OFF \
  -DTFLITE_ENABLE_RESOURCE=ON -DTFLITE_ENABLE_EXTERNAL_DELEGATE=OFF \
  -DTFLITE_C_BUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--no-undefined -Wl,--strip-debug" \
  -DCMAKE_C_FLAGS="-fPIC -fno-fast-math -include $script_root/release-defines.h" \
  -DCMAKE_CXX_FLAGS="-fPIC -fno-fast-math -include $script_root/release-defines.h" \
  -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG" \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
  -DFETCHCONTENT_SOURCE_DIR_FLATBUFFERS="$task_root/flatc-host/flatbuffers"
cmake --build "$task_root/build-armv6" --target tensorflowlite_c -j 4
