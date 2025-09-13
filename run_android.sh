#!/usr/bin/env bash
set -euo pipefail

adb push build_android/enumDevices /data/local/tmp && adb shell /data/local/tmp/enumDevices
