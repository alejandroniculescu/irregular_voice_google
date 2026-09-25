#!/bin/bash
# Keeps the USB bridge to an Android phone up for the web demo: whenever a phone
# (re)connects, forward its localhost:PORT to this Mac with adb reverse.
# Usage: scripts/phone-bridge.sh [port]   (Ctrl+C stops)
port=${1:-8766}
while true; do
  adb wait-for-device 2>/dev/null
  if ! adb reverse --list 2>/dev/null | grep -q "tcp:$port tcp:$port"; then
    if adb reverse "tcp:$port" "tcp:$port" >/dev/null 2>&1; then
      echo "$(date +%H:%M:%S) bridge up: $(adb get-serialno) -> localhost:$port"
    fi
  fi
  sleep 2
done
