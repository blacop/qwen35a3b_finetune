#!/usr/bin/env bash
set -euo pipefail

CMD=${1:-status}
shift || true
SERVICES=(tensorboard.service mlflow.service wandb.service)

case "$CMD" in
  start|stop|restart|status|enable|disable)
    systemctl --user "$CMD" "${SERVICES[@]}"
    ;;
  logs)
    journalctl --user -u tensorboard.service -u mlflow.service -u wandb.service -f
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|enable|disable|logs}"
    exit 1
    ;;
esac
