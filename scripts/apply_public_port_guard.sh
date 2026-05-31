#!/usr/bin/env bash
set -euo pipefail

# Keep model/vLLM/proxy internals reachable from localhost only.
# Public ingress should go through Nginx -> external-api-gateway.
RULE_PORTS="${RULE_PORTS:-8001,8010,8013,8014,8025}"

# Open WebUI runs in Docker and reaches host services through docker bridge.
# Keep that internal path working while still blocking public interfaces.
if ! iptables -C INPUT -i docker0 -p tcp -m multiport --dports "$RULE_PORTS" -j ACCEPT 2>/dev/null; then
  iptables -I INPUT 1 -i docker0 -p tcp -m multiport --dports "$RULE_PORTS" -j ACCEPT
fi

if ! iptables -C INPUT -i br+ -p tcp -m multiport --dports "$RULE_PORTS" -j ACCEPT 2>/dev/null; then
  iptables -I INPUT 1 -i br+ -p tcp -m multiport --dports "$RULE_PORTS" -j ACCEPT
fi

if ! iptables -C INPUT ! -i lo -p tcp -m multiport --dports "$RULE_PORTS" -j DROP 2>/dev/null; then
  iptables -A INPUT ! -i lo -p tcp -m multiport --dports "$RULE_PORTS" -j DROP
fi
