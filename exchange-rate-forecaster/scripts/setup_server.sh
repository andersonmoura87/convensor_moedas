#!/usr/bin/env bash
# setup_server.sh — Bootstrap a fresh Ubuntu 22.04 / Debian 12 server
# Run as root or with sudo.
# Usage: curl -fsSL https://raw.githubusercontent.com/.../setup_server.sh | bash

set -euo pipefail

DOCKER_COMPOSE_VERSION="2.27.0"
PROJECT_DIR="/opt/ml-platform"
SERVICE_USER="mlplatform"

log() { echo -e "\033[1;34m[setup]\033[0m $*"; }
err() { echo -e "\033[1;31m[error]\033[0m $*" >&2; exit 1; }

[[ $(id -u) -eq 0 ]] || err "Run as root"

log "Updating system packages..."
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq \
    ca-certificates curl gnupg lsb-release git \
    htop jq net-tools ufw fail2ban

# ── Docker ────────────────────────────────────────────────────────────────────
log "Installing Docker..."
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update -qq
apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin

# Docker Compose v2 (plugin)
COMPOSE_URL="https://github.com/docker/compose/releases/download/v${DOCKER_COMPOSE_VERSION}/docker-compose-linux-x86_64"
curl -fsSL "${COMPOSE_URL}" -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# ── Service user ──────────────────────────────────────────────────────────────
log "Creating service user ${SERVICE_USER}..."
id "${SERVICE_USER}" &>/dev/null || useradd -m -s /bin/bash "${SERVICE_USER}"
usermod -aG docker "${SERVICE_USER}"

# ── Project directory ─────────────────────────────────────────────────────────
log "Setting up project directory at ${PROJECT_DIR}..."
mkdir -p "${PROJECT_DIR}"
chown "${SERVICE_USER}:${SERVICE_USER}" "${PROJECT_DIR}"

# ── Firewall ──────────────────────────────────────────────────────────────────
log "Configuring UFW firewall..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 3000/tcp   # Grafana
ufw allow 9090/tcp   # Prometheus (restrict in production)
ufw --force enable

# ── Fail2ban ─────────────────────────────────────────────────────────────────
log "Enabling fail2ban..."
systemctl enable --now fail2ban

# ── Docker daemon hardening ───────────────────────────────────────────────────
log "Configuring Docker daemon..."
cat > /etc/docker/daemon.json << 'EOF'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "default-ulimits": {
    "nofile": {
      "Hard": 64000,
      "Name": "nofile",
      "Soft": 64000
    }
  },
  "live-restore": true
}
EOF

systemctl restart docker

# ── Systemd service ───────────────────────────────────────────────────────────
log "Installing systemd service..."
cat > /etc/systemd/system/ml-platform.service << EOF
[Unit]
Description=ML Platform — Exchange Rate Forecaster
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${PROJECT_DIR}
User=${SERVICE_USER}
ExecStart=/usr/bin/docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
ExecStop=/usr/bin/docker compose -f docker-compose.yml -f docker-compose.monitoring.yml down
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ml-platform

log "Server setup complete!"
log "Next steps:"
log "  1. Copy your project files to ${PROJECT_DIR}"
log "  2. Create ${PROJECT_DIR}/.env from .env.example"
log "  3. Run: systemctl start ml-platform"
