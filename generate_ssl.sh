#!/usr/bin/env bash
set -e
if [ -f cert.pem ] && [ -f key.pem ]; then
  echo "cert.pem/key.pem already exist."
  exit 0
fi
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"
echo "Created cert.pem and key.pem"
