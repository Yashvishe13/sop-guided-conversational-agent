# Hosted demo on AWS EC2

Live demo: https://98-81-151-1.sslip.io (instance `i-0f7763a12dc2433e2`, us-east-1, `t3.small`,
everything tagged `Project=claims-sop-demo`). Open to anyone; it runs on the OpenAI key in the
instance's `.env`, so set a spending limit on that key.

Layout on the instance (`/opt/claims`): this directory's `compose.yaml` and `Caddyfile`, plus a
`.env` (mode 600) with `OPENAI_API_KEY`, `STATE_ENCRYPTION_KEY`, and `DEMO_HOST`. Caddy terminates
HTTPS with an automatic Let's Encrypt certificate for the free `<ip-with-dashes>.sslip.io` name and
proxies to the app, which is not published on the host. The app runs with `APP_ENV=production`
(external encryption key required, Secure cookies). SSH is open only to the deployer's IP.

## How it was created

```bash
export AWS_PROFILE=terminal-project AWS_DEFAULT_REGION=us-east-1
aws ec2 create-key-pair --key-name claims-sop-demo --key-type ed25519 --query KeyMaterial --output text > ~/.ssh/claims-sop-demo.pem
aws ec2 create-security-group --group-name claims-sop-demo-sg ...   # 80, 443 open; 22 from your IP
aws ec2 run-instances --image-id <al2023 x86_64> --instance-type t3.small --key-name claims-sop-demo \
  --security-group-ids <sg> --user-data file://user-data.sh ...      # installs Docker + Compose
```

## Update the app

```bash
docker buildx build --platform linux/amd64 -t insurance-claims-agent:amd64 --load apps/insurance_claims
docker save insurance-claims-agent:amd64 | gzip > image.tar.gz
scp -i ~/.ssh/claims-sop-demo.pem image.tar.gz ec2-user@98.81.151.1:/opt/claims/
ssh -i ~/.ssh/claims-sop-demo.pem ec2-user@98.81.151.1 \
  'cd /opt/claims && gunzip -c image.tar.gz | docker load && rm image.tar.gz && docker compose up -d'
```

Logs: `ssh ... 'cd /opt/claims && docker compose logs --tail 100 app'`.

## Stop or remove it

```bash
export AWS_PROFILE=terminal-project AWS_DEFAULT_REGION=us-east-1
aws ec2 stop-instances --instance-ids i-0f7763a12dc2433e2        # pause billing for compute (disk still billed)
aws ec2 terminate-instances --instance-ids i-0f7763a12dc2433e2   # delete it (disk deleted too)
aws ec2 delete-security-group --group-name claims-sop-demo-sg    # after termination
aws ec2 delete-key-pair --key-name claims-sop-demo
```

A stopped instance gets a new public IP when started again, so the sslip.io hostname changes:
update `DEMO_HOST` in `.env` and restart with `docker compose up -d`.
