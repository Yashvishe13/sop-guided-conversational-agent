# SOP-guided insurance claims agent

A chat agent that follows a claims support SOP (standard operating procedure): it verifies the caller,
works out which claim they mean, explains it from the claim record, and closes with an optional
email summary. It talks naturally, while code and an independent guard model enforce the rules.

The app lives in [`apps/insurance_claims`](apps/insurance_claims). Its
[README](apps/insurance_claims/README.md) has the full configuration, tests, and layout.
The SOP itself is defined as data in [`sop.toml`](apps/insurance_claims/sop.toml) (phase order,
tools per phase, transitions, and which code enforces each rule). To read it as a table per phase,
run `docker compose exec claims-agent python -m insurance_claims.agent.sop` while the app is running.

## Run it (Docker)

Requires Docker with Compose and an OpenAI API key. From this directory:

```bash
cp .env.example .env        # then set OPENAI_API_KEY=... in .env
docker compose up --build
```

Open http://localhost:8000. The key is read at runtime from `.env` or your shell and is never
copied into the image. To pass it without a file, or to use another port:

```bash
OPENAI_API_KEY=sk-... PORT=8080 docker compose up --build     # then open http://localhost:8080
```

Without a key you can still click through with the offline demo model: `MODEL_PROVIDER=fake docker compose up --build`.

Stop with `Ctrl+C`; reset all demo data with `docker compose down -v`.

### Plain Docker (no Compose)

```bash
docker build -t insurance-claims-agent apps/insurance_claims
docker run --rm -p 8000:8000 -e OPENAI_API_KEY=sk-... -v claims-data:/data insurance-claims-agent
```

To hand the image over as a file: `docker save insurance-claims-agent | gzip > insurance-claims-agent.tar.gz`,
and on the other machine `docker load < insurance-claims-agent.tar.gz`, then the same `docker run`.
