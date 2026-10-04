# SOP-guided insurance claims agent

A chat agent that follows a claims support SOP (standard operating procedure): it verifies the caller,
works out which claim they mean, explains it from the claim record, and closes with an optional
email summary. It talks naturally, while code and an independent guard model enforce the rules.

The app lives in [`apps/insurance_claims`](apps/insurance_claims). Its
[README](apps/insurance_claims/README.md) has the full configuration, tests, and layout.
The SOP itself is defined as data in [`sop.toml`](apps/insurance_claims/sop.toml) (phase order,
tools per phase, transitions, and which code enforces each rule); print it in readable form with
`python -m insurance_claims.agent.sop` (from `apps/insurance_claims`, with `PYTHONPATH=src`).

## Hosted demo

https://98-81-151-1.sslip.io (AWS EC2, real `gpt-5.6-luna`).
Follow the walkthrough below in the chat page.

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

## Walk through the full workflow

The test UI is the chat page. Type the following.

1. **Identity verification.** `Hi, I'm calling about my denied healthcare claim from January.`
   The agent explains that claim details are private and asks for identity details. Then:
   `My name is Margaret Chen, date of birth March 15, 1985, SSN last four 4472.`
   Three matching details verify the caller (a policy number alone never counts).
2. **Intent resolution.** The agent remembers "denied healthcare claim from January" from before
   verification and selects claim CL-2048 without asking again (CL-2011 is also January healthcare,
   but closed). The progress bar moves to "Review claim".
3. **Claim processing.** It explains the denial from the record (missing pathology report and office
   note, appeal deadline already passed) and asks whether you can get the documents. Try
   `I can ask my doctor for both. How do I send them and how long does review take?`
4. **Post-case follow-up.** Say `That's all, thanks.` The agent offers an email summary with Send and
   Skip buttons; choose Send (or type "yes please"). The demo writes the email to a local outbox
   and says so honestly; nothing is delivered.

Things to try along the way: give the details over several messages or with a typo; say
`I'm calling for my mother`; ask `What is RL?` three times; refuse verification twice; or ask
`Is the pathology report on file?`.
