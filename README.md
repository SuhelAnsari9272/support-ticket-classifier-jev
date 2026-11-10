# Support Ticket Classifier

## Configure API access

The classifier sends requests to OpenRouter. Set `TYPESAFE_API_KEY` to a valid
OpenRouter API key before running it. In PowerShell, set it for the current
terminal session (replace the placeholder with your key):

```powershell
$env:TYPESAFE_API_KEY = "your-openrouter-api-key"
python all_question.py
```

Alternatively, set `TYPESAFE_API_KEY` in your Windows environment variables and
open a new terminal. `OPENAI_API_KEY` is also accepted as a fallback for
compatibility with existing setups.

If no key is configured, the classifier exits with a message explaining how to
configure one instead of sending a request without authentication.

`all_question.py` runs the combined classification and prints its structured
`TicketClassificationOutput` result.

## Evaluate the classifier

From the repository root, run:

```powershell
python evaluation\evaluate_jev_classifier.py
```

The evaluator uses `production_modules.all_question:get_classification` by
default and reads `data\support_ticket_eval_dataset.json`. The dataset contains
40 tickets, so the default `--repeats 1` makes 40 classifier calls. Increase
`--repeats` to repeat each ticket (for example, `--repeats 3` makes 120 calls).

## Continuous integration

GitHub Actions runs on `push` events only (there is no pull-request trigger).
It installs dependencies from `requirements.txt` and runs the offline pytest
suite. No provider key is needed for these tests.

The live classifier regression test is not part of CI yet because its
evaluation dataset is excluded from Git. To enable it later, make the dataset
available to the workflow through an approved private storage method or add a
reviewed, sanitized copy to the repository. The test makes provider requests
and uses API quota on every run.

Store provider credentials in GitHub, never in the repository: open
**Settings → Secrets and variables → Actions → New repository secret**, then
create a secret named `TYPESAFE_API_KEY`. The current offline workflow does
not read this secret. When enabling live CI or deployment, pass it only to the
specific step or job that needs it using
`${{ secrets.TYPESAFE_API_KEY }}`. The live test uses the thresholds documented
in `tests/regression_test.py`.

## Run with Docker Compose

Install Docker Desktop (Windows/macOS) or Docker Engine with the Compose
plugin (Linux). From the repository root, copy `.env.example` to `.env` and
put your real provider key in `.env`:

```powershell
Copy-Item .env.example .env
# Edit .env and replace the placeholder with your key.
docker compose up --build -d
```

The Compose service reads `.env` at container runtime; the key is not written
into the Dockerfile or Compose file and `.env` is excluded from Git and the
Docker build context. Never pass the key as a Docker build argument or commit
the `.env` file.

Check service health and logs:

```powershell
docker compose ps
docker compose logs -f classifier
Invoke-RestMethod http://localhost:8000/health
```

The API is available at `http://localhost:8000` and its interactive docs are
at `http://localhost:8000/docs`. To stop it, run `docker compose down`.
Set `APP_PORT` in `.env` if host port 8000 is already in use, for example
`APP_PORT=8080`; the container continues to listen on port 8000.

This Compose setup is intended for local/container testing. For production
deployment, supply `TYPESAFE_API_KEY` through the hosting platform's secret
manager rather than a checked-in file, and put the API behind HTTPS and an
appropriate access-control layer.