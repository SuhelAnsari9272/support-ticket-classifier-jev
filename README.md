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