# Troubleshooting

Keyed to the exact messages the platform prints. Start with `keystone doctor`: it connects to every service directly, checks the database's migration revision, TLS validity, free disk and the running app's readiness, and names the problem.

| You see | It means | Do this |
|---|---|---|
| `Missing Authorization: Bearer header` (401) | no API key on the request | send `Authorization: Bearer ks-…`, or `x-api-key: ks-…` from an Anthropic SDK |
| `Invalid API key` (401) | the key is unknown, revoked or expired, or `VS_SECRET_KEY` changed since it was minted | mint a new key (`keystone keys-create`); keep `VS_SECRET_KEY` stable — it peppers every key hash |
| `API key lacks required scope` (403) | the key was minted without `inference`, `agent` or `finetune` | `keystone keys-create <tenant> --scopes inference,agent,finetune` |
| `VS_SECRET_KEY is not set: in production it must be a stable secret` at startup | production refuses a generated pepper | set `VS_SECRET_KEY` (`openssl rand -hex 32`) in `.env` or the Secret; `keystone init` writes one |
| `still a placeholder/default value` in `doctor` | a secret is the `.env.example` placeholder | `keystone init` regenerates all secrets, or replace the value by hand |
| `no healthy model endpoint for role 'coding'; retry in 10s` (503, `Retry-After`) | no endpoint in the role's fallback chain answered its health probe | `keystone doctor` shows each endpoint; the Model Library shows the breaker and last error; check `VLLM_CODING_URL`, the model server's logs, GPU memory |
| `Monthly budget exhausted: $X of $Y spent this month` (429) | the tenant's dollar budget is spent | raise or clear it: `keystone tenants set-limits <tenant> --monthly-budget-usd 0` (0 = none) |
| `Daily token budget exhausted` / `Monthly token budget exhausted` (429) | a token cap set on the tenant or key | `keystone tenants set-limits`, or a key without an override |
| `max_tokens=… exceeds this deployment's limit` (422) | `MAX_TOKENS_PER_REQUEST` is set | raise it or set 0 (the model's own limit) |
| `tools and response_format cannot be combined in one request` (422) | vLLM serves guided decoding and tool calling separately | send one or the other per request |
| `content block type 'image' is not supported by this gateway` (400) | the Messages API translation covers text and tools only | send text; images and documents are refused rather than dropped |
| `repository_url host '…' is not in GIT_ALLOWED_HOSTS` (400) | the agent may only clone from hosts you allow | add the host to `GIT_ALLOWED_HOSTS` (a JSON list) and restart |
| `repository_url must use https:// or ssh://` (400) | plain `http://` or a bare path | use the host's https URL |
| `git clone failed: …` (400) | the host answered but the clone did not | check the token's permissions and the branch name; run the same clone from the sandbox daemon's host |
| `Only a completed job with a real output_model_path can be promoted` (409) | the fine-tune has not finished or produced no adapter | `keystone finetune status <job>` |
| `verdict fail: held-out loss … not better than the base model` (409) | the adapter did not beat the base model on held-out data | more or better data, more epochs; an admin may `--force`, which is audited |
| `verdict unknown: no held-out comparison` (409) | the job trained without a holdout split | rerun with `holdout_ratio > 0` (the guided path defaults to 20 %) |
| `force=true requires an admin user linked to this API key` (403) | the key is not linked to an `admin` user | `keystone users add … --role admin` and link the key |
| `lora_modules manifest not written` in a promote response | `FINETUNING_OUTPUT_DIR` is not mounted or writable on the API host | mount the same volume the trainer wrote to; the promotion itself succeeded |
| `not loaded live` in a promote response | vLLM refused `/v1/load_lora_adapter` | set `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` on the server; the manifest covers the next restart |
| `Concurrency limit exceeded` (429) on task submit | `max_concurrent_agents` is set for the tenant | wait, or raise it (`keystone tenants set-limits --max-concurrent-agents 0`) |
| `no alembic_version table` / `database at … code head is …` in `doctor` | migrations not applied for this version | `alembic upgrade head` (`make db-migrate`; Helm does it in the pre-upgrade hook) |
| `must have at least $0.01 in your account balance` from RunPod | the RunPod account is unfunded | add credit, rerun `keystone deploy runpod-serverless` |
| `Could not find .env.example` | `keystone init` ran outside a checkout | run it from inside the repository |
| `KEYSTONE_ROOT_ADMIN_TOKEN is not set` | an admin CLI command without the bootstrap token | `export KEYSTONE_ROOT_ADMIN_TOKEN=$(grep ^KEYSTONE_ROOT_ADMIN_TOKEN= .env \| cut -d= -f2)` |
| `X-VS-Route-Decision: … reason=fallback` | the primary role was unhealthy; a fallback served | expected during an outage; check the primary endpoint |
| the web UI shows `Failed to load … (401)` | no key saved in the browser | paste a key in the top bar and Save |

Every response carries `X-Request-ID`; quote it when asking for help — every log line for that request carries the same id.
