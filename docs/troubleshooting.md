# Troubleshooting

Common errors, what causes them and how to fix them. Every message below was copied from a real
run of OpenBench AI v0.5.0 (paths shortened). If your problem is not here, please open an issue
or ask in [Discussions](https://github.com/lakshaychhaba0001-sys/openbench-ai/discussions).

## Ollama is not running

```text
Provider Check [ollama]: UNAVAILABLE — Could not connect to Ollama server at http://localhost:11434 ([Errno 111] Connection refused). Ensure Ollama is installed and running (`ollama serve`) or verify the --endpoint URL.
```

During `openbench run` the same problem shows as a failed run:

```text
Run 'run_26ee47d2ac7e' [failed]: 0/20 completed (provider=ollama, model=llama3.2, ...)
  20 item(s) failed. First error: [connection_error] Could not connect to Ollama server at http://localhost:11434 ([Errno 111] Connection refused). ...
```

**Fix:** start Ollama (`ollama serve`), pull the model (`ollama pull llama3.2`), and check with
`openbench providers check ollama --model llama3.2`. If Ollama runs elsewhere, pass
`--endpoint http://host:11434`. Then finish the run with `openbench runs resume run_<id>`.

## An OpenAI-compatible server cannot be reached

```text
[connection_error] Could not connect to the OpenAI-compatible server at http://127.0.0.1:9/v1 ([Errno 111] Connection refused). Check that the server is running and that --endpoint includes the /v1 prefix if the server needs it.
```

**Fix:** check the server is running and the port is right. Most servers (vLLM, llama.cpp,
LM Studio) expect the `/v1` prefix in `--endpoint`. `openbench providers check openai_compatible
--endpoint ... --model ...` lists the models the server reports.

## Plain HTTP to a remote host is refused

```text
Validation/Evaluation Error: Plain HTTP is only allowed for loopback hosts (localhost / 127.0.0.1). Use 'https://' for remote host 'example.com' or explicitly enable allow_remote_http=True.
```

**Fix:** use `https://`. If the server is on a network you trust and has no TLS, set
`allow_remote_http: true` in the `provider:` section of a profile.

## The API key is rejected (HTTP 401 or 403)

OpenBench reads the key from an environment variable for each request; it never stores it.
The error names the variable it read.

**Fix:** set it (`export OPENAI_API_KEY=...`) or name another one with `--api-key-env MY_VAR`.

## A dataset does not load

```text
Validation/Evaluation Error: Missing required field 'id' (file=.../noid.jsonl, line=2)
Validation/Evaluation Error: Duplicate record ID 'a' found on line 2 (first defined on line 1) (file=.../dup.jsonl, line=2, id=a)
Validation/Evaluation Error: Malformed JSON on line 1: Expecting ',' delimiter (file=.../bad.jsonl, line=1)
```

**Fix:** every line must be one JSON object with `id`, `category`, `prompt` and
`expected_answer`, and every `id` must be unique. The message gives the file and line.
`openbench validate-dataset PATH --quality-checks` reports every problem at once, including
duplicates, placeholders and broken keywords. See [dataset_catalog.md](dataset_catalog.md).

## A provider name is not recognised

```text
Validation/Evaluation Error: Unknown model provider 'nosuch'. Registered providers: huggingface_transformers, local_callable, local_lookup, mock, ollama, openai_compatible
```

**Fix:** use one of the listed names (`openbench providers list`). If the provider comes from a
plugin, check that the plugin is installed (`openbench plugins list`); see [plugins.md](plugins.md).

## A run or group ID is not found

```text
Error: "Experiment run 'run_doesnotexist' does not exist in .../experiments"
```

**Fix:** list what exists with `openbench runs list`. Commands look in `./experiments` unless
you pass `--experiments-dir`. Use the same directory you used for `openbench run`.

## `analyze` refuses to compare

```text
Validation/Evaluation Error: Inputs are incompatible for inferential comparison:
alphabaseline-7b#1 vs alphabaseline-7b#2:
- [metric_not_in_arm] Metric 'exact_match' was not computed for challenger.
Use exploratory mode with a stated reason for a descriptive-only comparison.
```

**Fix:** compare like with like: the same dataset content, metrics, metric versions and
normalization. The message lists every problem. Re-run or re-evaluate one side so they match.
`--exploratory --exploratory-reason "..."` gives a descriptive-only comparison when you
knowingly compare different setups. See [model_comparison.md](model_comparison.md).

## The response cache is busy

If another OpenBench process holds the cache for more than 30 seconds, a run stops with a
message naming `cache_entries.lock`.

**Fix:** wait for the other run, or give each process its own `--cache-dir`. See
[reliability.md](reliability.md).

## A plugin fails to load

The error names the plugin, its entry point and its package. **Fix:** reinstall or update the
plugin, or set `OPENBENCH_DISABLE_PLUGINS=1` to run without plugins.

## Equivalence is "not established"

That is not an error. With few items, intervals are wide, so equivalence usually cannot be
shown. It is never evidence that the models differ. Use `openbench plan` to see how many items
you need. See [ranking_and_planning.md](ranking_and_planning.md) and
[reading_a_report.md](reading_a_report.md).
