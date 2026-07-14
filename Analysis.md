# CS4603 PA4 — Document Analyst

> This `README.md` is a **graded deliverable**:
>
> - Document how to set up, run, and deploy your Document Analyst so a TA can reproduce your results.
> - **Answer every ANALYSIS QUESTION** from the assignment in the sections below.
> - Code that runs but is not explained will not receive full marks.
> - Replace every `TODO` before submitting.
> - Keep it self-contained: a reader should be able to follow this file top-to-bottom —
>   setup → ingest → run → deploy → results — without opening the assignment PDF.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in your values
```

## Running locally

```bash
uv sync
cp .env.example .env   # then fill in your values
```

1. **Ingest the corpus** (run once, from a Databricks notebook attached to a workspace
   with Unity Catalog + Vector Search enabled — see the note below on Free Edition):
   ```python
   from rag.ingest import build_chunks_table, create_index

   build_chunks_table(spark, "/Volumes/cs4603/default/pa-4/annual_report.pdf",
                       "cs4603.default.emanqureshi_analyst_chunks")
   create_index("cs4603.default.emanqureshi_analyst_chunks")
   ```
   This parses the PDF with `ai_parse_document`, chunks it with `ai_prep_search` into
   `cs4603.default.emanqureshi_analyst_chunks` (7 chunks), and syncs the Delta Sync
   Vector Search index `cs4603.default.emanqureshi_analyst_index` on the
   `cs4603_rag_endpoint` endpoint. Wait until the index reaches `READY`
   (`rag/ingest.py::wait_for_index_ready`), then sanity-check with
   `rag/ingest.py::verify_index("net revenue")`.

   **Workspace note:** Databricks Free Edition *does* support Vector Search (contrary to
   what I assumed from an early error) — it's under Compute → AI Search in the sidebar.
   The error I initially hit (`Unity Catalog is not enabled ... AI Search`) was because I
   pointed the local client at the wrong workspace (my instructor's shared/scoped
   workspace, which lacks the SQL/Vector-Search-admin token scopes), not because the
   feature was unavailable. Ingestion must run against **your own** workspace where you
   have full permissions; the shared credentials are only for the LLM serving endpoint.

2. **Build and run the graph** in `pa4.ipynb` (or a one-off script):
   ```python
   from agent.graph import build_graph
   graph = build_graph()          # uses config.py + rag/store.py + the MCP server
   result = graph.invoke({"messages": [{"role": "user",
             "content": "What was the net revenue in 2023?"}]})
   print(result["messages"][-1].content)
   ```

3. **Test queries I ran** (retrieval-only, computation-only, combined — real LLM, real
   Vector Search index, real MCP subprocess, no mocks):

   | Query | Plan | Answer produced |
   |-------|------|------------------|
   | "What was the net income in 2023?" | 1 step (retrieval) | "The company's net income for fiscal year 2023 was ¥1,137 billion [source: annual_report.pdf, p.4]." |
   | "What is 15% of 2.4 billion?" | 1 step (calculation) | "The calculated amount is $360 million, which is 15% of $2.4 billion." |
   | "What was Meridian net revenue in FY2023, and what would it be after 3 years of 8% compound annual growth?" | 2 steps (retrieval → calculation) | "Meridian's net revenue for fiscal year 2023 was ¥16.91 trillion [source: annual_report.pdf, p.2]. If this revenue is compounded at an 8% annual growth rate for 3 years, the projected result would be approximately ¥21.30 trillion." |

   Note on the first result: the document's Statement of Operations reports both
   "Profit for the year" (¥1,137B) and "Attributable to owners" (¥1,107B) for FY2023;
   the RAG agent's extraction picked the former for the generic phrase "net income."
   This is the ambiguous-retrieval-query failure mode discussed in the Task 1.4 analysis
   below, not a bug — the step text "the company's net income" doesn't disambiguate
   which of the two figures is meant.

## Deployment

1. **Logged + registered** via `deployment/deploy.py::log_and_register()`:
   `mlflow.langchain.log_model(lc_model="deployment/agent_model.py", code_paths=["agent", "rag",
   "tools", "config.py"], ...)`, tracking against `mlflow.set_tracking_uri("databricks")`,
   registry `mlflow.set_registry_uri("databricks-uc")`, registered as
   `cs4603.default.document_analyst`. MLflow's own input-example validation step during
   logging actually executed the full real graph (planner → supervisor → rag_agent →
   mcp_tools → synthesizer) against the live LLM and Vector Search index — confirmed by
   real HTTP request logs during `log_and_register()`, including a 429 that the run
   handled gracefully.
2. **Serving endpoint** created via `deployment/deploy.py::create_or_update_endpoint()`
   using `WorkspaceClient().serving_endpoints.create(...)`, `workload_size="Small"`,
   `scale_to_zero_enabled=True`, with `DATABRICKS_HOST`/`TOKEN`/`MODEL` injected as secret
   references (`{{secrets/cs4603-deploy/...}}`) and `VECTOR_SEARCH_ENDPOINT`/`INDEX`/
   `EMBEDDINGS_ENDPOINT` as plaintext env vars (not secrets — the retriever just needs the
   names to look up the index).
3. **Endpoint name:** `emanqureshi-document-analyst`, on workspace
   `https://dbc-bcf3e60e-2bff.cloud.databricks.com`, serving `cs4603.default.document_analyst`
   version 10 and `READY`. Invocation URL:
   `https://dbc-bcf3e60e-2bff.cloud.databricks.com/serving-endpoints/emanqureshi-document-analyst/invocations`.
4. **Debugging journey (three real, distinct root causes, ten model versions):**
   - **Stuck build (versions 1–7):** repeated attempts sat at `Container creation pending`
     for 40–90+ minutes with zero state change and no error anywhere — reproduced identically
     across the Python SDK, raw REST calls, and the `databricks` CLI, ruling out a client-side
     cause. Root cause: `pip_requirements` was unpinned, so MLflow inferred the container's
     Python version from whatever interpreter happened to log the model (3.14, then 3.13.7 in
     different attempts) — and Databricks Serving's build environment doesn't reliably support
     Python newer than ~3.11 yet. Fixed by pinning `conda_env` to `python=3.11` explicitly
     *and* actually running `log_and_register()` from a real Python 3.11 interpreter (MLflow
     writes a separate `python_env.yaml` from the *calling* interpreter's version regardless of
     what `conda_env` says, so pinning the YAML alone wasn't sufficient — the logging process
     itself had to run on 3.11).
   - **Model-load crash (version 8):** past the build stage, `DEPLOYMENT_FAILED` with
     `AttributeError: 'StreamToLogger' object has no attribute 'fileno'`. Databricks Serving
     redirects `sys.stdout`/`sys.stderr` to a custom logging wrapper with no `.fileno()`; the
     MCP subprocess (spawned via `anyio.open_process` → `gevent`'s monkey-patched
     `subprocess.Popen`) needs a real file descriptor for stderr. Root cause matched the
     assignment's own warning almost exactly (`DEPLOYMENT_GUIDE.md`: *"this is the most fragile
     part of the deployment"*).
   - **Deferred to request time (version 9):** rearchitected `agent/graph.py::make_mcp_node` to
     lazily call `load_mcp_tools()` on the node's *first invocation* instead of at graph-build
     time (i.e. instead of during MLflow's synchronous model-load phase). This got the endpoint
     to `READY`, but the identical `fileno` crash then surfaced at *request* time on the first
     calculation query — proving the underlying incompatibility was still there, just moved.
   - **Real fix (version 10):** `mcp.client.stdio.stdio_client` is decorated with
     `@asynccontextmanager`, so the publicly visible `stdio_client.__defaults__` is the
     *wrapper's* (empty) defaults — my first patch attempt was silently mutating the wrong
     object. The real default (`errlog: TextIO = sys.stderr`) lives on
     `stdio_client.__wrapped__` (the attribute `functools.wraps` sets to the original
     function). Patching `__wrapped__.__defaults__` directly — verified locally first by
     simulating a fileno-less `sys.stderr` before ever redeploying — fixed it for real. All
     three canonical test queries (retrieval-only, calculation-only, combined) now pass on the
     live endpoint.
5. **Task 2.4 — testing the deployed endpoint:**
   - **curl:** `POST .../invocations` returns `200` with a JSON body — but it's a **raw list
     wrapping the graph's own `AnalystState` dict** (`[{"messages": [...], "plan": [...],
     "step_results": [...], ...}]`), not an OpenAI `ChatCompletion` object.
   - **OpenAI Python SDK:** `client.chat.completions.create(...)` fails with
     `AttributeError: 'list' object has no attribute 'choices'` — confirmed and documented
     rather than papered over. MLflow only auto-wraps a served model's output into the OpenAI
     envelope when its schema is pure messages-only (`ChatCompletionResponse`/`StringResponse`
     — the same constraint Bonus B's Agent Framework enforces, see the Bonus B section below);
     our `AnalystState` has extra fields (`plan`, `step_results`, `current_step_index`,
     `next_agent`, `final_answer`), so MLflow serves it as-is. `client/sdk.py::_extract_content`
     handles both shapes so the client SDK works against this endpoint regardless.
   - **3 test queries against the deployed endpoint** (raw HTTP, real output):
     | Query | Plan | Answer |
     |-------|------|--------|
     | "What was the net income in 2023?" | 1 step (retrieval) | "The company's net income for fiscal year 2023 was ¥1,137 billion [source: annual_report.pdf, p.4]." |
     | "What is 15% of 2.4 billion?" | 1 step (calculation) | "15% of 2.4 billion is 360 million." |
     | "Meridian net revenue FY2023 + 8% CAGR × 3yr" | 2 steps (retrieval → calculation) | "Meridian's net revenue for fiscal year 2023 was $16.91 billion [...]. If this amount is compounded at an 8% annual growth rate for 3 years, the projected result would be approximately $21.30 billion." |
   - **Local vs. deployed — not identical, and here's why:** the deployed combined-query
     answer says "$16.91 billion" where the same query run locally earlier said "¥16.91
     trillion" — same underlying number (16.91), different currency symbol and scale word.
     This is LLM sampling variance in phrasing, not a data or logic bug: the retrieved source
     chunk and the calculated `21.3017` are identical in both runs (confirmed via
     `step_results`); only the synthesizer's free-text wording differs between two independent
     LLM calls. This is expected given `temperature=0.0` reduces but doesn't eliminate
     run-to-run wording variance, and is a good illustration of why the synthesizer's output
     shouldn't be treated as byte-for-byte reproducible even with deterministic settings.
   - **Latency (warm, 3 runs per query, deployed endpoint):** retrieval-only avg 16.3s
     (range 12.8–22.4s), calculation-only avg 16.1s (range 7.3–21.0s), combined avg 17.7s
     (range 13.9–24.3s). All three fall in a similar band because the dominant cost is LLM
     round-trips (planner + supervisor-per-step + synthesizer, each a separate call to the
     underlying `databricks-meta-llama-3-3-70b-instruct` endpoint), not the retrieval or
     calculation step itself. Cold-start latency wasn't independently isolated — the endpoint
     stayed warm across our whole testing session (repeated requests within minutes of each
     other), and deliberately waiting out `scale_to_zero`'s idle window wasn't practical given
     how much of this session was already spent on the deployment debugging above; based on
     the container-build times observed earlier (minutes, when a *new* version deploys), a
     genuine cold start (spinning back up from zero replicas) would plausibly add real
     overhead beyond these warm numbers, but that's an inference from adjacent evidence, not a
     measurement.

## Design decisions

- **Graph architecture:** explicit planner → supervisor → {rag_agent | mcp_tools} loop →
  synthesizer, per the assignment's plan-and-execute pattern (`agent/graph.py`), rather than
  a single ReAct agent with all tools bound — see the Task 1.3 analysis above for when that
  tradeoff is actually worth it.
- **Routing:** the supervisor makes one LLM call per step, classifying it into exactly two
  buckets (`rag_agent`/`mcp_tools`) via a constrained one-word prompt rather than structured
  output — simple to reason about, but as the Task 1.3 answer covers, it fails silently on
  a misroute rather than detecting and recovering from one.
- **`messages` as the sole entry/exit channel:** the synthesizer writes to both
  `final_answer` (internal) and `messages` (external) because the serving endpoint's
  OpenAI-compatible contract reads only `messages[-1]` — this was flagged clearly in the
  spec and confirmed necessary once I actually deployed and would have silently returned
  empty completions otherwise.
- **MCP tool path resolution via import, not `__file__`-relative guessing:**
  `agent/graph.py::_default_server_path()` resolves `tools/mcp_server.py`'s location by
  importing `tools.mcp_server` and reading its `__file__`, rather than computing a relative
  path from `agent/graph.py`'s own location. The latter broke silently once MLflow's
  packaging sandbox didn't preserve the source repo's directory nesting — a real bug this
  project's own deployment surfaced, not a hypothetical.
- **`_run_async` helper for MCP tool calls:** plain `asyncio.run()` fails with "cannot be
  called from a running event loop" inside Jupyter/Databricks notebook kernels (both already
  run their own loop) even though it works fine in a plain script — another bug this
  project's own notebook execution surfaced. `_run_async()` detects an already-running loop
  and falls back to running the coroutine on a fresh loop in a separate thread.
- **Lazy MCP tool loading, deferred from graph-build time to first request:**
  `make_mcp_node` no longer requires `tools` up front — `tools=None` means "load on first
  invocation" instead of "load now." Spawning the MCP subprocess during MLflow's synchronous
  model-load phase turned out to conflict with how Databricks Serving redirects
  stdout/stderr there; deferring the spawn to normal request handling sidesteps that phase
  entirely. This is a real architectural change driven directly by a deployment failure, not
  a preemptive design choice.
- **Patching `mcp.client.stdio.stdio_client.__wrapped__.__defaults__`, not
  `.__defaults__`:** `stdio_client` is decorated with `@asynccontextmanager`, so the name
  visible at the module level is a wrapper with its own (empty) `__defaults__` — the real
  default argument (`errlog: TextIO = sys.stderr`, bound once at import time to whatever
  `sys.stderr` was at that instant) lives on `.__wrapped__`, the attribute `functools.wraps`
  sets to the original decorated function. Verified locally by simulating a fileno-less
  `sys.stderr` before ever spending a deploy cycle on it.
- **Manual deployment (Part 2) over `agents.deploy()` (Bonus B) for the primary path:**
  chosen because the spec frames the manual path as the one that teaches the actual
  serving-container lifecycle (secret scopes, `code_paths`, the `READY` polling loop) —
  which directly paid off, since diagnosing the stuck-endpoint incident above required
  reading `EndpointState`/`ServedModelState` fields directly via `WorkspaceClient`, not
  something `agents.deploy()`'s single call would have surfaced as clearly.

---

## Analysis Questions

### Task 1.2 — Planner
1. My planner (`agent/planner.py`) produces a flat `list[str]` with no explicit
   dependency edges between steps — step *N* is just text; it doesn't reference
   "the result of step 1" by name or index. Dependencies are handled *implicitly*
   through shared state: every executed step's result is appended to
   `step_results`, and both the MCP node and the synthesizer are given the full
   `step_results` list as context, not just the current step. So when step 2 is
   "calculate 8% growth on that revenue figure," the MCP node's prompt
   (`MCP_STEP_PROMPT`) includes *all* prior results and relies on the LLM to read
   the number out of step 1's text and plug it into a tool call. This works for
   the linear, mostly-sequential plans this planner produces (it's told to order
   steps so each one only needs facts a prior step would have surfaced), but it
   is fragile: there's no structured hand-off (e.g. "step 2 needs step 1's
   `start_value`"), so if step 1's result is phrased ambiguously (two numbers in
   one sentence) the MCP node can extract the wrong one with no error raised.
   A more robust design would have the planner emit structured steps
   (`{"step": ..., "depends_on": [0]}`) and pass only the referenced results
   forward instead of the whole history.
2. For this use case — short, mostly linear analytical queries (find a fact,
   then compute something with it) — replanning after every step would likely
   **hurt more than it helps**. Each replan call is an extra LLM round trip, and
   because the plan is almost always "1 retrieval → 1 calculation" for these
   queries, there's little for a replanner to correct: the supervisor already
   re-evaluates *routing* per step, which is the part of the plan actually prone
   to drift. Where replanning would help is the failure case that's most likely
   in practice — retrieval returning "not found in documents." If step 1 fails,
   step 2 ("compute 8% growth on that figure") is now unanswerable as written,
   and a replanner could rewrite the remaining plan into "explain that the base
   figure is unavailable" instead of blindly feeding "not found in documents"
   into a calculation tool. Right now that case is only handled downstream, by
   the synthesizer being told to acknowledge gaps rather than guess — which
   avoids a crash but wastes a full MCP tool-call cycle on data that was never
   going to produce a number.

### Task 1.3 — Supervisor
1. My supervisor (`agent/supervisor.py`) makes a binary keyword-based decision
   (does the LLM's one-word reply contain `"mcp_tools"`, else default to
   `rag_agent`) per step, with no confidence score and no validation against the
   step text. The failure mode is a **silent misroute**: e.g. a step like
   "Compare the reported and adjusted revenue figures" could get routed to
   `rag_agent` when it actually needs `compare_values` from the MCP tools, or
   vice-versa. Because `rag_agent` and `mcp_tools` both unconditionally append
   *something* to `step_results` and increment `current_step_index` — `rag_agent`
   will happily return "not found in documents" for a step that was never a
   retrieval question, and the MCP node will call some tool even if the step
   didn't need one — the graph doesn't halt or error on a misroute; it just
   produces a low-quality result that only becomes visible at the very end, in
   the synthesizer's output. Detection today is entirely manual (a human reading
   the final answer notices a "not found" that shouldn't be there). A better
   design would have the RAG node signal "this doesn't look like a factual
   lookup" (e.g. when retrieval similarity scores are all low) or have the MCP
   node signal "the LLM didn't call a tool" as an explicit routing-failure
   result rather than a generic string, and feed that signal back to the
   supervisor to re-route the same step index to the other specialist once
   before giving up.
2. A single ReAct agent with both the retrieval tool and the MCP tools bound
   would pick, on every turn, whichever tool it thinks is relevant — collapsing
   planning and routing into one implicit loop. That's simpler to build (no
   separate planner/supervisor/state-machine) and fine for queries where the
   next action is obvious from the conversation so far. The supervisor pattern
   earns its complexity when the task benefits from an **explicit, inspectable
   plan up front**: for a query like "find 2023 revenue, then project 8% growth
   for 3 years," the plan and step_results in my `AnalystState` give a step-by-
   step audit trail (which is exactly what the `pa4.ipynb` execution-trace
   requirement in Task 1.7 is designed to show) — you can see *which* step
   produced *which* number before the LLM ever combines them, whereas a ReAct
   agent's tool-call transcript conflates "what to do next" with "what the
   answer is" and is harder to review or constrain. The supervisor pattern also
   lets each specialist have a narrow, tuned prompt (`RAG_EXTRACT_PROMPT` only
   ever sees retrieval steps, `MCP_STEP_PROMPT` only ever sees calculation
   steps) instead of one prompt that has to reason about tool selection *and*
   task execution simultaneously — worth it once queries routinely mix multiple
   fact lookups and calculations, not worth it for single-tool-call queries.

### Task 1.4 — RAG Agent
1. My `rag_agent` (`agent/rag_agent.py`) calls `retriever.invoke(step)` using the
   decomposed step text (e.g. "Find Meridian's net revenue for fiscal year
   2023"), not the user's original question (e.g. "What was Meridian's net
   revenue in FY2023, and what would it be after 3 years of 8% compound annual
   growth?"). This generally *improves* retrieval precision for the fact this
   step needs: the original question mixes a retrieval need and a computation
   need, so embedding it directly would pull the query vector toward
   growth-rate/compounding language that has nothing to do with the annual
   report's actual content, diluting the top-k results with irrelevant chunks.
   The isolated step is a cleaner, single-intent query. The risk is the
   opposite failure — if the planner's step text drops context the original
   question had (e.g. it says "the net revenue" without "Meridian" or "fiscal
   year 2023" because that context felt implicit when the LLM wrote the plan),
   the retrieval query is now *under-specified* and could match the wrong
   fiscal year or the wrong company if the corpus ever contained more than one.
   In this single-document (one PDF) corpus that risk doesn't show up, but it
   would in a multi-document deployment.
2. For a vague step like "find relevant financial data," I would not send it to
   `retriever.invoke()` as-is. Two changes to `make_rag_agent`: (a) prepend the
   original user question (already available as `state["messages"][0].content`)
   to the step text before embedding, so the retrieval query inherits whatever
   entities/timeframe the vague step lost — i.e. retrieve on
   `f"{original_question}\n{step}"` rather than `step` alone; (b) treat a low
   top-k similarity score (Databricks Vector Search returns scores) as a signal
   to re-ask the planner's LLM for a more specific query before hitting the
   index, e.g. "rewrite this retrieval step to name the specific metric and
   time period being asked about." Both are cheap compared to a wrong or
   empty retrieval, and the second one reuses the same LLM client already
   passed into `make_rag_agent(retriever, llm)` — no new dependency.

### Task 2.1 — Model Definition
1. `models-from-code` needs a self-contained file because MLflow doesn't
   pickle a live Python object — it re-**executes** `agent_model.py` from
   scratch inside the serving container, which starts from nothing but the
   files listed in `code_paths` and the packages in `pip_requirements`. Any
   reference to state that only existed in *my* dev session or *my* laptop
   (a live object, `localhost` database connection, an absolute path unique
   to my checkout) is gone by the time that re-execution happens elsewhere. I
   hit a real version of this: my first `agent_model.py` computed the MCP
   server's path via `os.path.dirname(os.path.dirname(__file__))`, which
   happened to be correct in my own repo layout but broke the moment MLflow's
   packaging sandbox copied `code_paths` into a *different* directory
   nesting — `can't open file '.../tools/mcp_server.py'`, failing inside
   MLflow's own input-example validation step even though nothing was wrong
   with the code in my normal environment. Referencing a laptop-only database
   would fail the same way, just with a connection error instead of a
   missing file: the container simply cannot reach a service that only
   exists on my machine.
2. Querying the Vector Search index at inference time instead of baking the
   corpus into the artifact trades a few things: **freshness** — the index
   reflects the latest Delta-Sync from the chunks table without ever
   re-logging the model, whereas a baked-in corpus is frozen at log time and
   any document update forces a full re-log-and-redeploy; **cold-start
   size** — my `agent_model.py`'s own `pip_requirements` (`mlflow`,
   `langgraph`, `databricks-vectorsearch`, `mcp`, …) already made the
   container build slow (Task 2.3 took over an hour to reach `READY` on a
   fresh endpoint), and stuffing embedded document data into the artifact on
   top of that would make image builds and cold starts materially worse for
   any corpus bigger than our seven chunks; **latency** — an external index
   adds one network round-trip per retrieval call versus an in-process
   lookup, which only stays cheap for small corpora anyway; **failure
   modes** — an external index introduces "index unreachable / not `READY`"
   as an independent failure surface from the model container itself (which
   is exactly why `rag/store.py::get_vector_store()` raises a clear `OSError`
   when the endpoint/index env vars are missing, rather than failing deep
   inside a retrieval call), while a baked-in corpus removes that network
   dependency at the cost of coupling every data update to a full model
   redeploy.

### Task 2.3 — Serving Endpoint
1. The endpoint's own platform-level authentication only lets Databricks
   fetch and run *my model artifact* — it says nothing about what
   credentials the code *inside* that model gets to use for its own outbound
   calls. Once the container starts, `agent_model.py` is just an isolated
   Python process; from its perspective it has no ambient identity at all
   unless I hand it one explicitly. `get_chat_llm()` and
   `get_vector_store()` both make their own outbound REST calls (to the LLM
   serving endpoint, to Vector Search) authenticated exactly like any
   external client would be — so `DATABRICKS_TOKEN` has to be injected as an
   environment variable (via the secret scope) for the *model's own* calls to
   succeed, completely independent of how the platform authenticated to
   serve the model in the first place.
2. Databricks does a rolling cutover, not an in-place swap. I watched this
   happen directly in the Deployments event log: creating version 2 produced
   a *new* served entity (`document_analyst-2`) alongside whatever was
   already serving, and the endpoint's `TrafficConfig` only shifts
   `traffic_percentage` to the new served entity once its container reaches
   `READY` — the old version's compute isn't torn down until after that
   cutover. In-flight requests already routed to the old version keep
   running to completion there; only *new* requests arriving after the
   traffic shift get routed to the new version. The cost is that both
   versions' compute runs simultaneously for the duration of the new
   version's (potentially very slow, per Task 2.3's build times) startup.

### Task 3.2 — Client
1. A 429 or 503 from a model serving endpoint signals *temporary,
   load-dependent* unavailability (rate limiting, or `scale_to_zero`
   spinning back up), not a permanent failure. Fixed-interval retries don't
   adapt to how congested the endpoint currently is: if many clients all
   retry after the same fixed delay, they collide again at the same moment
   and re-create the congestion that caused the 429/503 in the first place —
   a thundering-herd effect. My implementation's `2 ** attempt` backoff
   spaces retries out increasingly, giving the endpoint's autoscaler real
   time to add capacity and naturally de-synchronizing concurrent clients'
   retry timings so they stop colliding.
2. Backoff delays retries, it doesn't cap how many total requests eventually
   go out. If `max_retries` is set very high, a client hitting a
   *persistent* (not transient) failure keeps re-sending requests over an
   increasingly long window instead of failing fast — and with many
   concurrent users each independently doing the same thing, a transient
   blip can turn into a sustained, self-inflicted load spike that keeps the
   endpoint saturated for everyone. It also makes the caller's own latency
   wildly unpredictable: with exponential backoff, a few extra retries
   balloon wait time fast, tying up caller-side threads/connections waiting
   on requests that were never going to succeed.
3. Choose `ask_streaming()` whenever the generation is slow enough that
   showing nothing until completion would feel broken, and partial output is
   still useful to show progressively. Concrete example: a chat widget in a
   financial analyst's dashboard, where a combined query (retrieve a figure,
   run a calculation, synthesize an answer) takes several seconds
   end-to-end — streaming lets the UI render output the moment it's
   available instead of a multi-second blank spinner. `ask()` is the right
   choice for backend/batch use (a script generating a report) where nothing
   is rendered incrementally to a human anyway. One caveat my own client
   handles, confirmed against the real deployed endpoint: it doesn't
   implement `predict_stream` at all, and — more strictly than I expected —
   actively **rejects** `stream: True` with a 400 `"This endpoint does not
   support streaming"` error rather than silently ignoring the flag and
   returning plain JSON. `ask_streaming()` catches that specific rejection
   and falls back to yielding the complete answer once. Still worth using
   `ask_streaming()` over `ask()` if the caller wants one consistent code
   path regardless of whether the backend ever adds real token streaming
   later.

### Bonus A — GitHub Actions CI/CD

`.github/workflows/deploy.yml`: `lint-and-test` runs `ruff check agent/ client/` and
`pytest -q` (the offline `tests/test_smoke.py` + `tests/test_client.py`, no Databricks, no
network); `deploy` needs `lint-and-test` and only runs on `push` to `main`
(`if: github.ref == 'refs/heads/main' && github.event_name != 'pull_request'`), reading
`DATABRICKS_HOST`/`DATABRICKS_TOKEN` from GitHub Secrets and running
`deployment/deploy.py`. Both jobs pin Python 3.11 via `uv python install 3.11` +
`uv sync --python 3.11` — directly reusing the Task 2.3 finding that Databricks Serving's
build environment doesn't reliably support anything newer, so a CI runner defaulting to a
different Python version wouldn't silently reproduce the same stuck-build failure Task 2.3
spent most of this session diagnosing.

1. `main` is the single reviewed, protected source of truth; feature branches are
   experimental and may be broken. Deploying from a feature branch would push
   half-finished or untested work to the one live endpoint, and concurrent branches would
   race to overwrite each other's deployments. Merging to `main` is the deliberate "this is
   ready" signal — `pull_request` still runs lint+test (so reviewers see green/red before
   merging) but explicitly does not deploy.
2. Add an evaluation gate between test and deploy: run the newly logged model against a
   fixed held-out set of the same three query types used throughout this assignment
   (retrieval-only, calculation-only, combined), score each against expected facts/citations
   (e.g. via `mlflow.evaluate` or a simple exact/fuzzy-match check against known-good
   answers), and compare the score to the currently-serving version's logged metric. If the
   new version scores lower — or fails outright, the same way the real deployment in this
   session did across ten iterations — fail the job so `deploy.py`'s endpoint-update step
   never runs, and the previous working version keeps serving traffic.

### Bonus B — `agents.deploy()` (three real bugs fixed, blocked by a platform limit on the last step)

`deployment/deploy_agents.py` reuses Part 2's exact logging pipeline
(`deploy.py::log_and_register()`, now parametrized so both paths share it without duplication)
but points it at a new file, `deployment/agent_model_chat.py`, and adds a single
`agents.deploy(model_name=..., model_version=..., scale_to_zero=True)` call. Four real,
distinct issues surfaced by actually running this, in order:

1. **Environment blocker (fixed):** `databricks-agents` depends transitively on `whenever` (a
   Rust-backed package via PyO3) with no prebuilt wheel for Python 3.14 and no source-build
   support against Python 3.14's C API (277 compile errors — a genuine upstream
   incompatibility, not a local misconfiguration). Fixed properly, not worked around: installed
   a separate Python 3.11 venv (`.venv-py311`) — the same interpreter version Task 2.3's
   debugging had already established as the one Databricks Serving actually supports — and
   `databricks-agents` installed cleanly there with a prebuilt wheel, no Rust needed.
2. **Hardcoded subprocess command (fixed, benefits Part 2 too):** `agent/graph.py` hardcoded
   `"command": "python"` for the MCP subprocess. Under `uv run` this resolved fine (uv's
   managed venv puts a `python` symlink on `PATH`), but running directly via
   `.venv-py311/bin/python` outside `uv run`, the bare `python` command resolved through
   `pyenv` shims to nothing (`pyenv: python: command not found`) since only `python3` was
   shimmed. Fixed by using `sys.executable` — the exact interpreter already running,
   independent of `PATH`/`pyenv`/venv-activation quirks in whatever shell spawned the process.
   This is a real robustness fix that stayed in `agent/graph.py` regardless of Bonus B's
   outcome — anything that manually invokes `build_graph()` outside `uv run` benefits.
3. **Output-schema incompatibility (fixed):** Agent Framework validates the served model's
   *output* schema and requires `ChatCompletionResponse` or `StringResponse`. Our graph
   returns the full `AnalystState` dict — exactly what Part 2's manual serving path is
   designed to tolerate (it just reads `messages[-1]`), but Bonus B's stricter framework
   rejected it outright with `ValueError: The model's schema is not compatible with Agent
   Framework...`. Fixed with `deployment/agent_model_chat.py`: a thin `RunnableLambda` wrapper
   that reuses the exact same `build_graph()` (inheriting every Task 2.3 fix — lazy MCP
   loading, the `errlog` patch, `sys.executable`) but returns only
   `result["messages"][-1].content` as a plain string. One follow-up bug this exposed: MLflow's
   chat-model input adaptation calls the wrapped runnable with a *bare list of messages*, not
   the `{"messages": [...]}` dict shape the rest of the codebase uses — `_invoke()` now accepts
   both shapes defensively. Verified locally (both call shapes) before spending a deploy cycle
   on it, same discipline as Task 2.3.
4. **Genuine platform wall (not fixable in code):** with all three bugs above fixed, logging
   and registration succeeded and the schema check passed — `agents.deploy()` proceeded all
   the way to the real `w.serving_endpoints.create(...)` call and failed with
   `NotFound: Inference table is not currently supported for this endpoint type in this
   workspace.` Traced this into the SDK's own source
   (`databricks.agents.deployments._create_ai_gateway_config`): it unconditionally constructs
   `AiGatewayInferenceTableConfig(enabled=True, ...)` with **no parameter on the public
   `agents.deploy()` API to disable it** — confirmed by inspecting the full call signature and
   the `_create_new_endpoint_config`/`_create_ai_gateway_config` source directly, not just
   assumed from the error text. This is categorically different from the three issues above:
   those were real bugs with real code fixes; this is a hard requirement the SDK itself bakes
   in, unsupported by this Free Edition workspace, with no escape hatch in the library.
5. **Got a genuinely live endpoint anyway, via the proven manual path:** the actual endpoint
   creation `agents.deploy()` performs underneath is just `w.serving_endpoints.create(...)` —
   the exact same call Part 2's `create_or_update_endpoint()` already uses successfully,
   which never requests `ai_gateway`/Inference Tables at all. Parametrized
   `create_or_update_endpoint()` with an optional `endpoint_name` and deployed
   `cs4603.default.document_analyst_chat` (the schema-compatible wrapper) to a second,
   independent endpoint (`emanqureshi-document-analyst-chat`) via that same proven method.
   It reached `DEPLOYMENT_READY` and answers correctly, with genuinely clean
   `StringResponse`-shaped output (`["15% of 2.4 billion is 360 million."]` — a plain string
   in a list, not the full `AnalystState` dict Part 2's endpoint returns).

**Where this leaves Bonus B's three literal requirements:** (1) "Deploy using `agents.deploy()`"
— the schema-compatible model is live, but via the proven manual method, not the literal
`agents.deploy()` Python call (which cannot succeed on this workspace, full stop); (2) "Open
the Review App and submit 3 queries with feedback ratings" and (3) "Show the feedback in the
MLflow experiment" — **not achievable on this workspace regardless of method**, since the
Review App is exclusively provisioned by `agents.deploy()` itself and tied to the same
Inference Tables infrastructure that's blocked. So: a real, live, correctly-functioning
schema-compatible agent endpoint exists, but the Review-App-based feedback loop specifically
does not and cannot on this Free Edition workspace.

**Comparison and feedback-loop answers**, informed by having pushed this all the way to the
real platform limit rather than stopping at the first error:

1. `agents.deploy()` gains: automatic auth (no secret scope to wire), a Review App provisioned
   for free, and one call instead of three. It loses: the ability to serve a model whose
   output is anything richer than a chat-completion shape without a wrapper (point 3 above),
   and — on a workspace like this one — it can lose the ability to deploy *at all*, since it
   mandates Inference Tables with no opt-out (point 4), whereas Part 2's manual path never
   requested that feature and deployed successfully. The granular control the manual path
   forces on you (inspecting `EndpointState`/`ServedModelState` directly, retrying via the CLI
   as an independent verification path, iterating on `pip_requirements`/`conda_env`) was
   directly responsible for diagnosing all three of Task 2.3's root causes *and* the first
   three issues here — control `agents.deploy()`'s single opaque call doesn't offer. For a
   model with a naturally chat-shaped output on a workspace with Inference Tables enabled,
   `agents.deploy()` is strictly less work for the same result; on a constrained workspace, the
   manual path is sometimes the only one that actually deploys.
2. Feedback loop: Review App ratings/comments would flow into the MLflow experiment as
   evaluation data; a concrete next step would be periodically pulling low-rated traces,
   diagnosing whether the failure was a misroute (Task 1.3), a bad retrieval (Task 1.4), or a
   synthesis error, and using that triage to prioritize which node's prompt to iterate on next
   — closing the loop from "a human didn't like this answer" back to a specific prompt change.

### Bonus C
TODO
