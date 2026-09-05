# Contract v2 — grouped targets and covariates

*Drafted: 2026-09-04. Status: proposed, not yet shipped.*

Contract v2 exists to carry Chronos-2's inputs. It is designed against that
model's actual mechanism, recorded in [`chronos2-feasibility.md`](./chronos2-feasibility.md),
rather than against an abstract notion of covariate support.

## The shape of the problem

Chronos-2 has one input mechanism, not three. Every series is a row; `group_ids`
says which rows form a task; a row's role is expressed by whether future values
are supplied for it. There are no separate multivariate, past-covariate, and
future-covariate channels to model.

Contract v1's batch entry point already passes a *list*:

```
predict_batch_fn(contexts, horizons, quantile_levels, device) -> list(matrix)
```

So v2 is additive. It says what else those rows mean.

## The v2 forward pass

```
predict_batch_fn(contexts, horizons, quantile_levels, device, groups) -> list(matrix)
```

`groups` is `NULL` for a contract-v1 request and otherwise a record with one
entry per element of `contexts`:

| Field | Type | Meaning |
|---|---|---|
| `id` | integer, length `n` | Task membership. Rows sharing an id exchange information; rows in different tasks do not. Values are labels, not indices. |
| `target` | logical, length `n` | Whether this row is forecast and returned. |
| `future` | list, length `n` | Known future values for the row, or `NULL`. Length must equal that row's horizon. |

A row is therefore one of three things, and the two fields express all three
without a separate role vocabulary:

- **target** — `target = TRUE`, `future = NULL`
- **past-only covariate** — `target = FALSE`, `future = NULL`
- **future-known covariate** — `target = FALSE`, `future = <values>`

That maps one-to-one onto the upstream NaN convention, so the port carries no
translation layer that could drift.

### Return alignment

`zuk_run_batches()` returns one matrix per **target** row, in the order those
rows appear in `contexts`. Under v1 every row is a target, so the existing
"aligned to and the same length as `contexts`" guarantee is the `groups = NULL`
special case of this rule, unchanged.

## How v1 architectures stay untouched

`new_zuk_model()` already stamps `contract_version` on every handle. The engine
dispatches on its major component:

- major 1 — `predict_batch_fn` is called with four arguments, exactly as today.
  No architecture written against v1 sees a new argument, gains a formal, or
  needs a dummy value.
- major 2 — called with five.

This is the backward compatibility `consumer-api.md` promises, and it is
enforced by the engine rather than by convention: a v1 architecture *cannot*
receive `groups`, so it cannot silently ignore one.

`predict_fn` is unchanged in both versions. It forecasts a single univariate
series and remains the fallback path.

## Capabilities

v2 unlocks three flags that contract v1 forces to `FALSE`:

| Flag | Meaning under v2 |
|---|---|
| `multivariate` | A task may contain more than one target row. |
| `past_covariates` | A task may contain covariate rows with no future values. |
| `future_covariates` | A task may contain covariate rows with future values. |

`static_covariates` and `samples` stay `FALSE`: neither has an execution
channel, and declaring them would be the "capability metadata must describe
executable behaviour" rule broken.

An architecture declares only what its fixtures demonstrate. Chronos-2 is
expected to declare all three; TimesFM and Toto declare none and remain v1.

## Pre-flight validation

The batch boundary rejects, before any tensor work, with the existing typed
conditions:

| Condition | Class |
|---|---|
| `groups` supplied to a v1 architecture | `zuk_error_contract` |
| more than one target in a task, `multivariate` false | `zuk_error_capability` |
| covariate row present, `past_covariates` false | `zuk_error_capability` |
| `future` supplied, `future_covariates` false | `zuk_error_capability` |
| `length(future[[i]])` not equal to that row's horizon | `zuk_error_capability` |
| a task with no target row | `zuk_error_capability` |
| `id`, `target`, `future` lengths differ from `contexts` | `zuk_error_capability` |

Horizons remain per row. Every row in a task must agree on its horizon, since
the future window is shared; disagreement is a `zuk_error_capability`.

## Missingness and alignment

- Contexts keep the v1 rule: `NA` removed, oldest first, truncated to
  `max_context`.
- Covariate rows are truncated on the same rule, so a task's rows may differ in
  observed length. Padding and validity masking belong to the architecture, not
  the engine, exactly as they do for a mixed-length v1 batch.
- `future[[i]]` is dense and exactly `horizons[[i]]` long. A caller with gappy
  future covariates fills them; the engine does not impute.

## The user-facing surface

The engine contract above is settled by Chronos-2. The *caller* surface should
not be invented, because R has already settled most of it. What follows is a
survey of how established packages answer the same three questions.

| Package | Declares a covariate | Future values arrive as | Multivariate | Series key |
|---|---|---|---|---|
| `fable` / `fabletools` | formula RHS, `xreg` special | `forecast(new_data = )` | `vars()` on the LHS | tsibble `key`, evaluated one at a time |
| `recipes` / tidymodels | `update_role(col, new_role = )`, arbitrary strings | `new_data` forged through the blueprint | multiple outcomes | a role, conventionally `"ID"` |
| `modeltime` | recipe roles | `modeltime_forecast(new_data = )` | via the recipe | via the recipe |
| `nixtlar` (TimeGPT client) | any extra column of the long panel; `hist_exog_list` names the past-only ones | `X_df`, a separate future frame | --- | `id_col` |
| `prophet` | `add_regressor(m, name)` before fitting | the future frame | --- | --- |

Verified locally for `fable` (`ARIMA(y ~ x)` with `forecast(new_data = )`
round-trips) and `recipes` (`update_role()` accepts arbitrary role strings,
selected with `has_role()`); read from source for the rest.

Three things are consistent across all five:

1. **Future values always arrive as a separate future frame.** `new_data`,
   `X_df`, the prophet future frame --- the same idea every time.
2. **A column's role is declared once, by name, at definition time**, not
   restructured into rows by the caller.
3. **Nobody exposes a rows-plus-group-ids API.** That representation is
   internal everywhere it exists, which is where it belongs here too: the
   `groups` record above is what the engine hands an architecture, never what a
   user builds.

`nixtlar` is the most instructive, being the only one of the five wrapping a
foundation model that supports both covariate kinds --- the same position
`zukzeit` is in. Its answer is the minimal one: extra panel columns are
covariates, `hist_exog_list` names those without a future, and everything else
is expected in `X_df`. `fable` and `prophet` do not make the distinction at all
because their models only accept future-known regressors.

### What this implies

Adopt each ecosystem's own idiom rather than a single new vocabulary:

- **plain R `forecast()`** --- extra columns are covariates, a future frame
  carries their future values, and one argument names the past-only ones. This
  is `nixtlar`'s shape, and it is the route that keeps batching.
- **`TSFM()`** --- `TSFM(y ~ xreg(promo))` with `forecast(fits, new_data = )`,
  fable's own convention, plus a special for past-only covariates.
- **tidymodels** --- recipe roles, which already express exactly this
  distinction with no new machinery.

### Recommendations

Each is checked against the machinery rather than proposed from the survey
alone.

**1. Plain R `forecast()` — adopt `nixtlar`'s shape, with covariates explicit.**

```r
forecast(model, new_data, h, quantile_levels, index, key, target,
         covariates = NULL,   # column names to condition on
         future     = NULL,   # frame of their future values (nixtlar's X_df)
         group      = NULL,   # column defining the task
         batch_size, device)
```

The one deviation from `nixtlar` is that covariates are **named rather than
inferred from leftover columns**, and it is forced: `panel_spec()` already
derives an unspecified `target` as the *first measured variable*, so treating
extra columns as covariates would silently change which column gets forecast.
`nixtlar` can infer because `target_col` is always named there.

Past-only versus future-known is then inferred rather than declared: a covariate
that appears in `future` is future-known, one that does not is past-only. That
drops `nixtlar`'s `hist_exog_list` for one fewer argument at equal
expressiveness.

**2. `TSFM()` — the tidyverts machinery already carries this.**

```r
TSFM(y ~ xreg(promo) + past_xreg(traffic), model_id = "...")
fabletools::forecast(fits, new_data = future)
```

`fabletools:::forecast.mdl_ts` re-parses the model's right-hand side against
`new_data` and passes the result to the model's `forecast` method. Verified
directly: a special declared on a model class returns the *training* values at
fit time and the *`new_data`* values at forecast time. `forecast.model_tsfm()`
already takes a `specials` argument and currently ignores it, so future-known
covariates need a special declared and consumed, not new plumbing.

`xreg()` keeps fable's exact meaning. `past_xreg()` is an extension with no
tidyverts precedent, because no tidyverts model distinguishes the two kinds ---
fable's `xreg` is always future-known. The name is deliberately unlike fable's
so it does not read as something fable defines.

This route gives covariates but **not** cross-key learning: `fabletools::model()`
evaluates one key at a time. That limitation should be stated in `?TSFM`
alongside the batching note already there.

**3. tidymodels — mirror the plain-R arguments; do not adopt recipe roles yet.**

Roles are the right long-term idiom and express this distinction exactly. But
consuming them means re-introducing the recipe and blueprint machinery
deliberately removed from `zuk_fit()`, which bought nothing when it was there.
Passing `covariates` and `future` through `set_engine()` matches the plain-R
route and adds no dependency. Recipe roles become worth it when a consumer
actually drives `zukzeit` from a recipe.

### The gap: cross-learning across keys

`fabletools::model()` evaluates one key at a time, and no tidyverts model
shares information between keys, so the tidyverts has no convention for it.
`nixtlar` sidesteps the question: the whole long panel is sent, and
cross-learning is implicit in what the caller chose to include.

Cross-series learning is one of Chronos-2's headline capabilities, so it can
only be expressed through the batched `forecast(model, panel, h)` route.

**4. Recommendation: `group = NULL` means one task per key; cross-learning is
opt-in.**

This reverses the earlier draft, which followed `nixtlar` in making the whole
panel one task. That is safe for `nixtlar` because it is their only mode. It is
not safe here, for two reasons:

- The same panel would return different numbers depending on which model is
  loaded, since a contract-1.0 model cannot cross-learn and a 1.1 one would.
- Adding an unrelated series to a panel would change the forecasts of every
  other series in it. Forecasts should be a function of the series and the
  model, not of what else happened to be in the call.

So `group = NULL` preserves exactly today's behaviour and is identical for
contract-1.0 models, while `group = "region"` opts in to tasks spanning several
keys. Opt-in costs one argument; the alternative costs reproducibility.

## Open decisions

1. Whether `groups` is a plain list or a constructed `zuk_groups()` object with
   validation attached. A constructor is more defensible at the boundary; a
   plain list is lighter for architecture authors to build in tests. Note the
   survey above: this record is engine-internal either way, so the cost falls
   on architecture authors rather than users.
2. Whether the return should carry the target rows' `id` values as names, so a
   caller can map results back without re-deriving the mapping.
3. Whether contract v1 handles should report `contract_version` 1.0.0 forever
   or be re-stamped 2.0.0 once the engine implements v2. They should not: the
   version records what the *architecture* was written against.


## Closing `consumer-api.md`

The consumer contract documents five release gates, all written against the
univariate surface. Grouped inputs belong there as a sixth, but it should be
amended **after** the caller surface lands rather than now: R1--R5 describe what
a consumer calls, and until `forecast()` accepts `covariates`, `future`, and
`group`, an R6 would document an API that does not exist.

What it will need to say, once it does:

- which capability flags a consumer inspects to learn that a checkpoint accepts
  covariates, and that they are per-checkpoint rather than per-package;
- that grouped requests are refused before download or tensor work, with the
  same typed conditions as every other capability failure;
- that a contract-1.0 checkpoint remains callable exactly as today, so a
  consumer written against the univariate surface needs no change;
- that cross-key learning is opt-in, so a consumer that does not ask for it gets
  reproducible per-series forecasts.
