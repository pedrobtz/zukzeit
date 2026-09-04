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

The engine contract above is settled by Chronos-2. The *caller* surface is a
separate decision, and deliberately not frozen here.

R users do not think in rows and group ids; they think in a panel with a key
column, an index, a target, and further columns that are covariates. The
translation from that shape into grouped rows belongs in `forecast()` and the
adapters, and is the part the roadmap flags as possibly unable to preserve
batching. It is specified once the engine contract is shipped and exercised.

## Open decisions

1. Whether `groups` is a plain list or a constructed `zuk_groups()` object with
   validation attached. A constructor is more defensible at the boundary; a
   plain list is lighter for architecture authors to build in tests.
2. Whether the return should carry the target rows' `id` values as names, so a
   caller can map results back without re-deriving the mapping.
3. Whether contract v1 handles should report `contract_version` 1.0.0 forever
   or be re-stamped 2.0.0 once the engine implements v2. They should not: the
   version records what the *architecture* was written against.
