# The Northwind Logistics ticket dataset

A synthetic support-ticket corpus built for this workshop.

| File | Rows | Purpose |
|------|------|---------|
| `tickets_train.jsonl` | 281 | Training split |
| `tickets_eval.jsonl` | 71 | Held-out evaluation split |

## Schema

Each line is a JSON object:

```json
{
  "ticket": "Blocking my whole team — VPN will not connect from home for anyone on Fleet Planning since the maintenance window. Please advise.",
  "label": "category: network-connectivity | priority: P2 | queue: network-ops",
  "category": "network-connectivity",
  "priority": "P2",
  "queue": "network-ops"
}
```

`label` is the exact string the model must learn to produce. The three separate
fields are there so the scoring code in `common/triage.py` can report accuracy
per component rather than one all-or-nothing number.

## Label space

**Categories (8):** `access-management`, `billing-inquiry`, `data-request`,
`hardware-failure`, `network-connectivity`, `onboarding-request`,
`security-incident`, `software-bug`

**Priorities (4):** `P1` `P2` `P3` `P4`

**Queues (6):** `app-support`, `endpoint-ops`, `finance-ops`, `identity-team`,
`network-ops`, `security-ops`

`queue` is a deterministic function of `category` — two categories route to
`identity-team`, two to `app-support`, the rest one-to-one. `priority` is
signalled by the severity language at the start of the ticket, independently of
category, so the model has to read both parts of the text.

## Honest disclosure

This data is **synthetic and template-generated**, and it is deliberately more
learnable than production ticket data. That is a workshop design decision:
it means an 82M model on 4 CPU cores shows a clear, reliable before/after
improvement in under three minutes, which is what makes the labs teachable.

Real ticket corpora are messier — inconsistent labels, ambiguous categories,
heavy class imbalance, and multi-issue tickets. Expect lower absolute scores
and a much larger payoff from data curation than from hyperparameter tuning.
The workflow you learn here transfers directly; the numbers do not.

Names, companies, sites, and applications are fictional. Any resemblance to
real people or organizations is coincidental.

## Regenerating or extending

To add your own examples, append JSONL rows in the schema above. Keep the
`label` string format exactly as shown — `common/triage.py` parses it with a
strict regex, and a formatting drift will silently score as invalid.

---

© 2026 Tech Skills Transformations and Brent C. Laster. All rights reserved.
