---
name: mmm-channel-semantics
description: |
  How each type of marketing channel should be represented in an MMM — its role (paid media, reach-and-frequency, organic, non-media treatment, control), whether it is driven by spend or exposure, its carryover and saturation behaviour, and whether a ROAS is even defined for it. Use when deciding what a column means, choosing per-channel adstock/saturation, grouping or splitting channels, handling organic/owned/email/price/distribution variables, deciding what enters the budget optimiser, or diagnosing an implausible channel result.
---

# Channel Semantics: what each variable *is*, and how the model should treat it

Most MMM errors that survive review are not statistical. They are category errors: a
price index modelled as a media channel, an email programme reported with a ROAS,
brand search credited with the demand that TV created. The model cannot detect any of
these, because each one is a perfectly well-formed regression.

The decision below is upstream of every prior, transformation and report.

---

## 1. Five roles, and what follows from each

| Role | Adstock | Saturation | Has spend | ROAS defined | In budget optimiser | Coefficient sign |
|---|---|---|---|---|---|---|
| **Paid media** | yes | yes | yes | yes | yes | non-negative |
| **Paid reach & frequency** | yes | yes (on reach) | yes | yes | yes, plus frequency | non-negative |
| **Organic media** | yes | yes | **no** | **no** | no | non-negative |
| **Non-media treatment** | **no** | **no** | no | no | **no** | sign-constrained by theory |
| **Control** | no | no | no | no | no | unconstrained |

The role determines what the number *means*, and every framework has explicit slots for
these: `role` in this plugin's spec, `organic_media` / `non_media_treatments` / `controls`
in Meridian, `organic_vars` / `context_vars` in Robyn. In pymc-marketing organic media
goes in `channel_columns` (it needs the transformations) while non-media treatments go in
`control_columns` — with the consequence that you must remember not to quote a ROAS for
the organic ones, because the library will happily compute one.

### Paid media
Money buys exposure, exposure drives response, response decays. This is the only role for
which "ROAS" is a well-posed question, and the only one that belongs in a budget optimiser.

### Paid reach & frequency
Video and TV are bought as reach × frequency, and the two do different things: reach adds
new people, frequency adds repetition to people already reached. Frequency saturates far
faster than reach and eventually turns negative through wear-out. Modelling only spend
conflates a CPM change with a delivery change and makes optimal-frequency questions
unanswerable. Meridian models RF natively (`reach`, `frequency`, `rf_spend`, and an
`optimal_freq()` analysis); pymc-marketing has no RF likelihood — put reach in
`channel_columns` and frequency in `control_columns`, and say in the report that the
frequency recommendation is out of scope.

### Organic media
Email, organic social, PR, SEO, owned content, app push. These have carryover and
saturation like paid media, so they get the transformations — but no cost, so **ROAS is
undefined**. Reporting "email ROAS = 40" is not an impressive result, it is a division by
a number that does not exist. Report incremental units per thousand sends, or incremental
units per post.

They also carry a specific trap: organic activity is often *caused by* paid activity.
Organic search rises when TV runs. Modelling organic search as an independent channel lets
the model pay twice for the same work — once to TV, once to the organic response TV
produced. Prefer a mediator (a funnel model) or a control.

### Non-media treatment
Price, discount depth, promotion flags, distribution/ACV, store count, stock availability,
product launches, new pack sizes. They move sales directly, they correlate with media
plans, and they have no carryover in the media sense — a price cut does not "decay", it
ends.

Omitting price is the single most common cause of overstated media ROAS in retail and CPG.
Promotions and media are planned together, so the promotion's effect goes to whatever ran
alongside it. If you have exactly one hour to improve a retail MMM, add price.

Sign constraints matter here. Price should be negative (or an elasticity, in logs);
distribution should be positive. Robyn has `context_signs`; Meridian's
`non_media_treatments_prior_type="contribution"` puts the prior on the contribution;
pymc-marketing needs a prior with the mass on the correct side.

### Control
Weather, macro indicators, competitor spend, category volume, calendar. We adjust for
them; we never attribute to them and never optimise them. A control's job is to stop its
effect being credited to media, not to be interesting itself.

---

## 2. Spend or exposure: what should drive the transformation?

| Model input | Right when | Cost of getting it wrong |
|---|---|---|
| **Spend** | Cost per unit is stable; no exposure data | CPM inflation reads as more media; a price rise looks like a delivery increase |
| **Impressions / GRPs / clicks** | Delivery is measured and cost varies | None, provided spend is kept alongside for ROAS |
| **Reach** | Broad-reach video/TV with RF data | Frequency effects invisible |

Prefer exposure, keep spend. In pymc-marketing, pass exposure as the channel column and
supply `cost_per_unit` so ROAS and the optimiser stay in currency. In Meridian, `media` is
exposure and `media_spend` is cost — the split is built in.

**Audit the implied cost.** Divide spend by exposure per period. Real auction prices move;
they do not move tenfold between the 5th and 95th percentile. A wild CPM range means spend
and delivery were joined on mismatched dates or a partially populated feed — a data bug
that a model will faithfully convert into a channel effect.

---

## 3. Behaviour by channel type

Half-lives are for weekly data; multiply by 7 for daily, divide by ~4.3 for monthly.
`l_max` should keep ~95% of the impulse: `ceil(log(0.05)/log(alpha))` where
`alpha = 0.5**(1/halflife)`.

| Channel | Half-life (wk) | Saturation | Adstock | The thing that actually matters |
|---|---|---|---|---|
| Brand search | 0.3 | very fast | geometric | Harvests demand others created. High measured ROAS, low incremental ROAS. Only a holdout settles it. |
| Generic search | 0.5 | fast | geometric | Captures existing intent. Auction prices rise steeply with impression share. |
| Shopping / PMax | 0.4 | fast | geometric | Feed- and price-driven. Control for price or the ROAS is a price effect. |
| Retail media | 0.5 | fast | geometric | Sits next to the purchase; confounded with the retailer's own promotions. |
| Social prospecting | 2.0 | medium | geometric | Genuinely incremental reach; longer tail than retargeting. |
| Social retargeting | 0.4 | fast | geometric | Targets people who already engaged. Structurally over-credited. |
| Meta (FB+IG) | 1.5 | medium | geometric | Combine FB and IG unless bought and flighted separately. |
| TikTok | 1.5 | medium | geometric | Fast creative wear-out; widen priors if spend is flat. |
| YouTube | 2.5 | slow | geometric | Reach-and-frequency channel; use reach if you have it. |
| CTV / OLV | 3.0 | slow | geometric | TV-like carryover, digital-like targeting. |
| Linear TV | 3.5 | slow | **delayed** | Longest tail, delayed peak. Prefer GRPs over spend. |
| Cinema | 3.0 | slow | **delayed** | Low frequency, high attention. |
| Display / programmatic | 1.5 | medium | geometric | Use viewable impressions; raw impressions include fraud and unviewed inventory. |
| Native | 1.0 | medium | geometric | Click-driven, low intent. |
| Digital audio / podcast | 2.0 | slow | geometric | Partial attention, repeated exposure. |
| Radio | 2.5 | slow | **delayed** | High frequency, low per-exposure impact. |
| OOH | 4.0 | slow | **delayed** | Two-week buying cycles create step-shaped spend easily confused with seasonality. |
| Print | 3.0 | slow | **delayed** | Publication date ≠ readership date. |
| Direct mail | 2.0 | medium | **delayed** | Delivery lags the send by days to weeks; the delayed peak is real. |
| Affiliate | 0.3 | fast | geometric | Paid on last click. Largely cannibalised organic and brand search. |
| Influencer | 1.5 | medium | geometric | Lumpy and often mis-dated — align to post dates, not invoice dates. |
| Sponsorship | 4.0 | slow | **delayed** | Accrued spend is flat and unrelated to exposure. Model event dates or broadcast reach. |
| Email / CRM | 0.5 | fast | geometric | Organic: sends, not spend. No ROAS. |
| Organic social | 1.5 | medium | geometric | Organic; correlated with paid delivery on the same platform. |
| Organic search / SEO | 2.0 | slow | geometric | Endogenous with paid search and brand strength. Usually a mediator. |
| Price / promotion | — | — | **none** | Non-media treatment, negative sign (or positive for discount depth). |
| Distribution / ACV | — | — | **none** | Non-media treatment, positive sign. |

Two patterns underlie the whole table:

* **Carryover tracks how the ad is consumed.** A click converts now; a TV spot is
  remembered. Anything bought for attention has a long tail; anything bought for intent
  does not.
* **Saturation tracks the size of the addressable audience.** Brand search saturates
  almost immediately — there are only so many people typing your name. National video
  barely saturates at all within realistic budgets.

---

## 4. Splitting and grouping

**Split when the halves behave differently and are flighted differently.**
Brand vs generic search is the highest-value split in most accounts: they have opposite
incrementality profiles, and blending them produces a number that describes neither.
Prospecting vs retargeting is the second.

**Group when the parts always move together.** If Facebook and Instagram spend correlate
at 0.95, no model can separate them; whatever split it reports is the prior's opinion
dressed as a finding. Group them and report one honest number instead of two fabricated
ones.

The test is not organisational, it is statistical: check the pairwise correlation and the
VIF. Above r ≈ 0.8 or VIF ≈ 10, you are asking the model a question the data cannot answer.

**Cost of splitting**: each channel adds ~3 parameters (decay, saturation steepness,
scale) with no new observations. With 104 weeks and 10 channels you are at roughly 3
observations per parameter, and the priors are doing most of the work. Count the budget
before adding the eleventh channel.

---

## 5. Always-on channels

A channel that never goes dark and never changes budget carries almost no information.
Its coefficient is set by the prior, and the saturation curve near zero is pure
extrapolation — yet that is exactly the region a budget optimiser will explore when it
considers cutting the channel.

What to do, in order of preference:

1. Run a holdout test. It is the only way to create the variation the history lacks.
2. Use geo variation — the same channel at different intensities across markets is
   variation, even when the national total is flat.
3. Widen the prior and say so in the report: this is an assumption, not an estimate.
4. Refuse to quote marginal ROAS below the observed spend range.

Brand search, always-on generic search, and long-running affiliate programmes are the
usual suspects.

---

## 6. Funnel position and double counting

Upper-funnel media creates demand; lower-funnel media captures it. When both are in the
same model, the lower-funnel channel gets credit for the demand the upper-funnel channel
created, because it is closer to the conversion in time.

Symptoms: brand search with an implausibly high ROAS; TV with a suspiciously small effect;
retargeting outperforming prospecting.

Three treatments:

* **Mediator/funnel model.** Model search volume or site visits as an intermediate
  outcome, then model conversions given the intermediate. A channel's total effect is its
  direct effect plus its effect through the mediator. pymc-marketing 1.x supports
  funnel-aware structures; this is also what a DAG with `treatment_nodes` /
  `outcome_node` encodes.
* **Control, not channel.** Include search volume as a control in the conversion model so
  brand search is credited only with what it adds beyond existing demand.
* **Experiment.** A brand-search holdout measures the real number directly, and is
  usually cheap because pausing brand search for two weeks in a few geos costs little.

Never fix this by deleting the lower-funnel channel. That moves its effect somewhere else
rather than to where it belongs.

---

## 7. Deciding a column's role — the questions to ask

1. **Did we pay a media owner for it?** No → organic or non-media treatment.
2. **Does more of it mean more exposure to advertising?** No → non-media treatment or
   control.
3. **Can we choose how much of it to buy next quarter?** No → control, and keep it out of
   the optimiser.
4. **Would its effect persist after we stopped?** No → no adstock.
5. **Does it have diminishing returns within the observed range?** No → no saturation, or
   a near-linear one.
6. **Is it caused by the target, or by another channel?** Yes → it is a mediator or a
   collider, not a channel. See the causal-design skill.

Question 6 catches the worst cases: website sessions, app installs, and "leads" are
usually consequences of marketing rather than inputs to it, and putting them on the
right-hand side lets the model explain sales with sales.

---

## 8. Red flags in fitted results, and what each usually means

| Result | Usual cause |
|---|---|
| Brand search ROAS is the highest in the model | Upper-funnel credit leaking downstream |
| A channel has a negative contribution | Spend dated by invoice not delivery; or collinearity with an opposite-signed variable |
| One channel takes >70% of media effect (with ≥3 channels) | Collinearity — check VIF and the spend correlation matrix |
| Email or organic social has a "ROAS" | Category error: no spend, no ROAS |
| Retargeting beats prospecting | Selection effect, not media effect. Validate with a holdout |
| A long-carryover channel looks weak | `l_max` too short, or the optimisation window is narrower than the carryover |
| An always-on channel has a tight, confident interval | The prior is confident; the data is not |
| Media explains >60% of an established brand's sales | The baseline is being starved — check the intercept and missing drivers |

---

## Applying this in the spec

```yaml
channels:
  - column: brand_search_spend
    channel_type: sem_brand
    role: paid_media
    funnel_stage: lower
    always_on: true
    halflife_periods: 0.3
    notes: "Demand harvesting. Holdout scheduled Q2 — treat the modelled ROAS as an upper bound."

  - column: tv_grps
    channel_type: tv
    role: paid_media
    spend_column: tv_spend        # exposure drives the model, spend gives the ROAS
    exposure_column: tv_grps
    funnel_stage: upper
    adstock: delayed
    halflife_periods: 3.5

  - column: email_sends
    channel_type: email
    role: organic_media           # no spend -> the reports will not quote a ROAS

  - column: price_index
    channel_type: price
    role: non_media_treatment     # becomes a control; never adstocked, never optimised
    expected_sign: negative
```

`agent_mmm.utils.channel_classifier.classify_channel()` proposes a type from the column
name; `get_channel_type()` returns its half-life, funnel stage and default role. It is a
starting point — column names lie, and confirming each assignment with the client is part
of intake.
