"""Channel taxonomy: map column names to a channel type with modelling semantics.

The taxonomy is deliberately finer-grained than "digital vs offline" because the
distinctions that matter for an MMM are behavioural, not organisational:

* **Carryover** is a property of how the ad is consumed. A brand-search click
  converts the same week; a TV spot is still working three weeks later.
* **Saturation** is a property of the addressable audience. Brand search
  saturates almost immediately (there are only so many people searching your
  name); broad-reach video saturates slowly.
* **Identifiability** is a property of how the channel is bought. An always-on
  channel with flat spend cannot be identified from observational data at all,
  no matter which framework you use.

Classification is a *starting point*. The intake questionnaire should confirm
every assignment with the client, because column names lie.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChannelType:
    """Modelling semantics for one channel type."""

    key: str
    label: str
    family: str
    funnel_stage: str
    default_role: str = "paid_media"
    typical_halflife_weeks: float = 1.0
    """Weeks for the carryover effect to fall to half. Converted to an adstock
    alpha via alpha = 0.5 ** (1 / halflife)."""
    saturation_speed: str = "medium"
    """fast = saturates at low spend, slow = long linear region."""
    usually_always_on: bool = False
    default_adstock: str = "geometric"
    """geometric for immediate-onset channels; delayed for channels whose peak
    effect lands after the exposure (TV, cinema, print)."""
    notes: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)


# Ordered most-specific first: the first alias hit wins.
CHANNEL_TYPES: tuple[ChannelType, ...] = (
    # ---- Search --------------------------------------------------------- #
    ChannelType(
        key="sem_brand", label="Brand Search", family="search", funnel_stage="lower",
        typical_halflife_weeks=0.3, saturation_speed="fast", usually_always_on=True,
        notes=(
            "Almost entirely demand harvesting. Brand search volume is driven by "
            "upper-funnel activity, so its measured ROAS is largely borrowed credit. "
            "Model it, but treat a high brand-search ROAS as a red flag rather than "
            "a result: it usually means an upper-funnel channel is under-credited. "
            "A brand-search holdout test is the only reliable way to size it."
        ),
        aliases=("brand_search", "sem_brand", "search_brand", "brandsearch", "ppc_brand"),
    ),
    ChannelType(
        key="sem_generic", label="Generic/Non-brand Search", family="search", funnel_stage="lower",
        typical_halflife_weeks=0.5, saturation_speed="fast",
        notes="Captures existing intent. Short carryover, saturates quickly as auction "
              "competition rises. Include search-volume as a control, never as a channel.",
        aliases=("generic_search", "nonbrand", "non_brand", "sem_generic", "search_generic"),
    ),
    ChannelType(
        key="sem", label="Paid Search", family="search", funnel_stage="lower",
        typical_halflife_weeks=0.5, saturation_speed="fast",
        notes="Split brand vs generic whenever the data allows — they behave nothing alike.",
        aliases=("sem", "paid_search", "adwords", "google_ads", "googleads", "ppc", "cpc", "bing", "search"),
    ),
    ChannelType(
        key="shopping", label="Shopping / PLA", family="search", funnel_stage="lower",
        typical_halflife_weeks=0.4, saturation_speed="fast",
        notes="Feed-driven and price-sensitive. Control for price and stock availability.",
        aliases=("shopping", "pla", "product_listing", "pmax", "performance_max"),
    ),
    # ---- Retail media & marketplaces ------------------------------------ #
    ChannelType(
        key="retail_media", label="Retail Media", family="retail", funnel_stage="lower",
        typical_halflife_weeks=0.5, saturation_speed="fast",
        notes="Amazon/Walmart/Instacart ads. Highly confounded with distribution and "
              "promotions on the same retailer — control for both or ROAS is inflated.",
        aliases=("amazon", "retail_media", "walmart", "instacart", "criteo", "sponsored_product"),
    ),
    # ---- Social --------------------------------------------------------- #
    ChannelType(
        key="social_prospecting", label="Paid Social — Prospecting", family="social", funnel_stage="upper",
        typical_halflife_weeks=2.0, saturation_speed="medium",
        notes="Genuinely incremental audience-building. Longer carryover than retargeting.",
        aliases=("prospecting", "social_prospecting", "acquisition_social"),
    ),
    ChannelType(
        key="social_retargeting", label="Paid Social — Retargeting", family="social", funnel_stage="lower",
        typical_halflife_weeks=0.4, saturation_speed="fast",
        notes=(
            "Structurally over-credited: it targets people who already engaged. "
            "Expect low true incrementality and validate with a holdout before "
            "believing any ROAS the model reports."
        ),
        aliases=("retargeting", "remarketing", "rtg_"),
    ),
    ChannelType(
        key="meta", label="Meta (Facebook + Instagram)", family="social", funnel_stage="mid",
        typical_halflife_weeks=1.5, saturation_speed="medium",
        notes="Combine Facebook and Instagram unless they are bought and flighted "
              "independently — otherwise collinearity destroys both estimates.",
        aliases=("meta", "facebook", "fb_", "instagram", "ig_"),
    ),
    ChannelType(
        key="tiktok", label="TikTok", family="social", funnel_stage="upper",
        typical_halflife_weeks=1.5, saturation_speed="medium",
        aliases=("tiktok", "tik_tok"),
    ),
    ChannelType(
        key="social", label="Paid Social (other)", family="social", funnel_stage="mid",
        typical_halflife_weeks=1.5, saturation_speed="medium",
        aliases=("social", "snapchat", "snap_", "pinterest", "linkedin", "reddit", "twitter", "x_ads"),
    ),
    # ---- Video ---------------------------------------------------------- #
    ChannelType(
        key="youtube", label="YouTube", family="video", funnel_stage="upper",
        typical_halflife_weeks=2.5, saturation_speed="slow",
        notes="Reach-and-frequency channel. If you have reach and frequency columns, "
              "use them — spend alone confounds price changes with delivery changes.",
        aliases=("youtube", "yt_"),
    ),
    ChannelType(
        key="ctv", label="Connected TV / OLV", family="video", funnel_stage="upper",
        typical_halflife_weeks=3.0, saturation_speed="slow",
        aliases=("ctv", "connected_tv", "olv", "online_video", "preroll", "instream", "video"),
    ),
    ChannelType(
        key="tv", label="Linear TV", family="tv", funnel_stage="upper",
        typical_halflife_weeks=3.5, saturation_speed="slow", default_adstock="delayed",
        notes="Longest carryover and a delayed peak. Prefer GRPs/TVRs over spend as the "
              "model input — spend mixes delivery with cost inflation.",
        aliases=("tv", "television", "broadcast", "linear_tv", "grp", "trp", "tvr"),
    ),
    ChannelType(
        key="cinema", label="Cinema", family="tv", funnel_stage="upper",
        typical_halflife_weeks=3.0, saturation_speed="slow", default_adstock="delayed",
        aliases=("cinema", "movie_theater"),
    ),
    # ---- Display -------------------------------------------------------- #
    ChannelType(
        key="display", label="Display / Programmatic", family="display", funnel_stage="upper",
        typical_halflife_weeks=1.5, saturation_speed="medium",
        notes="Viewability and fraud make raw impressions unreliable; prefer viewable "
              "impressions when available.",
        aliases=("display", "banner", "programmatic", "dsp", "gdn", "dv360", "trade_desk"),
    ),
    ChannelType(
        key="native", label="Native / Content Discovery", family="display", funnel_stage="mid",
        typical_halflife_weeks=1.0, saturation_speed="medium",
        aliases=("native", "taboola", "outbrain"),
    ),
    # ---- Audio ---------------------------------------------------------- #
    ChannelType(
        key="audio", label="Digital Audio / Podcast", family="audio", funnel_stage="upper",
        typical_halflife_weeks=2.0, saturation_speed="slow",
        aliases=("audio", "podcast", "spotify", "streaming_audio"),
    ),
    ChannelType(
        key="radio", label="Radio", family="audio", funnel_stage="upper",
        typical_halflife_weeks=2.5, saturation_speed="slow", default_adstock="delayed",
        aliases=("radio", "am_fm", "broadcast_radio"),
    ),
    # ---- Out of home & print -------------------------------------------- #
    ChannelType(
        key="ooh", label="Out of Home", family="ooh", funnel_stage="upper",
        typical_halflife_weeks=4.0, saturation_speed="slow", default_adstock="delayed",
        notes="Bought in fixed 2-week cycles in most markets, which creates step-shaped "
              "spend that is easy to confuse with seasonality.",
        aliases=("ooh", "outdoor", "billboard", "transit", "dooh"),
    ),
    ChannelType(
        key="print", label="Print", family="print", funnel_stage="upper",
        typical_halflife_weeks=3.0, saturation_speed="slow", default_adstock="delayed",
        aliases=("print", "newspaper", "magazine", "press", "insert"),
    ),
    ChannelType(
        key="direct_mail", label="Direct Mail", family="print", funnel_stage="lower",
        typical_halflife_weeks=2.0, saturation_speed="medium", default_adstock="delayed",
        notes="Delivery lags the send date by days to weeks — the delayed adstock "
              "peak is real, not an artefact.",
        aliases=("direct_mail", "dm_", "mailer", "catalog", "catalogue"),
    ),
    # ---- Partner / performance ------------------------------------------ #
    ChannelType(
        key="affiliate", label="Affiliate", family="partner", funnel_stage="lower",
        typical_halflife_weeks=0.3, saturation_speed="fast",
        notes="Frequently last-click cannibalisation of organic and brand search. "
              "Low true incrementality is the norm.",
        aliases=("affiliate", "partner_", "cj_", "awin"),
    ),
    ChannelType(
        key="influencer", label="Influencer", family="partner", funnel_stage="mid",
        typical_halflife_weeks=1.5, saturation_speed="medium",
        notes="Lumpy and often mis-dated — align spend to the post date, not the invoice date.",
        aliases=("influencer", "creator", "ugc_"),
    ),
    ChannelType(
        key="sponsorship", label="Sponsorship", family="partner", funnel_stage="upper",
        typical_halflife_weeks=4.0, saturation_speed="slow", default_adstock="delayed",
        notes="Contracted annually; spend is usually accrued evenly and does not "
              "reflect exposure. Use event dates instead.",
        aliases=("sponsorship", "sponsor", "partnership"),
    ),
    # ---- Owned / organic ------------------------------------------------- #
    ChannelType(
        key="email", label="Email / CRM", family="owned", funnel_stage="lower",
        default_role="organic_media", typical_halflife_weeks=0.5, saturation_speed="fast",
        notes="Has no media spend, so it has no ROAS. Model sends or delivered volume "
              "as an organic channel; reporting a ROAS for it is a category error.",
        aliases=("email", "crm", "newsletter", "edm_", "sends"),
    ),
    ChannelType(
        key="organic_social", label="Organic Social", family="owned", funnel_stage="upper",
        default_role="organic_media", typical_halflife_weeks=1.5, saturation_speed="medium",
        aliases=("organic_social", "owned_social"),
    ),
    ChannelType(
        key="organic_search", label="Organic Search / SEO", family="owned", funnel_stage="lower",
        default_role="organic_media", typical_halflife_weeks=2.0, saturation_speed="slow",
        notes="Strongly endogenous with paid search and with the brand's own popularity. "
              "Usually better as a mediator or a control than as a channel.",
        aliases=("organic_search", "seo", "organic_traffic"),
    ),
    ChannelType(
        key="pr", label="PR / Earned", family="owned", funnel_stage="upper",
        default_role="organic_media", typical_halflife_weeks=2.0, saturation_speed="slow",
        aliases=("pr_", "earned", "press_coverage"),
    ),
    ChannelType(
        key="app_push", label="Push / In-app", family="owned", funnel_stage="lower",
        default_role="organic_media", typical_halflife_weeks=0.3, saturation_speed="fast",
        aliases=("push", "in_app", "inapp"),
    ),
    # ---- Non-media treatments -------------------------------------------- #
    ChannelType(
        key="price", label="Price / Discount", family="business", funnel_stage="lower",
        default_role="non_media_treatment", typical_halflife_weeks=0.0, saturation_speed="fast",
        default_adstock="none",
        notes="Never a media channel. Enter as a non-media treatment with a negative "
              "expected sign (for price) or positive (for discount depth). Omitting "
              "price is the most common source of inflated media ROAS in retail.",
        aliases=("price", "asp_", "discount", "promo", "markdown", "coupon"),
    ),
    ChannelType(
        key="distribution", label="Distribution / Availability", family="business", funnel_stage="lower",
        default_role="non_media_treatment", typical_halflife_weeks=0.0, saturation_speed="fast",
        default_adstock="none",
        notes="Store count, ACV, shelf facings, stock availability. Moves sales "
              "independently of media and correlates with media plans.",
        aliases=("distribution", "acv", "store_count", "stores", "availability", "stock", "oos"),
    ),
)

_DEFAULT_KEY = "display"
_BY_KEY: dict[str, ChannelType] = {ct.key: ct for ct in CHANNEL_TYPES}

# Aliases shorter than this only match on token boundaries. Without that rule
# "pla" (product listing ads) matches inside "display" and "dm" matches inside
# "admin" — the kind of silent misclassification that produces a confident,
# wrong prior.
_UNAMBIGUOUS_ALIAS_LENGTH = 7

_CAMEL_BOUNDARIES = (
    re.compile(r"(?<=[a-z0-9])(?=[A-Z])"),   # spendTv   -> spend_Tv
    re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])"), # TVSpend   -> TV_Spend
)


def normalize_column(column_name: str) -> str:
    """Lowercase, split camelCase, and delimit a column name for alias matching.

    Returns the name wrapped in underscores so token matches can be tested as
    plain substrings: ``"spend_TV"`` becomes ``"_spend_tv_"``.
    """
    name = column_name
    for pattern in _CAMEL_BOUNDARIES:
        name = pattern.sub("_", name)
    name = re.sub(r"[^A-Za-z0-9]+", "_", name).lower()
    return f"_{name.strip('_')}_"


def _alias_matches(alias: str, normalized: str) -> bool:
    token = alias.strip("_").replace("-", "_")
    if f"_{token}_" in normalized:
        return True
    # Long, distinctive aliases may also appear without a delimiter
    # ("spendfacebook"), where a boundary rule would produce a false negative.
    return len(token) >= _UNAMBIGUOUS_ALIAS_LENGTH and token in normalized.replace("_", "")


# Aliases are tested longest-first so the most specific one wins regardless of
# declaration order: "organic_search" must beat "search", and "brand_search"
# must beat both. Ties fall back to declaration order.
_ALIAS_INDEX: tuple[tuple[str, str], ...] = tuple(
    sorted(
        ((alias, ct.key) for ct in CHANNEL_TYPES for alias in ct.aliases),
        key=lambda pair: -len(pair[0].strip("_")),
    )
)


def classify_channel(column_name: str) -> str:
    """Return the taxonomy key for a column name.

    Matching is on token boundaries and longest-alias-first, so
    ``organic_search_sessions`` resolves to ``organic_search`` rather than
    ``sem``. Falls back to ``display`` — the most neutral paid-media profile —
    when nothing matches.
    """
    normalized = normalize_column(column_name)
    for alias, key in _ALIAS_INDEX:
        if _alias_matches(alias, normalized):
            return key
    return _DEFAULT_KEY


def classify_channels(columns: list[str]) -> dict[str, str]:
    return {col: classify_channel(col) for col in columns}


def get_channel_type(key: str) -> ChannelType:
    """Look up a taxonomy entry, falling back to the neutral default."""
    return _BY_KEY.get(key, _BY_KEY[_DEFAULT_KEY])


def halflife_to_alpha(halflife_periods: float) -> float:
    """Geometric adstock decay rate implied by a half-life.

    ``alpha ** halflife == 0.5``, so ``alpha = 0.5 ** (1 / halflife)``.
    A half-life of 0 (no carryover) maps to alpha = 0.
    """
    if halflife_periods <= 0:
        return 0.0
    return float(0.5 ** (1.0 / halflife_periods))


def alpha_to_halflife(alpha: float) -> float:
    """Inverse of :func:`halflife_to_alpha`, in data periods."""
    import math

    if alpha <= 0:
        return 0.0
    if alpha >= 1:
        return float("inf")
    return float(math.log(0.5) / math.log(alpha))


def suggested_l_max(halflife_periods: float, retained: float = 0.05) -> int:
    """Lag horizon that retains at least ``retained`` of the initial impulse.

    Truncating the adstock kernel too early silently discards effect and biases
    the decay parameter downwards; the default keeps 95% of the mass.
    """
    import math

    if halflife_periods <= 0:
        return 1
    alpha = halflife_to_alpha(halflife_periods)
    if alpha <= 0:
        return 1
    return max(1, int(math.ceil(math.log(retained) / math.log(alpha))))
