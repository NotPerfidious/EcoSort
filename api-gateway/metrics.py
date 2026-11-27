from prometheus_client import Counter, Gauge, Histogram

REQUESTS_TOTAL = Counter(
    "ecosort_requests_total",
    "Total inference requests processed by api-gateway",
    ["category", "tier_used", "status"],
)

REQUEST_LATENCY_MS = Histogram(
    "ecosort_request_latency_ms",
    "End-to-end gateway inference latency in milliseconds",
    buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000),
)

TIER_LATENCY_MS = Histogram(
    "ecosort_tier_latency_ms",
    "Downstream inference latency per tier in milliseconds",
    ["tier"],
    buckets=(2, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000),
)

FALLBACK_TOTAL = Counter(
    "ecosort_fallback_total",
    "Total requests served via SLA safe-default fallback",
    ["reason"],
)

CIRCUIT_BREAKER_STATE = Gauge(
    "ecosort_circuit_breaker_state",
    "Tier-2 circuit breaker state (0=CLOSED, 1=HALF_OPEN, 2=OPEN)",
)

SHADOW_REQUESTS_TOTAL = Counter(
    "ecosort_shadow_requests_total",
    "Total asynchronous shadow requests mirrored to INT8 MobileNetV4",
    ["status"],
)

SHADOW_AGREEMENT_TOTAL = Counter(
    "ecosort_shadow_agreement_total",
    "Shadow predictions matching vs diverging from primary FP32",
    ["agreed"],
)

SHADOW_LATENCY_DELTA_MS = Gauge(
    "ecosort_shadow_latency_delta_ms",
    "Latest shadow INT8 latency minus primary FP32 latency (ms)",
)

ACTIVE_LEARNING_UPLOADS_TOTAL = Counter(
    "ecosort_active_learning_uploads_total",
    "Low-confidence samples pushed to S3 active learning bucket",
    ["status"],
)

INPUT_BRIGHTNESS_DRIFT_ZSCORE = Gauge(
    "ecosort_input_brightness_drift_zscore",
    "Absolute z-score shift of rolling input brightness vs training reference",
)

PREDICTION_CONFIDENCE_ROLLING_MEAN = Gauge(
    "ecosort_prediction_confidence_rolling_mean",
    "Rolling mean Tier-1 prediction confidence over recent window",
)

PREDICTION_DISTRIBUTION_PSI = Gauge(
    "ecosort_prediction_distribution_psi",
    "Population Stability Index of rolling predicted class mix vs reference",
)

DRIFT_ALERT_ACTIVE = Gauge(
    "ecosort_drift_alert_active",
    "1 when live input or prediction drift threshold is breached, else 0",
)

CONFIDENCE_THRESHOLD_GAUGE = Gauge(
    "ecosort_confidence_threshold",
    "Currently active dynamic routing confidence threshold (tau)",
)
