"""Experiment constants — centralized magic numbers.

All thresholds, statistical parameters, and experiment defaults live here.
Import from this module instead of hardcoding values in experiment files.

Each constant documents WHY it has its particular value.
"""

# ================================================================
# AUROC Verdict Thresholds
# ================================================================

# Above this = classifier picks up prompt-level variation, not condition signal.
# 0.65 is conservative: 0.50 is chance, 0.65 indicates systematic structure.
AUROC_SAME_CONDITION = 0.65

# Above this = signal driven by input structure (token count), not cache geometry.
# Set above SAME_CONDITION to avoid false positives from prompt variation alone.
AUROC_INPUT_CONFOUND = 0.70

# Below this = signal collapsed after FWL residualization. Near chance.
AUROC_FWL_COLLAPSE = 0.55

# Above this = signal preserved after FWL residualization. Genuine geometry.
AUROC_FWL_PRESERVE = 0.60

# Minimum meaningful AUROC change between conditions. Below = noise.
AUROC_DELTA = 0.05

# Minimum AUROC advantage of cache features over response-format features.
AUROC_GEOMETRY_ADVANTAGE = 0.10

# Tight tolerance for "confirmation" verdict (delta must be smaller than this).
AUROC_CONFIRMATION_TOLERANCE = 0.02


# ================================================================
# Statistical Parameters
# ================================================================

# Significance level for all hypothesis tests.
ALPHA = 0.05

# Confidence interval level (1 - ALPHA).
CI_LEVEL = 0.95

# Default random seed for reproducibility.
DEFAULT_SEED = 42


# ================================================================
# Power Analysis
# ================================================================

# Adequate statistical power (Cohen 1988).
POWER_ADEQUATE = 0.80

# Below this = underpowered, results unreliable.
POWER_UNDERPOWERED = 0.50

# TOST equivalence margin in Cohen's d units.
TOST_DELTA = 0.3


# ================================================================
# Resampling
# ================================================================

# Number of permutations for null distribution estimation.
N_PERMUTATIONS = 10_000

# Number of bootstrap resamples for confidence intervals.
N_BOOTSTRAP = 10_000

# Repeated cross-validation iterations for variance estimation.
N_REPEATED_CV = 20

# F02 held-out bootstrap (lower due to small held-out sets).
N_BOOTSTRAP_F02 = 2_000


# ================================================================
# Cross-Validation
# ================================================================

# Default number of folds for StratifiedKFold / GroupKFold.
N_SPLITS = 5


# ================================================================
# Classifier
# ================================================================

# LogisticRegression convergence limit.
LOGREG_MAX_ITER = 5_000

# Solver (lbfgs is default in sklearn, explicit for reproducibility).
LOGREG_SOLVER = "lbfgs"


# ================================================================
# Feature Extraction
# ================================================================

# Maximum tokens to generate during model inference.
MAX_NEW_TOKENS = 200

# Generation temperature (0.0 = greedy/deterministic).
TEMPERATURE = 0.0


# ================================================================
# Experiment-Specific
# ================================================================

# F01a null experiment: number of random-split repeats.
F01A_N_REPEATS = 100

# V03: if this many comparisons show FWL collapse, verdict = FALSIFIED.
FWL_COLLAPSE_COUNT = 3

# V10 power analysis: simulation count.
POWER_N_SIM = 10_000

# Default sample size per condition group.
N_PER_GROUP = 200
