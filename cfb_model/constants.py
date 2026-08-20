"""Every magic value from the notebook pipeline, verbatim.

Sources: p1_DataImportAndWrangle (rolling stats), p4_AdvancedStats (predictMissing,
provider order), p5_SuccessAnalysis (thresholds, gates), p6_CurrentSeasonModel
(training recipe), MidSeason.ipynb (weekly ops — its provider order, with the
'EPSN Bet' typo fixed, is authoritative).
"""

# The 25 base box-score stats that get rolling sums (p1 createTeam).
BASE_STATS = (
    "Points",
    "firstDowns",
    "fumblesLost",
    "fumblesRecovered",
    "interceptions",
    "kickReturnYards",
    "kickingPoints",
    "netPassingYards",
    "passesDeflected",
    "passesIntercepted",
    "passingTDs",
    "puntReturns",
    "qbHurries",
    "rushingAttempts",
    "rushingTDs",
    "rushingYards",
    "sacks",
    "tacklesForLoss",
    "totalFumbles",
    "totalPenaltiesYards",
    "totalYards",
    "turnovers",
    "yardsPerPass",
    "yardsPerRushAttempt",
    "totalTDs",
)

# Trailing-game windows for rolling sums.
WINDOWS = (20, 8)

# Rate stats whose rolling sums are divided by the window (sum -> average).
PER_GAME_STATS = ("yardsPerPass", "yardsPerRushAttempt")

# Feature selection: any column containing one of these substrings (~104 cols).
FEATURE_SUBSTRINGS = ("rolling_sum", "talent", "SP")

# totalTDs components (skipna sum).
TOTAL_TD_COMPONENTS = (
    "passingTDs",
    "rushingTDs",
    "interceptionTDs",
    "kickReturnTDs",
    "puntReturnTDs",
)

# Stat categories whose raw values are compound strings ('12-95', '31:24')
# encoded as decimals via '-'/':' -> '.'  (existing convention since p1).
COMPOUND_STAT_CATEGORIES = (
    "totalPenaltiesYards",
    "completionAttempts",
    "thirdDownEff",
    "fourthDownEff",
    "possessionTime",
)

# Betting-line provider preference (first available wins; else first row).
# NOTE: fixes the 'EPSN Bet' typo present in MidSeason's grabLastWeekData.
PROVIDER_ORDER = ("ESPN Bet", "DraftKings", "consensus", "Bovada")

# CFBD school name -> canonical model key (applied on every load/ingest).
NAME_MAP = {
    "Appalachian State": "App State",
    "Louisiana Monroe": "UL Monroe",
    "Connecticut": "UConn",
    "UT San Antonio": "UTSA",
    "UMass": "Massachusetts",
    "Southern Mississippi": "Southern Miss",
}

# Bet-classification thresholds on historical bin success rate (p5).
THRESH_GOOD = 0.595
THRESH_GREAT = 0.645
THRESH_BEST = 0.695

# Inclusion gates and warm-up.
MIN_FBS_GAMES = 56   # team frame must have >= this many games to get a model
MIN_FCS_GAMES = 10   # FCS frames kept if >= this many games
MIN_TRAIN_ROWS = 8   # buildModel returns None below this
WARMUP_DROP = 20     # first N rows dropped (rolling-sum warm-up)

# predictMissing (p4): rolling cols matching these substrings treat 0 as
# missing; talent columns are appended explicitly (fixCols.extend([...])).
LEGACY_ZERO_AS_MISSING_PATTERNS = (
    "tackle",
    "sacks",
    "Deflected",
    "defensive",
    "qbHurries",
    "fumbles",
    "kickReturn",
    "penalty",
    "Fumble",
)
PREDICT_MISSING_EXTRA_COLS = ("talent", "talent_opp")
PREDICT_MISSING_SEED = 50  # p4 calls predictMissing(..., randomState=50)

# Training recipe (p5/p6 buildModel).
ALPHA_GRID_ARGS = (-4, 4, 50)  # np.logspace(*ALPHA_GRID_ARGS)
LASSO_MAX_ITER = 10_000
BAGGING_N_ESTIMATORS = 50
GRIDSEARCH_CV = 5
GRIDSEARCH_SCORING = "neg_mean_absolute_error"

# Analysis slices (MidSeason cells 43-44).
SINCE_WEEK_RANGE = 14      # 'win rate since week N' for N in range(14)
BIG_SPREAD = 27.5          # |spread| >= this slice

# Prediction/results CSV column orders (legacy-exact).
PREDS_COLUMNS = ("pred", "spread", "spreadDiff", "cover", "gameID", "team", "oppTeam")
RESULTS_COLUMNS = PREDS_COLUMNS + ("day", "scoreDiff", "result", "successRate")
