"""Configurable version of the notebook 01 cleaning pipeline (NumPy only).

Notebook 01 keeps 45 of the 321 raw BRFSS columns: it drops whole groups of
columns by hand (administrative, post-diagnosis, optional modules, ...). This
module reproduces that pipeline with each group as a switch, so notebook 06
can measure what every group is worth, and adds a few optional extra
features (squares of numeric columns, interactions with age and sex).

The decoding rules and group lists are copied from notebook 01; keep them in
sync if notebook 01 changes.

Typical use:

    raw = load_raw(DATA_DIR)
    x_tr, x_te, names = build_features(raw, keep_groups=("redundant",))

Preprocessing statistics (medians, clipping percentiles, means, category
levels) are computed on the whole training set, as in notebook 01.
"""

from pathlib import Path

import numpy as np

# ------------------------------------------------------------- raw data loading


def load_raw(data_dir, cache_path=None):
    """Load x_train.csv, y_train.csv and x_test.csv as float arrays.

    Parsing the CSVs takes a few minutes, so the arrays are cached in
    `cache_path` (default: dataset/prepared/raw_arrays.npz, which is
    git-ignored) and loaded from there the next times.

    Returns :

    - raw dict : x_train (N, 321) and x_test (M, 321) with blanks as NaN,
      y_train (N, ) in {0, 1}, test_ids (M, ), names (321, )"""

    data_dir = Path(data_dir)
    cache_path = Path(cache_path or data_dir / "prepared" / "raw_arrays.npz")

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        return {key: data[key] for key in data.files}

    def read(path):
        header = path.open().readline().strip().split(",")
        values = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float,
                               missing_values="", filling_values=np.nan)
        return np.array(header), values

    names, x_train = read(data_dir / "x_train.csv")
    _, x_test = read(data_dir / "x_test.csv")
    _, y = read(data_dir / "y_train.csv")

    raw = {
        "x_train": x_train[:, 1:],
        "x_test": x_test[:, 1:],
        "y_train": (y[:, 1] == 1).astype(int),
        "test_ids": x_test[:, 0].astype(int),
        "names": names[1:],
    }
    assert np.array_equal(x_train[:, 0], y[:, 0]), "x_train and y_train rows differ"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **raw)
    return raw


# ------------------------------------------- BRFSS missing codes (notebook 01)

PROTECTED_COLUMNS = {
    "Id", "_STATE", "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR",
    "DISPCODE", "SEQNO", "_PSU",
}
SMALL_MISSING_CODES = {7.0, 9.0}
LARGE_MISSING_CODES = {77.0, 99.0, 777.0, 999.0, 7777.0, 9999.0, 99999.0}

# Columns where the generic rule is wrong (checked in the BRFSS codebook).
MISSING_CODE_OVERRIDES = {
    "EMPLOY1": {9.0},          # code 7 means "retired", a real answer
    "_RACE": {9.0},            # code 7 means "multiracial, non-Hispanic"
    "PHYSHLTH": {77.0, 99.0},  # 7 or 9 days are real values
    "MENTHLTH": {77.0, 99.0},
    "_AGE80": set(),
    "_BMI5": set(),
    "_FRUTSUM": set(),
    "_VEGESUM": set(),
}
ZERO_CODES = {"PHYSHLTH": 88.0, "MENTHLTH": 88.0}  # code 88 means "no days"


def missing_codes_for(name, train_column):
    """Codes treated as missing for one column, decided from the train column."""

    if name in MISSING_CODE_OVERRIDES:
        return MISSING_CODE_OVERRIDES[name]
    if name in PROTECTED_COLUMNS:
        return LARGE_MISSING_CODES
    # 7 / 9 are missing codes only for single-digit answers.
    has_values = np.any(~np.isnan(train_column))
    max_value = np.nanmax(train_column) if has_values else 0
    return LARGE_MISSING_CODES | (SMALL_MISSING_CODES if max_value <= 9 else set())


def decode_missing(raw):
    """Replace BRFSS "don't know / refused" codes with NaN (train and test)."""

    x_tr, x_te = raw["x_train"].copy(), raw["x_test"].copy()
    for j, name in enumerate(raw["names"]):
        codes = list(missing_codes_for(name, x_tr[:, j]))
        for x in (x_tr, x_te):
            x[np.isin(x[:, j], codes), j] = np.nan
            if name in ZERO_CODES:
                x[x[:, j] == ZERO_CODES[name], j] = 0.0
    return x_tr, x_te


# ------------------------------------------------ column groups (notebook 01)

FEATURE_GROUPS = {
    "administrative": {
        "Id", "SEQNO", "IDAY", "IMONTH", "IDATE", "FMONTH", "IYEAR",
        "DISPCODE", "QSTVER", "QSTLANG", "_PSU", "_STRWT", "_STSTR",
        "_LLCPWT", "_WT2RAKE", "_RAWRAKE", "LANDLINE", "CADULT",
        "CTELNUM1", "CELLFON2", "PVTRESD2", "CSTATE", "STATERES",
        "CTELENUM", "CELLFON3", "PVTRESD1", "HHADULT", "NUMHHOL2",
        "NUMMEN", "NUMWOMEN", "NUMADULT", "CPDEMO1", "MSCODE",
        "_DUALUSE", "_DUALCOR", "COLGHOUS", "CCLGHOUS", "LADULT", "PCDMDECN",
        "NUMPHON2",
    },
    # Notebook 01's "post-diagnosis" group, split in two. These questions are
    # about care or advice that typically FOLLOWS a cardiovascular event
    # (rehab after a heart attack, aspirin to prevent another one...), so
    # they may encode the label itself. Treat any gain from them as suspect.
    "cvd_care": {
        "HAREHAB1", "STREHAB1", "CVDASPRN", "ASPUNSAF", "RDUCHART",
        "RDUCSTRK", "RLIVPAIN",
    },
    # The rest of the post-diagnosis group: general risk-factor management,
    # not specific to heart disease.
    "risk_management": {
        "BPMEDS", "DRADVISE", "WTCHSALT", "LONGWTCH", "VINOCRE2", "PCPSADE1",
    },
    "optional_modules": {
        "HPVADSHT", "PSATIME", "PCPSARS1", "LENGEXAM", "PROFEXAM",
        "PCPSADI1", "PSATEST1", "PCPSAAD2", "PCPSARE1", "CASTHNO2",
        "HPLSTTST", "HPVADVC2", "HIVTSTD3", "WHRTST10", "CAREGIV1",
        "CIMEMLOS", "SXORIENT", "TRNSGNDR", "IMFVPLAC", "FLSHTMY2",
        "RCSGENDR", "RCSRLTN2", "_CPRACE", "_CRACE1", "_CLLCPWT",
        "_CHISPNC", "PREGNANT", "PDIABTST", "PREDIAB1", "HIVTST6",
        "_AIDTST3", "PNEUVAC3", "_PNEUMO2", "FLUSHOT6", "_FLSHOT6",
        "SHINGLE2", "TETANUS",
    },
    "follow_up": {
        "POORHLTH", "SMOKDAY2", "STOPSMK2", "LASTSMK2", "AVEDRNK2",
        "DRNK3GE5", "MAXDRNKS", "JOINPAIN", "LMTJOIN3", "DIABAGE2",
        "DOCTDIAB", "CHKHEMO3", "FEETCHK", "FEETCHK2", "BLDSUGAR",
        "EYEEXAM", "DIABEYE", "DIABEDU", "INSULIN", "ARTTODAY",
        "HADMAM", "HOWLONG", "HADPAP2", "LASTPAP2", "HPVTEST",
        "HADHYST2", "BLDSTOOL", "LSTBLDS3", "HADSIGM3", "HADSGCO1",
        "LASTSIG3", "CASTHDX2", "EMTSUPRT", "LSATISFY", "MISTMNT",
        "_HCVU651", "_LMTWRK1", "_LMTACT1", "_LMTSCL1",
    },
    "data_quality": {
        "_VEGETEX", "_FRUITEX", "_VEG23", "_FRT16", "_VEGRESP", "_FRTRESP",
        "_MISVEGN", "_MISFRTN",
    },
    "redundant": {
        "PAFREQ2_", "PAFREQ1_", "PADUR2_", "PADUR1_", "EXERHMM2", "EXERHMM1",
        "EXEROFT2", "EXEROFT1", "EXRACT21", "EXRACT11", "PAMIN21_", "PAMIN11_",
        "_MINAC21", "_MINAC11", "METVL21_", "METVL11_", "PAVIG21_", "PAVIG11_",
        "ACTIN21_", "ACTIN11_", "PA1MIN_", "PA1VIGM_", "PAMISS1_", "STRFREQ_",
        "STRENGTH", "_PASTRNG", "_PAREC1", "_PASTAE1", "_PA30021", "_PA300R2",
        "_PACAT1", "_PA150R2", "EXERANY2", "FRUIT1", "FRUITJU1", "FVGREEN",
        "FVORANG", "FVBEANS", "VEGETAB1", "FRUTDA1_", "FTJUDA1_", "GRENDAY_",
        "ORNGDAY_", "BEANDAY_", "VEGEDA1_", "_FRTLT1", "_VEGLT1", "WEIGHT2",
        "WTKG3", "HEIGHT3", "HTIN4", "HTM4", "_BMI5CAT", "_RFBMI5", "SMOKE100",
        "_RFSMOK3", "ALCDAY5", "DRNKANY5", "DROCDY3_", "_DRNKWEK", "_AGEG5YR",
        "_AGE_G", "_AGE65YR", "_MRACE1", "_PRACE1", "_RACEG21", "_RACE_G1",
        "_RACEGR3", "_HISPANC", "EDUCA", "INCOME2", "CHILDREN", "_RFHYPE5",
        "_RFHLTH", "_CASTHM1", "_ASTHMS1", "_LTASTH1", "_DRDXAR1", "CHOLCHK",
        "BLOODCHO", "TOLDHI2", "MAXVO2_", "FC60_", "SEATBELT", "_RFSEAT2",
        "_RFSEAT3", "INTERNET",
    },
}
OPTIONAL_MODULE_PREFIXES = ("AS", "VI", "AD", "ARTH", "CD", "CRGV", "SCNT")

# Groups that can be switched back on (administrative columns never are).
OPTIONAL_GROUPS = [g for g in FEATURE_GROUPS if g != "administrative"]


def column_groups(names):
    """Map every excluded column name to its group (notebook 01 rules)."""

    groups = {}
    for group, members in FEATURE_GROUPS.items():
        for name in members:
            groups.setdefault(name, group)
    for name in names:
        if name.startswith(OPTIONAL_MODULE_PREFIXES) and name != "ASTHMA3":
            groups.setdefault(name, "optional_modules")
    return groups


def selected_columns(names, keep_groups=()):
    """Boolean mask of the raw columns kept when `keep_groups` are added back."""

    unknown = set(keep_groups) - set(OPTIONAL_GROUPS)
    if unknown:
        raise ValueError(f"Unknown groups: {sorted(unknown)}")

    groups = column_groups(names)
    return np.array([groups.get(name) in (None, *keep_groups) for name in names])


# ------------------------------------------------------------- matrix building


def detect_categorical(x, names, max_unique=15):
    """Whole-number columns with at most `max_unique` values are categorical."""

    is_cat = np.zeros(x.shape[1], dtype=bool)
    for j in range(x.shape[1]):
        values = np.unique(x[:, j][~np.isnan(x[:, j])])
        is_cat[j] = np.all(np.mod(values, 1) == 0) and len(values) <= max_unique
    is_cat[np.asarray(names) == "_STATE"] = True  # state codes are labels
    return is_cat


def build_features(
    raw,
    keep_groups=(),
    square_numeric=False,
    interact_with=(),
    min_level_count=20,
    decoded=None,
):
    """Build the model matrices (no bias column) for train and test.

    Same steps as notebook 01: missing categorical answers become their own
    category, numeric columns get median imputation (+ a missing indicator
    when more than 5% is missing), 1st/99th percentile clipping and
    standardization, categorical columns are one-hot encoded with the first
    level dropped.

    Inputs :

    - raw dict : output of `load_raw`

    - keep_groups iterable of str : groups from OPTIONAL_GROUPS to add back

    - square_numeric bool : add the standardized square of every numeric
      column (squares of 0/1 columns would only duplicate them)

    - interact_with iterable of str : output column names (e.g. "_AGE80",
      "SEX=2") to multiply with the other columns that come from notebook
      01's original 45 raw columns (all columns would be too many)

    - min_level_count int : drop one-hot columns with fewer training rows

    - decoded tuple or None : output of `decode_missing(raw)`, to reuse it

    Returns :

    - x_train ndarray (N, D), x_test ndarray (M, D), feature_names (D, )"""

    x_tr_all, x_te_all = decoded if decoded is not None else decode_missing(raw)
    mask = selected_columns(raw["names"], keep_groups)
    x_tr, x_te = x_tr_all[:, mask], x_te_all[:, mask]
    names = raw["names"][mask]

    # Columns that are empty or constant in train carry no information.
    informative = np.array([
        len(np.unique(col[~np.isnan(col)])) > 1 for col in x_tr.T
    ])
    x_tr, x_te, names = x_tr[:, informative], x_te[:, informative], names[informative]

    is_cat = detect_categorical(x_tr, names)
    out = _Columns()
    numeric = _add_numeric(out, x_tr, x_te, names, np.nonzero(~is_cat)[0])
    _add_categorical(out, x_tr, x_te, names, np.nonzero(is_cat)[0], min_level_count)
    if square_numeric:
        _add_squares(out, numeric)
    base_sources = set(raw["names"][selected_columns(raw["names"])])
    _add_interactions(out, interact_with, base_sources)

    return np.column_stack(out.train), np.column_stack(out.test), np.array(out.names)


class _Columns:
    """Output columns of `build_features`, built one at a time.

    `sources` keeps the raw column each output column comes from."""

    def __init__(self):
        self.train, self.test, self.names, self.sources = [], [], [], []

    def add(self, train_col, test_col, name, source):
        self.train.append(train_col)
        self.test.append(test_col)
        self.names.append(name)
        self.sources.append(source)


def _add_numeric(out, x_tr, x_te, names, columns):
    """Impute (median), flag missing, clip and standardize numeric columns.

    Returns the output indices of the standardized columns."""

    numeric = []
    for j in columns:
        col_tr, col_te = x_tr[:, j].copy(), x_te[:, j].copy()
        missing_tr, missing_te = np.isnan(col_tr), np.isnan(col_te)
        median = np.nanmedian(col_tr)
        col_tr[missing_tr], col_te[missing_te] = median, median

        lo, hi = np.percentile(col_tr, [1, 99])
        col_tr, col_te = np.clip(col_tr, lo, hi), np.clip(col_te, lo, hi)
        mu, sigma = col_tr.mean(), col_tr.std()
        sigma = sigma if sigma > 0 else 1.0

        out.add((col_tr - mu) / sigma, (col_te - mu) / sigma, names[j], names[j])
        numeric.append(len(out.names) - 1)
        if missing_tr.mean() > 0.05:
            out.add(missing_tr.astype(float), missing_te.astype(float),
                    f"{names[j]}_missing", names[j])
    return numeric


def _add_categorical(out, x_tr, x_te, names, columns, min_level_count):
    """One-hot encode categorical columns, missing = its own level (-1).

    The first level is dropped, and levels with fewer than `min_level_count`
    training rows are skipped."""

    for j in columns:
        col_tr = np.where(np.isnan(x_tr[:, j]), -1.0, x_tr[:, j])
        col_te = np.where(np.isnan(x_te[:, j]), -1.0, x_te[:, j])
        levels, counts = np.unique(col_tr, return_counts=True)
        for level in levels[1:][counts[1:] >= min_level_count]:
            out.add((col_tr == level).astype(float), (col_te == level).astype(float),
                    f"{names[j]}={level:g}", names[j])


def _add_squares(out, numeric):
    """Standardized squares of the numeric output columns."""

    for k in numeric:
        sq_tr, sq_te = out.train[k] ** 2, out.test[k] ** 2
        mu, sigma = sq_tr.mean(), sq_tr.std()
        if sigma < 1e-12:
            continue  # two symmetric values -> constant square, no information
        out.add((sq_tr - mu) / sigma, (sq_te - mu) / sigma,
                f"{out.names[k]}^2", out.sources[k])


def _add_interactions(out, partners, base_sources):
    """Products of each partner column with the columns from `base_sources`."""

    base = len(out.names)
    for partner in partners:
        p = out.names.index(partner)
        for k in range(base):
            if out.sources[k] in base_sources and out.sources[k] != out.sources[p]:
                out.add(out.train[p] * out.train[k], out.test[p] * out.test[k],
                        f"{partner}*{out.names[k]}", out.sources[k])


# ------------------------------------------------------- collinearity removal


def independent_columns(tx, tol=1e-9):
    """Indices of a maximal set of linearly independent columns of `tx`.

    One-hot groups, missing indicators and derived columns (e.g. age in years
    and age bands) make some columns exact linear combinations of others, so
    tx.T @ tx is singular and least squares has no unique solution. Ridge
    hides this with its penalty; least squares needs the redundant columns
    removed.

    Greedy Gram-Schmidt on the Gram matrix G = tx.T @ tx / N: a column is
    kept if the part of it not explained by the columns already kept has a
    squared norm above `tol` times its own. Uses no labels.

    Returns :

    - keep ndarray of int : column indices to keep, in their original order"""

    gram = tx.T @ tx / len(tx)
    keep = []
    chol = np.zeros((0, 0))  # Cholesky factor of gram[keep][:, keep]

    for j in range(gram.shape[0]):
        if keep:
            z = np.linalg.solve(chol, gram[keep, j])
            residual = gram[j, j] - z @ z
        else:
            z, residual = np.zeros(0), gram[j, j]
        if residual > tol * max(gram[j, j], 1e-300):
            k = len(keep)
            new = np.zeros((k + 1, k + 1))
            new[:k, :k] = chol
            new[k, :k] = z
            new[k, k] = np.sqrt(residual)
            chol = new
            keep.append(j)

    return np.array(keep)


# ------------------------------------------------- Newton logistic regression


def reg_logistic_newton(
    y, tx, lambda_, max_iters=30, tol=1e-8, chunk_rows=50_000, pos_weight=1.0
):
    """L2-regularized logistic regression solved with Newton's method.

    Minimizes the same objective as implementations.reg_logistic_regression
    (mean logistic loss + lambda_ * ||w||^2) but uses the Hessian, so it
    converges in a few tens of steps whatever the conditioning of tx, with no
    step size to tune. Each step costs O(N D^2) for the Hessian.

    A step is halved until the objective decreases (needed only far from
    the optimum).

    With `pos_weight` != 1 every positive row counts `pos_weight` times in
    the (weighted average) logistic loss, an alternative to undersampling.

    Returns :

    - w ndarray (D, ) : weights

    - n_iters int : number of Newton steps taken"""

    n, d = tx.shape
    w = np.zeros(d)
    # Row weights summing to 1; all equal to 1/n for pos_weight = 1.
    c = np.where(y == 1, pos_weight, 1.0)
    c = c / c.sum()

    def objective(w):
        z = tx @ w
        return c @ (np.logaddexp(0, z) - y * z) + lambda_ * w @ w

    current = objective(w)
    for it in range(1, max_iters + 1):
        p = 1 / (1 + np.exp(-np.clip(tx @ w, -500, 500)))
        grad = tx.T @ (c * (p - y)) + 2 * lambda_ * w
        # X^T S X built in row chunks, so we never copy the whole matrix.
        hessian = 2 * lambda_ * np.eye(d)
        s = c * p * (1 - p)
        for i in range(0, n, chunk_rows):
            block = tx[i:i + chunk_rows]
            hessian += (block * s[i:i + chunk_rows, None]).T @ block
        step = np.linalg.solve(hessian, grad)

        t = 1.0
        while True:
            candidate = objective(w - t * step)
            if candidate <= current or t < 1e-4:
                break
            t /= 2
        w = w - t * step

        decrease, current = current - candidate, candidate
        if decrease < tol:
            break

    return w, it
