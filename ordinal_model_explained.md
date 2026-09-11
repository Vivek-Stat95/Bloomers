# `statsmodels.miscmodels.ordinal_model` — Annotated Walkthrough

This file implements **`OrderedModel`**, an ordinal regression model (ordered
probit/logit, i.e. the *proportional odds model*). Below: the statistical
model first, then the code section by section.

---

## 1. The statistical model

Ordinal data (e.g. survey ratings "poor/fair/good/excellent") is not
continuous and not unordered-categorical — it has a natural order but no
meaningful numeric distance between levels. The standard way to model this is
via a **latent variable**:

```
y_latent = X β + u,        u ~ F  (standard normal or standard logistic)
```

`y_latent` is unobserved. What we observe is a coarsened version of it,
determined by a set of increasing **thresholds** (cutpoints)
`-∞ = a_0 < a_1 < ... < a_{K-1} < a_K = ∞`:

```
y = k   if   a_k < y_latent ≤ a_{k+1}
```

So the probability of observing category *k* given `X` is

```
P(y = k | X) = P(a_k < Xβ + u ≤ a_{k+1})
             = P(a_k − Xβ < u ≤ a_{k+1} − Xβ)
             = F(a_{k+1} − Xβ) − F(a_k − Xβ)
```

where `F` is the CDF of `u` (normal ⇒ **ordered probit**, logistic ⇒
**ordered logit / proportional odds model**).

Two structural points that shape the code:

- **No intercept is allowed in `X β`.** An intercept would be perfectly
  collinear with (i.e. indistinguishable from) shifting all the thresholds by
  a constant — the model would not be identified. The thresholds absorb the
  role of the intercept.
- **Thresholds must be increasing.** Rather than constraining an optimizer
  with inequality constraints, the code re-parameterizes: the first threshold
  is free, and each subsequent gap is `exp(increment)`, which is always
  positive. This turns a constrained optimization problem into an
  unconstrained one — a very common trick.

Estimation is by **maximum likelihood**: for each observation with observed
category `y_i`, the log-likelihood contribution is
`log[F(a_{y_i+1} − X_iβ) − F(a_{y_i} − X_iβ)]`, summed over all observations.

---

## 2. Imports and class docstring

```python
from itertools import pairwise
...
class OrderedModel(GenericLikelihoodModel):
```

- `GenericLikelihoodModel` is statsmodels' base class for "write your own
  `loglike`/`loglikeobs`, get an MLE optimizer, standard errors (from the
  numerical Hessian), and a `.fit()`/results API for free."
- `pairwise` (Python 3.10+) turns `[a, b, c]` into `[(a, b), (b, c)]` — used
  later to build threshold names like `"poor/fair"`.
- The docstring restates the latent-variable model above and explicitly
  requires **no constant**, for the identification reason discussed.

---

## 3. `__init__`

```python
def __init__(self, endog, exog, offset=None, distr="probit", **kwds):
    if distr == "probit":
        self.distr = stats.norm
    elif distr == "logit":
        self.distr = stats.logistic
    else:
        self.distr = distr
```
Chooses the link function `F`. `stats.norm` → probit (Gaussian CDF, Φ);
`stats.logistic` → logit (the CDF is the logistic sigmoid, giving the
familiar odds-ratio interpretation of coefficients — this is exactly the
**proportional odds model**). Any `scipy.stats`-like distribution object can
be substituted, since only `.cdf`, `.pdf`, `.ppf` are used.

```python
    if offset is not None:
        offset = np.asarray(offset)
    self.offset = offset
```
An **offset** is a term added to `Xβ` with a fixed coefficient of 1 (common
in count/rate models to include known structural terms without estimating a
coefficient for them).

```python
    endog, labels, is_pandas = self._check_inputs(endog, exog)
    super().__init__(endog, exog, **kwds)
```
Normalizes `endog` (see `_check_inputs` below), then hands off to
`GenericLikelihoodModel.__init__`, which stores `exog`, computes `nobs`,
detects a constant column (`k_constant`), etc.

```python
    k_levels = None
    if not is_pandas:
        if self.endog.ndim == 1:
            unique, index = np.unique(self.endog, return_inverse=True)
            self.endog = index
            labels = unique
            if np.isnan(labels).any():
                raise ValueError(...)
        elif self.endog.ndim == 2:
            ...
            k_levels = self.endog.shape[1]
```
If `endog` was **not** a pandas ordered Categorical, the raw category values
get mapped to consecutive integers `0, 1, 2, ...` via `np.unique(...,
return_inverse=True)` — this is exactly `LabelEncoder`-style behavior. Caveat
baked into the docstring: without a pandas ordered Categorical, categories
are sorted **lexicographically**, which may not match the "true" ordering if
the labels aren't naturally sortable (e.g. "low/medium/high" would sort
wrong). The 2-D branch is a `from_formula`-only path used for categorical
endog handled by patsy's dummy encoding.

```python
    if self.k_constant > 0:
        raise ValueError("There should not be a constant in the model")
    self._initialize_labels(labels, k_levels=k_levels)

    self.k_extra = self.k_levels - 1
    self.df_model = self.k_vars
    self.df_resid = self.nobs - (self.k_vars + self.k_extra)
    self.results_class = OrderedResults
```
Enforces the "no intercept" identification constraint. `k_levels` is the
number of ordinal categories `K`; with `K` categories there are `K − 1`
internal thresholds (`k_extra`), which are additional free parameters beyond
the `k_vars` regression coefficients — hence `df_resid` subtracts both.

---

## 4. `_check_inputs`

Handles the case where `endog` is a pandas ordered `Categorical`:

```python
labels = endog.values.categories
endog = endog.cat.codes
if endog.min() == -1:
    raise ValueError("missing values ... not supported")
```
`.cat.codes` converts categories to integers `0..K-1` **in the category's
defined order** — this is the statistically correct path, since it respects
a user-specified ordering rather than relying on sort order. `-1` is pandas'
code for a missing/unmatched category, so it's explicitly rejected (silently
treating missing as a valid class would corrupt the likelihood). A
`RuntimeWarning` fires if `endog.ordered` is `False`, since then the encoded
order might not be meaningful.

---

## 5. `_initialize_labels`

```python
self.labels = labels
self.k_levels = len(labels) if k_levels is None else k_levels
self.nobs, self.k_vars = self.exog.shape   # or endog.shape[0], 0

threshold_names = [str(x) + "/" + str(y) for x, y in pairwise(labels)]
self.exog_names.extend(threshold_names)
```
Builds human-readable names for the `K − 1` thresholds by pairing adjacent
category labels, e.g. categories `[poor, fair, good]` → threshold names
`"poor/fair"`, `"fair/good"` (the cutpoint separating those two categories).
These get appended to `exog_names` so the fitted output labels
coefficients **and** thresholds together.

---

## 6. `from_formula`

```python
model = super().from_formula(formula, data, *args, drop_cols=["Intercept"], **kwargs)
```
Patsy (the formula engine) always wants to add an intercept when categorical
regressors are present; since an intercept isn't identified here, the code
forces an explicit `Intercept` term into the design and then **drops it**
after the fact — a workaround for identifiability rather than a modeling
choice.

```python
if model.endog.ndim == 2:
    ...
    labels = original_endog.values.categories
    model._initialize_labels(labels)
    model.endog = model.endog.argmax(1)
```
If the formula produced a one-hot–encoded (dummy) endog matrix (patsy's
default for categorical variables), the ordered Categorical's own categories
are used for labeling, and `argmax(1)` collapses the one-hot rows back to a
single integer code per observation (recovers "which column is 1").

---

## 7. `cdf`, `pdf`, `prob`

```python
def cdf(self, x):  return self.distr.cdf(x)
def pdf(self, x):  return self.distr.pdf(x)
def prob(self, low, upp):
    return np.maximum(self.cdf(upp) - self.cdf(low), 0)
```
Thin wrappers around the chosen distribution `F`. `prob` computes the
interval probability `F(upp) − F(low)`, i.e. `P(low < u ≤ upp)`. The
`np.maximum(..., 0)` guards against tiny negative numbers from floating-point
error when `upp ≈ low` (probability can't legitimately be negative).

---

## 8. `transform_threshold_params` — the core reparameterization

```python
def transform_threshold_params(self, params):
    th_params = params[-(self.k_levels - 1):]
    thresh = np.concatenate((th_params[:1], np.exp(th_params[1:]))).cumsum()
    thresh = np.concatenate(([-np.inf], thresh, [np.inf]))
    return thresh
```
Take the last `K − 1` entries of the parameter vector (`th_params`), these
are the *optimizer's* internal, unconstrained representation of the
thresholds. Statistically, this implements:

```
a_1 = th_params[0]
a_2 = a_1 + exp(th_params[1])
a_3 = a_2 + exp(th_params[2])
  ...
```

Because `exp(·) > 0` always, each successive gap is strictly positive, so
`a_1 < a_2 < ... < a_{K-1}` is **guaranteed by construction**, no matter what
real-valued `th_params` the optimizer tries. This is a standard technique for
turning an ordering/monotonicity constraint into an unconstrained
optimization (compare: log-transforming a variance to keep it positive).
Finally `-∞` and `+∞` are appended as the outer bounds `a_0` and `a_K`, matching
the model definition `y=0 ⇔ y_latent ≤ a_1` and `y=K-1 ⇔ y_latent > a_{K-1}`.

## 9. `transform_reverse_threshold_params` — the inverse map

```python
thresh_params = np.concatenate((params[:1], np.log(np.diff(params[:-1]))))
```
Given real cutoffs `a_1 < ... < a_{K-1}` (`params`, with the ± infinities
dropped by `[:-1]` — actually here `params` is passed *without* the
appended infinities, just the finite thresholds), recovers the
unconstrained parameterization: keep `a_1` as is, and take `log` of each
successive difference `a_{k+1} − a_k` (which is `exp`'s inverse). Used only
to build a good **starting value** for optimization (Section 13), not during
the likelihood evaluation itself.

---

## 10. `predict`

```python
thresh = self.transform_threshold_params(params)
xb = self._linpred(params, exog=exog, offset=offset)
if which == "linpred":
    return xb
xb = xb[:, None]
low = thresh[:-1] - xb
upp = thresh[1:] - xb
if which == "prob":
    return self.prob(low, upp)
else:
    return self.cdf(upp)   # which in ("cum","cumprob")
```
For each observation, `low`/`upp` are `a_k − Xβ` and `a_{k+1} − Xβ` for
*every* `k` at once (broadcasting: `thresh` has length `K+1`, `xb` has shape
`(n, 1)`, so `low`/`upp` are `(n, K)` matrices). `self.prob(low, upp)` then
gives an `(n, K)` matrix of `P(y=k | x_i)` for all categories — this is the
full predictive distribution per observation. `"cumprob"` instead returns
`F(a_{k+1} − Xβ)`, the CDF, i.e. `P(y ≤ k | x)`.

---

## 11. `_linpred`

```python
linpred = _exog.dot(_params[: -(self.k_levels - 1)])
```
The regression parameters are simply the *first* `k_vars` entries of
`params` (everything before the last `K−1` threshold parameters); this is
`Xβ`, the systematic component of the latent variable, then `offset` is
added on: `Xβ + offset`.

---

## 12. `_bounds`

```python
thresh = self.transform_threshold_params(params)
thresh_i_low = thresh[self.endog]
thresh_i_upp = thresh[self.endog + 1]
xb = self._linpred(params)
low = thresh_i_low - xb
upp = thresh_i_upp - xb
```
This is the **observation-specific** version of the interval bounds used in
Section 10, but instead of computing bounds for *all* K categories per row,
it picks out only the two thresholds that bracket the *actually observed*
category `y_i` (`thresh[y_i]` and `thresh[y_i+1]`), using `self.endog`
(integers `0..K-1`) as an index array — a form of fancy/vectorized indexing.
This is exactly what the log-likelihood needs: `a_{y_i} − X_iβ` and
`a_{y_i+1} − X_iβ`.

---

## 13. `loglike` / `loglikeobs`

```python
def loglikeobs(self, params):
    low, upp = self._bounds(params)
    prob = self.prob(low, upp)
    return np.log(prob + 1e-20)

def loglike(self, params):
    return self.loglikeobs(params).sum()
```
Directly implements
`ℓ_i(β, a) = log[F(a_{y_i+1} − X_iβ) − F(a_{y_i} − X_iβ)]`
per observation, and the total log-likelihood `Σ ℓ_i` for optimization. The
`+ 1e-20` is a numerical safeguard against `log(0)` when a fitted probability
underflows to exactly zero during optimization (keeps the optimizer's
objective finite instead of `-inf`/`NaN`).

---

## 14. `score_obs_` — analytic gradient (partial)

```python
score_factor = (pdf_upp - pdf_low)[:, None]
score_factor /= prob[:, None]
so = np.column_stack((-score_factor[:, :1] * self.exog, score_factor[:, 1:]))
```
This is the analytic **score** (gradient of log-likelihood) with respect to
the regression coefficients only (the code comment admits the threshold
derivatives aren't correctly implemented for this exp-increment
parameterization — a known limitation, so by default the optimizer likely
falls back to numerical derivatives for thresholds).

The math: since `ℓ_i = log[F(upp) − F(low)]`,
```
∂ℓ_i/∂β = [f(upp) − f(low)] / [F(upp) − F(low)] · ∂(upp,low)/∂β
```
where `f` is the pdf. Because `low = a_k − Xβ` and `upp = a_{k+1} − Xβ`,
`∂low/∂β = ∂upp/∂β = −X`, giving
`∂ℓ_i/∂β = −X_i · (f(upp) − f(low)) / (F(upp) − F(low))`
— exactly `-score_factor * self.exog`, matching classic derivations of the
ordered probit/logit gradient (e.g. Greene's *Econometric Analysis*).

---

## 15. `start_params`

```python
freq = np.bincount(self.endog) / len(self.endog)
start_ppf = self.distr.ppf(np.clip(freq.cumsum(), 0, 1))
start_threshold = self.transform_reverse_threshold_params(start_ppf)
start_params = np.concatenate((np.zeros(self.k_vars), start_threshold))
```
A smart, closed-form starting point for the optimizer, based on the
**null model** (no regressors, `β=0`): if `β=0`, then `P(y ≤ k) = F(a_{k+1})`.
Empirically, `P(y ≤ k)` is just the observed cumulative frequency of
categories `0..k` (`freq.cumsum()`). Inverting `F` via the **percent-point
function / quantile function** (`ppf`, the inverse CDF) recovers the implied
thresholds `a_{k+1} = F⁻¹(cumulative frequency)` directly from the data — no
iteration needed. `np.clip(..., 0, 1)` guards against cumulative sums drifting
fractionally above 1 due to floating point. These raw thresholds are then
converted to the optimizer's exp-increment parameterization via the inverse
transform (Section 9), and regression coefficients start at zero.

---

## 16. `fit`

```python
fit_method = super().fit
mlefit = fit_method(start_params=..., method="nm", maxiter=500, ...)
ordmlefit = OrderedResults(self, mlefit)
ordmlefit.hasconst = 0
result = OrderedResultsWrapper(ordmlefit)
```
Delegates the actual numerical optimization to
`GenericLikelihoodModel.fit` (default optimizer `method="nm"` — the
**Nelder-Mead simplex algorithm**, a derivative-free method; a sensible
default given the noted incomplete analytic score for thresholds). Wraps the
raw MLE fit into the package's standard results object, explicitly marking
`hasconst = 0` since (unlike most linear models) there is deliberately no
constant term here.

---

## 17. `OrderedResults` — post-estimation statistics

- **`pred_table`**: builds a confusion matrix of predicted-vs-observed
  categories (`predicted = argmax` of predicted probabilities per row),
  cross-tabulated via `pd.crosstab`, with margins (row/column totals).
- **`llnull`**: log-likelihood of the intercept-only ("null") model — refits
  nothing, just evaluates `loglike` at the closed-form `start_params` (which
  *is* the MLE for the null/no-regressor model, since with only thresholds to
  fit, matching empirical cumulative frequencies exactly maximizes the
  likelihood).
- **`prsquared`**: **McFadden's pseudo-R²** = `1 − llf/llnull`, a common
  goodness-of-fit measure for MLE models lacking a natural R² (compares
  fitted log-likelihood `llf` to the null model's `llnull`; 0 = no
  improvement over intercept-only, closer to 1 = big improvement, though it
  rarely approaches 1 in practice).
- **`llr`**: the **likelihood-ratio test statistic** `-2(llnull − llf)`,
  asymptotically `χ²`-distributed under `H0: all β = 0`.
- **`llr_pvalue`**: p-value from the `χ²` survival function with degrees of
  freedom `k_vars` (number of restrictions being tested = number of
  regression coefficients).
- **`resid_prob`**: **probability-scale residuals** (Shepherd, Li & Liu
  2016; Li & Shepherd 2012), `P(Y < y) − P(Y > y)` for the observed
  category under the fitted distribution — a residual concept generalized to
  work uniformly across continuous, discrete, and ordinal outcomes, useful
  here because ordinary residuals (`observed − predicted`) aren't well
  defined for unordered-distance categorical data.

---

## Summary of the statistical throughline

1. **Model**: latent linear variable + threshold discretization ⇒ ordered
   probit/logit.
2. **Identification**: no intercept (absorbed into thresholds).
3. **Constrained optimization avoided**: thresholds via cumulative sum of
   `exp()`-transformed increments, guaranteeing monotonicity for any
   unconstrained parameter vector.
4. **Likelihood**: interval probability `F(upp) − F(low)` evaluated at the
   two thresholds bracketing each observation's actual category.
5. **Estimation**: generic MLE machinery (Nelder-Mead by default), warm-started
   from closed-form quantile-based thresholds of the null model.
6. **Inference**: standard MLE diagnostics — LR test, McFadden's pseudo-R²,
   confusion matrix, and a specialized ordinal residual.
