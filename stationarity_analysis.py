from imports import *

def adf_stationarity_test(series, name="Time Series"):
    """
    Performs the Augmented Dickey-Fuller test on a time series.

    Parameters:
        series: list or 1D array-like, the reward series to test
        name: str, name for labeling the output

    Prints:
        ADF test statistic, p-value, and conclusion
    """
    print(f"\nADF Stationarity Test for {name}")
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]:.5f}")
    print(f"p-value: {result[1]:.5f}")
    for key, value in result[4].items():
        print(f"Critical Value {key}: {value:.5f}")

    if result[1] < 0.05:
        print(f"✅ Result: The {name} is likely stationary (reject null hypothesis).")
    else:
        print(f"❌ Result: The {name} is likely non-stationary (fail to reject null hypothesis).")

def kpss_test(series, name="Time Series"):
    print(f"\nKPSS Stationarity Test for {name}")
    statistic, p_value, _, crit_values = kpss(series, regression='c', nlags="auto")
    print(f"KPSS Statistic: {statistic:.5f}")
    print(f"p-value: {p_value:.5f}")
    for key, value in crit_values.items():
        print(f"Critical Value {key}: {value:.5f}")
    if p_value < 0.05:
        print(f"❌ Result: {name} is likely non-stationary (reject null hypothesis).")
    else:
        print(f"✅ Result: {name} is likely stationary.")


def rolling_stationarity_test(series, window_size=1000, step_size=500, verbose=False):
    """
    Runs ADF and KPSS tests over sliding windows of a time series.

    Parameters:
        series: list or array-like, the time series to test
        window_size: int, length of each rolling window
        step_size: int, step between windows
        verbose: bool, if True, prints stats for each window

    Returns:
        DataFrame with columns: ['start', 'end', 'adf_pvalue', 'kpss_pvalue']
    """
    results = []

    series = pd.Series(series)

    for start in range(0, len(series) - window_size + 1, step_size):
        end = start + window_size
        window = series[start:end]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InterpolationWarning)
                kpss_result = kpss(window, regression='c', nlags='auto')

            adf_result = adfuller(window, autolag="AIC")
        except Exception as e:
            if verbose:
                print(f"Window {start}-{end} failed: {e}")
            continue

        adf_p = adf_result[1]
        kpss_p = kpss_result[1]

        if verbose:
            print(f"Window {start}-{end}: ADF p={adf_p:.4f}, KPSS p={kpss_p:.4f}")

        results.append({
            "start": start,
            "end": end,
            "adf_pvalue": adf_p,
            "kpss_pvalue": kpss_p
        })

    return pd.DataFrame(results)

def plot_rolling_stationarity_pvalues(df, alpha=0.05, title_prefix=""):
    """
    Plots ADF and KPSS p-values from a rolling stationarity test.

    Parameters:
        df: DataFrame returned by `rolling_stationarity_test`
        alpha: significance level (default 0.05)
        title_prefix: optional prefix for plot titles
    """
    x = df["start"]

    plt.figure(figsize=(8, 4))

    plt.plot(x, df["adf_pvalue"], label="ADF p-value", color="tab:blue")
    plt.axhline(y=alpha, color="tab:blue", linestyle="--", alpha=0.5)

    plt.plot(x, df["kpss_pvalue"], label="KPSS p-value", color="tab:orange")
    plt.axhline(y=alpha, color="tab:orange", linestyle="--", alpha=0.5)

    plt.title(f"{title_prefix}Rolling ADF and KPSS p-values")
    plt.xlabel("Window Start Index")
    plt.ylabel("p-value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def test_stationarity_by_context(contexts, rewards, min_len=200):
    """
    Runs ADF and KPSS stationarity tests for each context with enough samples.

    Parameters:
        contexts: list of context values over time
        rewards: list of reward values over time
        min_len: minimum length to run stationarity test

    Returns:
        DataFrame with context, ADF p-value, KPSS p-value
    """
    results = []
    context_series = pd.Series(rewards).groupby(pd.Series(contexts))

    for context, series in context_series:
        if len(series) < min_len:
            continue

        try:
            adf_p = adfuller(series)[1]
        except Exception:
            adf_p = np.nan

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InterpolationWarning)
                kpss_p = kpss(series, regression='c', nlags='auto')[1]
        except Exception:
            kpss_p = np.nan

        results.append({"context": context, "adf_pvalue": adf_p, "kpss_pvalue": kpss_p})

    return pd.DataFrame(results).sort_values("context").reset_index(drop=True)
