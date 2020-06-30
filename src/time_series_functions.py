import pandas as pd
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt

def process_sector_file(sector, time_horizon):

    filename = sector + time_horizon + ".csv"
    df = pd.read_csv("../data/" + filename, header=[0,1,2])

    dates = df["Unnamed: 0_level_0", "Unnamed: 0_level_1", "Date"]
    closes = df.iloc[:, df.columns.get_level_values(1) == "Close"]
    companies = []
    for col in closes.columns:
        companies.append(col[0])

    closes_df = pd.DataFrame(closes.values, index=dates, columns=companies)

    # Static index with initial capital equally distributed amongst all stocks
    weights = np.empty(len(companies))
    for i, company in enumerate(companies):
        weights[i] = 1 / float(closes_df[company].iloc[0])
    weights /= sum(weights)

    sector_index_df = pd.DataFrame(np.zeros(len(dates)), index=dates, columns=[sector])
    for i, company in enumerate(companies):
        sector_index_df[sector] = sector_index_df[sector] + weights[i] * closes_df[company]

    return sector_index_df


def person_r_confidence_interval(r, n, alpha):

    def r_to_z(r_local):
        return np.log((1 + r_local) / (1 - r_local)) / 2.0

    def z_to_r(z_local):
        e = np.exp(2 * z_local)
        return (e - 1) / (e + 1)

    ci = np.empty([len(r), 2])
    for i, x in enumerate(r):
        z = r_to_z(x)
        se = 1.0 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(1 - alpha / 2)  # 2-tailed z critical value

        lo = z - z_crit * se
        hi = z + z_crit * se

        ci[i][0] = z_to_r(lo)
        ci[i][1] = z_to_r(hi)

    return np.transpose(ci)


def plot_series(df, pair, n, alpha):

    fig, ax = plt.subplots()
    x = df.index
    y = df[pair].values
    ci = person_r_confidence_interval(y, n, alpha)
    ax.plot(x, y)
    ax.fill_between(
        x,
        ci[0],
        ci[1],
        color='b',
        alpha=.1
    )
    plt.show()

def main():

    # inputs
    sectors = ["consumer", "financial", "industrial", "materials", "public"]
    time_horizons = ["5y", "10y"]
    n = 30  # moving window size
    alpha = 0.9

    # By sector index close levels
    by_sector_close_df = process_sector_file(sectors[0], time_horizons[0])
    for i, sector in enumerate(sectors):
        if i == 0:
            continue
        sector_index_df = process_sector_file(sectors[i], time_horizons[0])
        by_sector_close_df = pd.merge(
            by_sector_close_df,
            sector_index_df,
            how="inner",
            left_index=True,
            right_index=True
        )

    # By sector index return levels - assumed normal, i.e. index log-normal
    dates = by_sector_close_df.index
    returns = np.empty([len(sectors), len(dates)-1])
    for i, sector in enumerate(sectors):
        closes = by_sector_close_df[sector].values
        returns[i] = np.log(closes[1:]/closes[:len(closes)-1])
    by_sector_return_df = pd.DataFrame(np.transpose(returns), index=dates[1:], columns=sectors)

    # n day rolling window Pearson's r
    size_x = int((len(sectors)*(len(sectors)-1))/2)
    size_y = int(len(dates)-n-1)
    correlations = np.empty([size_x, size_y])
    pairs = []
    cont = 0
    for j, sector_1 in enumerate(sectors):
        for k, sector_2 in enumerate(sectors):
            if k < j:
                corr_by_date = np.empty((len(dates) - n - 1))
                for i in range(len(dates) - n - 1):
                    n_window_returns = by_sector_return_df.iloc[i:(i + n)]
                    corr_df = n_window_returns.corr()
                    corr_by_date[i] = corr_df[sector_1][sector_2]
                correlations[cont] = corr_by_date
                cont += 1
                pairs.append(sector_1 + "-" + sector_2)

    pairwise_n_day_pearson_r_df = pd.DataFrame(np.transpose(correlations), index=dates[n+1:], columns=pairs)
    pair = "financial-consumer"
    plot_series(pairwise_n_day_pearson_r_df, pair, n, alpha)

    return