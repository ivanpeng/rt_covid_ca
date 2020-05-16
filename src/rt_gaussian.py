import os
import requests
import pymc3 as pm
import pandas as pd
import numpy as np

from datetime import date
from datetime import datetime


from joblib import Parallel, delayed
from scipy import stats as sps
from scipy.stats import dirichlet
from scipy.interpolate import interp1d

from pygam import GammaGAM, PoissonGAM, s, l
from sklearn.utils import resample
from src import io

"""
This function essentially uses Bayes theorem to calculate Rt. We assume the number of new cases as a Poisson process,
and estimate lambda with a MLE function. From lambda we can infer Rt. We then leverage Bayesian jnference, starting with 
an original guess of Rt, and then using the previous's day posterior as the next day's prior to build a converging sense
of what Rt is for each province. 

There is some Gaussian smoothing + noise to remove bias and normalization in there. 
"""
# First define constants. Would love to move them into a job
# N_JOBS is parallelism
N_JOBS = 4

R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX * 100 + 1)

# Gamma is 1/serial interval
# https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article
# https://www.nejm.org/doi/full/10.1056/NEJMoa2001316
GAMMA = 1 / 7

# OPTIMAL_SIGMA = 0.35 # through Kevin's Optimization
OPTIMAL_SIGMA = 0.01


def run_full_model(cases, sigma=OPTIMAL_SIGMA):
    # initializing result dict
    result = {''}

    # smoothing series
    new, smoothed = smooth_new_cases(cases)

    # calculating posteriors
    posteriors, log_likelihood = calculate_posteriors(smoothed, sigma=sigma)

    # calculating HDI
    result = highest_density_interval(posteriors, p=.9)

    return result


def smooth_new_cases(new_cases):
    """
    Function to apply gaussian smoothing to cases
    Arguments
    ----------
    new_cases: time series of new cases
    Returns
    ----------
    smoothed_cases: cases after gaussian smoothing
    See also
    ----------
    This code is heavily based on Realtime R0
    by Kevin Systrom
    https://github.com/k-sys/covid-19/blob/master/Realtime%20R0.ipynb
    """

    smoothed_cases = new_cases.rolling(7,
                                       win_type='gaussian',
                                       min_periods=1,
                                       center=True).mean(std=2).round()

    zeros = smoothed_cases.index[smoothed_cases.eq(0)]
    if len(zeros) == 0:
        idx_start = 0
    else:
        last_zero = zeros.max()
        idx_start = smoothed_cases.index.get_loc(last_zero) + 1
    smoothed_cases = smoothed_cases.iloc[idx_start:]
    original = new_cases.loc[smoothed_cases.index]

    return original, smoothed_cases


def calculate_posteriors(sr, sigma=0.15):
    """
    Function to calculate posteriors of Rt over time
    Arguments
    ----------
    sr: smoothed time series of new cases
    sigma: gaussian noise applied to prior so we can "forget" past observations
           works like exponential weighting
    Returns
    ----------
    posteriors: posterior distributions
    log_likelihood: log likelihood given data
    See also
    ----------
    This code is heavily based on Realtime R0
    by Kevin Systrom
    https://github.com/k-sys/covid-19/blob/master/Realtime%20R0.ipynb
    """

    # (1) Calculate Lambda
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data=sps.poisson.pmf(sr[1:].values, lam),
        index=r_t_range,
        columns=sr.index[1:])

    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range,
                              scale=sigma
                              ).pdf(r_t_range[:, None])

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)

    # (4) Calculate the initial prior
    prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index=r_t_range,
        columns=sr.index,
        data={sr.index[0]: prior0}
    )

    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):
        # (5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]

        # (5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior

        # (5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)

        # Execute full Bayes' Rule
        posteriors[current_day] = numerator / denominator

        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)

    return posteriors, log_likelihood


def highest_density_interval(pmf, p=.9):
    """
    Function to calculate highest density interval
    from posteriors of Rt over time
    Arguments
    ----------
    pmf: posterior distribution of Rt
    p: mass of high density interval
    Returns
    ----------
    interval: expected value and density interval
    See also
    ----------
    This code is heavily based on Realtime R0
    by Kevin Systrom
    https://github.com/k-sys/covid-19/blob/master/Realtime%20R0.ipynb
    """

    # If we pass a DataFrame, just call this recursively on the columns
    if (isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                            index=pmf.columns)

    cumsum = np.cumsum(pmf.values)

    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]

    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()

    # Find the smallest range (highest density)
    best = (highs - lows).argmin()

    low = pmf.index[lows[best]]
    high = pmf.index[highs[best]]
    most_likely = pmf.idxmax(axis=0)

    interval = pd.Series([most_likely, low, high], index=['ML', f'Low_{p * 100:.0f}', f'High_{p * 100:.0f}'])

    return interval


def transform_df_to_dict(df):
    # Transform to JSON
    # Build a quick lookup map to standardize on formats
    province_lookup_map = {
        'Alberta': 'AB',
        'BC': 'BC',
        'Manitoba': 'MB',
        'NL': 'NL',
        'NWT': 'NWT',
        'New Brunswick': 'NB',
        'Nova Scotia': 'NS',
        'Ontario': 'ON',
        'PEI': 'PEI',
        'Quebec': 'QB',
        'Saskatchewan': 'SK',
        'Yukon': 'YT'
    }
    d = df.to_dict(orient='index')
    data = []
    for key in d:
        # key is a 2-tuple of province, timestamp
        # val is a dict of ML, Low_90, High_90
        # data needs to be an array of literally combination of all keys + values into one object
        province, ds = key
        # Discard anything before 2020-03-11
        if ds < datetime(2020, 3, 11, 0, 0, 0):
            continue
        val = d[key]
        obj = {'province': province_lookup_map[province],
               'date': int(ds.strftime('%s')),
               'ML': val['ML'],
               'Low_90': val['Low_90'],
               'High_90': val['High_90']
               }
        data.append(obj)
    return data


def run():
    # Assume data is already downloaded to the specific region
    file_path = "../data/linelist.csv"
    #io.download_can_case_file(filename=file_path)

    # Read data into pandas DF and do some simple filtering
    date_parser = lambda d: datetime.strptime(d, "%d-%m-%Y")

    df = pd.read_csv(file_path, parse_dates=['date_report'], date_parser=date_parser)
    # "case_id","provincial_case_id","age","sex","health_region","province","country","date_report","report_week","travel_yn","travel_history_country","locally_acquired","case_source","additional_info","additional_source","method_note"
    # Filter by province because data is not complete for some provinces
    df = df[~df['province'].isin(['Nunavut', 'Repatriated', 'Yukon', 'NWT', 'PEI'])]
    cases = df.groupby(['province', 'date_report']).count()[['case_id']].rename(
        columns={"case_id": 'confirmed_cases_by_day'})
    province_df = cases['confirmed_cases_by_day']
    with Parallel(n_jobs=N_JOBS) as parallel:
        results = parallel(delayed(run_full_model)(grp[1], sigma=0.01) for grp in province_df.groupby(level='province'))

    final_results = pd.concat(results)
    data = transform_df_to_dict(final_results)
    io.write_dict_to_file(data, 'whatever for now')


if __name__ == "__main__":
    run()
