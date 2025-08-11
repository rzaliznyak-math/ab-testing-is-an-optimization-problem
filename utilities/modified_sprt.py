import logging


import pandas as pd
from numpy import (
    add,
    divide,
    log,
    multiply,
    seterr,
    subtract,
    where,
    power,
    seterr,
    sqrt,
)
from scipy import stats
from stat_utils import pooled_standard_error



logging.basicConfig()
logger = logging.getLogger("sprt_utils")
logger.setLevel(logging.WARNING)
seterr(all="raise")


def llr_two_sample_ttest(
    control_mean,
    control_deviation,
    total_control_samples,
    treatment_mean,
    treatment_deviation,
    total_treatment_samples,
    index_to_control_planned,
):
    """Likelihood Ratio for two sample t-test
    LLR is the difference of log likelihoods of two competing hypothesis (Null vs alternative)
    The direction of LLR will allow us to reject Null, reject Alt, or results remain inconclusive.
    In this case:
    - Null Hypothesis; itc = 1.00.
    - Alternative Hypothesis; itc = index_to_control.
    The direction of log-lambda will allow us to reject Null, reject Alt, or results remain inconclusive.
    :param float_or_pandas.series control_mean: The experimental measured mean of control values
    :param float_or_pandas.series control_deviation: The experimental measured standard deviation of control values
    :param int_or_pandas.series total_control_samples: The number of samples upon which control mean is based
    :param float_or_pandas.series treatment_mean: The experimental measured mean of treatment values
    :param float_or_pandas.series treatment_deviation: The experimental measured standard deviation of treatment values
    :param int_or_pandas.series total_treatment_samples: The number of samples upon which treatment mean is based
    :param float_or_pandas.series index_to_control_planned: The minimum detectable effect we are looking for
    :return: llr (Log Likelihood Ratio)
    :rtype: pandas.series
    """

    try:
        difference_of_mean = subtract(treatment_mean, control_mean)

        ##Calculate 
        standard_error = pooled_standard_error(
            control_deviation,
            total_control_samples,
            treatment_deviation,
            total_treatment_samples,
        )

        ##Assume differnece of 0 for Null
        mu_null = 0

        ##Calculate Alternative Difference of Mean implied by index_to_control_planned
        mu_blended = divide(add(control_mean, treatment_mean), 2)
        mu_control = divide(
            multiply(mu_blended, add(total_control_samples, total_treatment_samples)),
            add(
                total_control_samples,
                multiply(index_to_control_planned, total_treatment_samples),
            ),
        )

        mu_treatment = multiply(mu_control, index_to_control_planned)
        mu_alt = subtract(mu_treatment, mu_control)

        ##Calculate Likelihood of Null
        #ll_null = log(stats.norm.pdf(difference_of_mean, 0, standard_error) + eps)
        #ll_alt = log(stats.norm.pdf(difference_of_mean, mu_alt, standard_error) + eps)

        ll_null = stats.norm.logpdf(difference_of_mean, loc=0, scale=standard_error)

        ll_alt = stats.norm.logpdf(difference_of_mean, loc=mu_alt, scale=standard_error)
        # Calculate LLR
        llr = subtract(ll_alt, ll_null)


    except FloatingPointError:
        logger.error(f"llr_two_sample_ttest: Cannot divide by Zero!!")
        return None
    except TypeError as e:
        logger.error(f"llr_two_sample_ttest Error: {e} - incorrect input data type!")
        return None
    else:
        logger.info(f"llr_two_sample_ttest calculated likelihood ratio for two sample t-test!")
        return pd.Series(llr)

def sprt_bounds(alpha=0.05, statistical_power=0.8):
    """Calculates the upper and lower cutoffs for an SPRT Analysis

    :param float alpha: The p-value threshold for rejection of Null (0.05 --> 5% False Positive Risk)
    :param float statistical_power: The power of experiment to detect true effects (0.80 --> 20% False Negative Risk)
    :return: (lower_cutoff, upper_cutoff)
    :rtype: tuple
    """

    try:
        beta = subtract(1, statistical_power)
        upper_cutoff = log(divide(statistical_power, alpha))
        lower_cutoff = log(divide(beta, (1 - alpha)))
    except FloatingPointError as e:
        logger.error(f"Error: {e} - cannot divide by Zero!!")
        return None
    except TypeError as e:
        logger.error(f"Error: {e} - incorrect input data type!")
        return None
    else:
        logger.info(f"calculated sprt boundry!")
        return (lower_cutoff, upper_cutoff)