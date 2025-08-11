from math import ceil, sqrt

from scipy.stats import norm


def two_proportion_required_samples(
    alpha,
    power,
    control_rate,
    treatment_rate,
    control_to_treatment_ratio=1,
    tail_type="one_tail",
):
    """Calculate the required number of samples for two-proportion test.

    :param float alpha: Value in (0,1) that specifies desired False Positive Rate
    :param float power: Value in (0,1) that specifies desired 1 - False Negative Rate
    :param float control_rate: Value in (0,1) that specifies expected control rate
    :param float treatment_rate: Value in (0,1) that specifies expected treatment rate
    :param float control_to_treatment_ratio: The ratio of control to treatment samples
    :param string tail_type: Specifies one-tail or two-tail experiment\n
        Defaults to "two_tail" if anything other than "one_tail" is given
    :return: prob_tuple: Tuple of required control samples and treatment samples
    :rtype: tuple
    """
    alpha_adjusted = (
        alpha if tail_type is None or tail_type.lower() == "one_tail" else alpha / 2
    )
    beta = 1 - power

    ##Determine how extreme z-score must be to reach statistically significant result
    z_stat_critical = norm.ppf(1 - alpha_adjusted)
    ##Determine z-score of Treatment - Control that corresponds to a True Positive Rate (1-beta)
    z_power_critical = norm.ppf(beta)
    ##Expected Difference between treatment and control
    expected_delta = treatment_rate - control_rate
    ##Control Allocation Rate
    control_allocation = control_to_treatment_ratio / (control_to_treatment_ratio + 1)
    ##Treatment Allocation Rate
    treatment_allocation = 1 - control_allocation

    ##Calculate Variance of Treatment Rate - Control Rate
    blended_p = (
        treatment_rate * treatment_allocation + control_rate * control_allocation
    )
    blended_q = 1 - blended_p
    variance_blended = (
        blended_p * blended_q / (control_allocation * treatment_allocation)
    )
    ##Total Samples Required
    total_samples_required = (
        variance_blended
        * ((z_power_critical - z_stat_critical) / (0 - expected_delta)) ** 2
    )
    ##Split total samples into control and treatment
    control_samples = ceil(control_allocation * total_samples_required)
    treatment_samples = ceil(treatment_allocation * total_samples_required)
    ##Perhaps return required_delta in the future
    required_delta = z_stat_critical * sqrt(variance_blended) + 0
    return (control_samples, treatment_samples)
