import logging

from numpy import array, maximum, mean, minimum, percentile, random, where

logging.basicConfig()
logger = logging.getLogger("bayesian_utils")
logger.setLevel(logging.INFO)

def expected_loss_or_gain(list_of_posterior_samples_arrays, type="loss"):
    """Calculate the expected loss or gain for any one condition.
    Takes a list or array that contains posterior samples from any number of conditions to calculate expected loss or gain.

    :param list list_of_posterior_samples_arrays: A list that contains arrays of posterior samples from any number of conditions
    :param string type: "loss" returns expected loss any condition may generate. Otherwise, returns expected gain
    :return: loss_or_gain_tuple: Tuple of expected loss or gain for each condition
    :rtype: tuple
    """
    number_conditions = len(list_of_posterior_samples_arrays)
    if number_conditions < 2:
        logger.error("Error: At least two conditions are required")
        return None
    number_samples_list = []
    for i in range(number_conditions):
        number_samples = len(list_of_posterior_samples_arrays[i])
        number_samples_list.append(number_samples)
    number_sample_set = set(number_samples_list)
    if len(number_sample_set) > 1:
        logger.error(
            "Error: Variable number of posterior samples per condition present"
        )
        return None

    loss_or_gain_tuple = []
    ##Depending on Type, calcs the max or min value at each row for all arrays
    ##max_or_min array will be compared to each condition's samples
    max_or_min_array = array(list_of_posterior_samples_arrays[0])
    for j in range(1, number_conditions):
        if type.lower() == "loss":
            max_or_min_array = maximum(
                max_or_min_array, array(list_of_posterior_samples_arrays[j])
            )
        else:
            max_or_min_array = minimum(
                max_or_min_array, array(list_of_posterior_samples_arrays[j])
            )

    for k in range(number_conditions):
        max_or_min_delta_condition_array = max_or_min_array - array(
            list_of_posterior_samples_arrays[k]
        )
        if type.lower() == "loss":
            expected_loss = mean(
                where(
                    max_or_min_delta_condition_array >= 0,
                    max_or_min_delta_condition_array,
                    0,
                )
            )
            loss_or_gain_tuple.append(expected_loss)
        else:
            expected_increase = mean(
                where(
                    max_or_min_delta_condition_array <= 0,
                    abs(max_or_min_delta_condition_array),
                    0,
                )
            )
            loss_or_gain_tuple.append(expected_increase)

    logger.info(
        "Calculated expected_{0}_tuple".format(
            "loss" if type.lower() == "loss" else "gain"
        )
    )
    return tuple(loss_or_gain_tuple)