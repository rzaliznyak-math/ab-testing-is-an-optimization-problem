import copy
import logging
import timeit
import time
from scipy.stats import norm
from modified_sprt import llr_two_sample_ttest, sprt_bounds
from bayesian_methods import expected_loss_or_gain
from power_methods import two_proportion_required_samples
from stat_utils import pooled_standard_error as pooled_se
import builtins
from numpy import (
    argmax,
    argmin,
    diff,
    add,
    array,
    cumsum,
    dot,
    empty,
    full,
    max,
    mean,
    min,
    minimum,
    percentile,
    maximum,
    repeat,
    median,
    sqrt,
    std,
    sum,
    where,
    zeros,
    newaxis,
)
from numpy.random import binomial, normal, seed
from scipy.stats import norm

logging.getLogger("bayesian_utils").setLevel(logging.WARNING)
import json



def expected_loss_delta_vectorized(delta_samples, risk_type="loss"):
    true_delta = delta_samples.mean(axis=-1)

    if risk_type == "loss":
        expected_loss = abs(minimum(delta_samples, 0).mean(axis=-1))
        expected_gain = true_delta + expected_loss
    else:
        expected_loss = abs(maximum(delta_samples, 0).mean(axis=-1))
        expected_gain = expected_loss - true_delta

    return expected_gain, expected_loss


def create_candidate_threshold_values(
    control_mean,
    control_deviation,
    index_to_control_planned,
    is_rate,
    metric_direction,
    number_conditions,
    allocation_list,
    max_volume_test,
    number_of_threshold_values=1000,
):
    """Return a list of candidate Risk Stopping Threshold Values
    :param float control_mean: The control average
    :param float control_deviation: The control deviation
    :param float index_to_control_planned: The control deviation
    :param float interval_width: The percentage of plausible effects contained within the interval
    :return: lower_bound , upper_bound
    :rtype: tuple
    """
    threshold_values = []
    NUMBER_POSTERIOR_SAMPLES = int(1e5)
    total_traffic = sum(allocation_list)
    posterior_samples_list = []
    for i in range(number_conditions):
        condition_trials = int(allocation_list[i] * max_volume_test / total_traffic)
        condition_mean = (
            control_mean
            if i < number_conditions - 1
            else control_mean * index_to_control_planned
        )
        if is_rate:
            condition_deviation = (
                control_deviation
                if i < number_conditions - 1
                else sqrt(condition_mean * (1 - condition_mean))
            )
        else:
            condition_deviation = control_deviation
        condition_se = condition_deviation / sqrt(condition_trials)
        condition_posterior_samples = normal(
            condition_mean, condition_se, NUMBER_POSTERIOR_SAMPLES
        )
        posterior_samples_list.append(condition_posterior_samples)

    expected_loss_or_gain_tuple = expected_loss_or_gain(
        posterior_samples_list,
        "loss" if metric_direction == "increase" else "gain",
    )

    max_threshold = max(expected_loss_or_gain_tuple)
    increment = max_threshold / (1 * number_of_threshold_values)
    for i in range(number_of_threshold_values):
        threshold_values.append(i * increment)
    return threshold_values


def analyze_experiment_simulations_fixed(list_of_experiment_simulations):
    start = timeit.default_timer()
    number_simulations = len(list_of_experiment_simulations)
    number_conditions = list_of_experiment_simulations[0].number_conditions
    max_volume_year = list_of_experiment_simulations[0].max_volume_year

    total_potential_sum = []
    total_winner_rate = []

    for i in range(number_simulations):
        a_simulation = list_of_experiment_simulations[i]
        total_potential_sum.append(a_simulation.fixed_potential_sum)
        total_winner_rate.append(
            1 if a_simulation.fixed_winning_condition == number_conditions - 1 else 0
        )
    return {
        "fixed_winner_rate": mean(total_winner_rate),
        "fixed_analysis_run_time": timeit.default_timer() - start,
        "fixed_potential_mean": mean(total_potential_sum) / max_volume_year,
        "fixed_potential": array(total_potential_sum) / max_volume_year,
    }


def analyze_experiment_simulations_sequential(list_of_experiment_simulations):
    start = timeit.default_timer()
    number_simulations = len(list_of_experiment_simulations)
    number_conditions = list_of_experiment_simulations[0].number_conditions
    # threshold_values = list_of_experiment_simulations[0].threshold_values
    max_volume_year = list_of_experiment_simulations[0].max_volume_year

    total_run_time = []
    total_potential_sum = []
    total_winner_rate = []
    total_loser_rate = []
    for i in range(number_simulations):
        a_simulation = list_of_experiment_simulations[i]
        total_run_time.append(a_simulation.sequential_winning_time)
        total_potential_sum.append(a_simulation.sequential_potential_sum)
        total_winner_rate.append(
            1
            if a_simulation.sequential_final_winning_condition == number_conditions - 1
            else 0
        )
        total_loser_rate.append(a_simulation.lower_bound_crossed)
        pass
    return {
        "sequential_winner_rate": mean(total_winner_rate),
        "sequential_loser_rate": mean(total_loser_rate),
        "sequential_run_times": mean(total_run_time),
        "sequential_analysis_run_time": timeit.default_timer() - start,
        "sequential_potential_mean": mean(total_potential_sum) / max_volume_year,
        "sequential_potential": array(total_potential_sum) / max_volume_year,
    }


def analyze_experiment_simulations_bayesian(
    list_of_experiment_simulations, metric_direction="increase"
):
    start = timeit.default_timer()
    number_simulations = len(list_of_experiment_simulations)
    number_conditions = list_of_experiment_simulations[0].number_conditions
    # threshold_values = list_of_experiment_simulations[0].threshold_values
    max_volume_year = list_of_experiment_simulations[0].max_volume_year
    for i in range(number_simulations):
        # temp_simulation = copy.copy(list_of_experiment_simulations[i])
        temp_simulation = list_of_experiment_simulations[i]
        if i == 0:
            all_sums = array(temp_simulation.potential_sum_at_each_ths)
            # all_sums_breach = array(temp_simulation.potential_sum_at_each_ths_breach)
            run_times = array(temp_simulation.winning_time_at_each_ths)
            true_winners = where(
                array(temp_simulation.winning_condition_at_each_ths)
                == number_conditions - 1,
                1,
                0,
            )
            # true_winners_breach = where(
            #    array(temp_simulation.winning_condition_at_each_ths_breach)
            #    == number_conditions - 1,
            #    1,
            #    0,
            # )
        else:
            all_sums += array(temp_simulation.potential_sum_at_each_ths)
            # all_sums_breach += array(temp_simulation.potential_sum_at_each_ths_breach)
            run_times += array(temp_simulation.winning_time_at_each_ths)
            true_winners += where(
                array(temp_simulation.winning_condition_at_each_ths)
                == number_conditions - 1,
                1,
                0,
            )
            # true_winners_breach += where(
            #    array(temp_simulation.winning_condition_at_each_ths_breach)
            #    == number_conditions - 1,
            #    1,
            #    0,
            # )

    ##CALCULATE MEDIANS AND 80th percentiles here
    all_sums = all_sums / number_simulations
    all_means = all_sums / max_volume_year

    # all_sums_breach = all_sums_breach / number_simulations
    # all_means_breach = all_sums_breach / max_volume_year

    run_times = run_times / number_simulations
    true_winners = true_winners / number_simulations
    # true_winners_breach = true_winners_breach / number_simulations

    run_time_lists = []

    # potential_mean_at_risk_level = []
    for i in range(number_simulations):
        temp_simulation = list_of_experiment_simulations[i]
        winning_time_at_each_ths = temp_simulation.winning_time_at_each_ths
        run_time_lists.append(winning_time_at_each_ths)

    run_time_array = array(run_time_lists)  # shape: [simulations, thresholds]

    # Per-threshold median and 80th percentile run time
    median_run_times = median(run_time_array, axis=0)
    percentile_80_run_times = percentile(run_time_array, 80, axis=0)
    return {
        "all_means": all_means,
        # "all_means_breach": all_means_breach,
        "run_times": run_times,
        "winner_rate": true_winners,
        "bayesian_analysis_run_time": timeit.default_timer() - start,
        "metric_direction": metric_direction,
        "median_run_times": median_run_times,
        "percentile_80_run_times": percentile_80_run_times,
    }


class experimentSimulationObject:
    # constructor function
    def __init__(
        self,
        control_mean,
        control_deviation,
        index_to_control_planned,
        is_rate,
        number_conditions,
        allocation_list,
        min_number_test_days,
        max_number_test_days,
        max_volume_test,
        max_volume_year,
    ):
        start = timeit.default_timer()

        condition_mean_list = []
        condition_trial_list = []
        total_traffic = sum(allocation_list)
        for i in range(number_conditions):
            condition_trials = int(allocation_list[i] * max_volume_test / total_traffic)
            condition_trial_list.append(condition_trials)
            condition_mean = (
                control_mean * index_to_control_planned
                if (i == number_conditions - 1)
                else control_mean
            )
            condition_mean_list.append(condition_mean)

        trials_matrix = []
        mean_matrix = []
        sum_matrix = []
        cumulative_sum_matrix = []
        cumulative_trials_matrix = []
        cumulative_mean_matrix = []
        cumulative_se_matrix = []

        for k in range(number_conditions):
            condition_mean = condition_mean_list[k]
            condition_increment = int(condition_trial_list[k] / max_number_test_days)

            condition_trials = full(max_number_test_days, condition_increment)
            trials_matrix.append(condition_trials)

            cumulative_condition_trials = cumsum(condition_trials)
            cumulative_trials_matrix.append(cumulative_condition_trials)

            if is_rate:
                condition_events = binomial(
                    condition_increment, condition_mean, max_number_test_days
                )
                sum_matrix.append(condition_events)
                condition_means = condition_events / condition_trials
                mean_matrix.append(condition_means)

                cumulative_condition_events = cumsum(condition_events)
                cumulative_sum_matrix.append(cumulative_condition_events)
                cumulative_condition_means = (
                    cumulative_condition_events / cumulative_condition_trials
                )
                cumulative_mean_matrix.append(cumulative_condition_means)

                cumulative_condition_errors = (
                    (cumulative_condition_means * (1 - cumulative_condition_means))
                    ** 0.5
                ) / (cumulative_condition_trials**0.5)
                cumulative_se_matrix.append(cumulative_condition_errors)
            else:
                """
                condition_outcomes = normal(
                    condition_mean,
                    control_deviation,
                    condition_increment * max_number_test_days,
                )
                

                #cumulative_condition_trials = array(cumulative_condition_trials)

                # 1. Compute daily start indices and lengths
                #daily_lengths = diff(
                #    [0] + cumulative_condition_trials.tolist()
                #)  # daily trial counts
                start_indices = array([0] + cumulative_condition_trials[:-1].tolist())

                # 2. Daily sums and means (fully vectorized)
                condition_sums = add.reduceat(condition_outcomes, start_indices)
                #condition_means = condition_sums / daily_lengths
                condition_means = condition_sums / condition_trials

                # 3. Cumulative means and standard errors
                cumulative_condition_means, cumulative_condition_stds = (
                    cumulative_mean_std(condition_outcomes, cumulative_condition_trials)
                )
                cumulative_condition_errors = cumulative_condition_stds / sqrt(
                    cumulative_condition_trials
                )
                """
                

                # Rather than estimate condition means through direct outcomes, let's simulate the average directly
                
                expected_se_std = control_deviation/sqrt(2*(condition_increment-1))
                condition_means = normal(condition_mean, control_deviation/sqrt(condition_increment), max_number_test_days) 
                condition_stds = normal(control_deviation, expected_se_std, max_number_test_days)
                condition_sums = condition_trials * condition_means


                weighted_cumulative_sums = cumsum(condition_means * condition_trials)
                cumulative_condition_means = weighted_cumulative_sums / cumulative_condition_trials

                numerator = cumsum((condition_trials - 1) * condition_stds**2)
                denominator = cumsum(condition_trials - 1)
                cumulative_condition_stds = sqrt(numerator / maximum(denominator, 1))
                cumulative_condition_errors = cumulative_condition_stds / sqrt(
                    cumulative_condition_trials
                )
                
            
                


            
                
                
                sum_matrix.append(condition_sums)
                cumulative_sum_matrix.append(cumsum(condition_sums))
                mean_matrix.append(condition_means)
                cumulative_mean_matrix.append(cumulative_condition_means)
                cumulative_se_matrix.append(cumulative_condition_errors)

        #print(std(condition_sums_0))
        #print(cumulative_condition_stds)
        self.run_time = timeit.default_timer() - start
        self.trials_matrix = trials_matrix
        self.cumulative_trials_matrix = cumulative_trials_matrix
        self.mean_matrix = mean_matrix
        self.sum_matrix = sum_matrix
        self.cumulative_sum_matrix = cumulative_sum_matrix
        self.cumulative_mean_matrix = cumulative_mean_matrix
        self.cumulative_se_matrix = cumulative_se_matrix
        self.number_conditions = number_conditions
        self.min_number_test_days = min_number_test_days
        self.max_number_test_days = max_number_test_days
        self.max_volume_test = max_volume_test
        self.max_volume_year = max_volume_year
        self.condition_mean_list = condition_mean_list
        self.index_to_control_planned = index_to_control_planned

        # temp_trials_matrix = trials_matrix
        # temp_sum_matrix = sum_matrix
        # for i in range(len(trials_matrix)):
        #    if i == 0:
        #        all_trials = temp_trials_matrix[i]
        #        all_sums = temp_sum_matrix[i]
        #    else:
        #        all_trials += temp_trials_matrix[i]
        #        all_sums += temp_sum_matrix[i]

        all_trials = sum(trials_matrix, axis=0)
        all_sums = sum(sum_matrix, axis=0)

        # Add total_mean and cumulative_total_mean
        self.all_trials = all_trials
        self.cumulative_all_trials = cumsum(all_trials)
        self.all_sums = all_sums
        self.cumulative_all_sums = cumsum(all_sums)
        self.expected_loss_gain_run_time = None
        self.expected_loss_matrix = None
        self.expected_gain_matrix = None
        self.cumulative_potential_sum_matrix = None

    def calculate_z_scores(self, actual_fixed_days=None):
        start = timeit.default_timer()
        z_scores = [0]
        decision_position = self.max_number_test_days - 1
        if actual_fixed_days is not None:
            decision_position = actual_fixed_days - 1
        control_mean = self.cumulative_mean_matrix[0][decision_position]
        control_trials = self.cumulative_trials_matrix[0][decision_position]
        control_std = self.cumulative_se_matrix[0][decision_position] * sqrt(
            control_trials
        )
        for k in range(1, self.number_conditions):
            condition_mean = self.cumulative_mean_matrix[k][decision_position]
            condition_trials = self.cumulative_trials_matrix[k][decision_position]
            condition_std = self.cumulative_se_matrix[k][decision_position] * sqrt(
                condition_trials
            )

            pooled_standard_error = float(
                pooled_se(
                    control_std, control_trials, condition_std, condition_trials
                )
            )

            z_score = (condition_mean - control_mean) / pooled_standard_error

            z_scores.append(z_score)

        self.z_scores = z_scores
        self.z_run_time = timeit.default_timer() - start

    def calculate_llr_old(self, index_to_control_test=None):
        start = timeit.default_timer()
        llr_matrix = []
        index_to_control_planned = (
            self.index_to_control_planned
            if index_to_control_test is None
            else index_to_control_test
        )
        llr_matrix.append(list(full(0, self.max_number_test_days)))
        for k in range(1, self.number_conditions):
            condition_mean_array = self.cumulative_mean_matrix[k]
            condition_trials_array = self.cumulative_trials_matrix[k]
            condition_std_array = self.cumulative_se_matrix[k] * sqrt(
                condition_trials_array
            )

            control_mean_array = self.cumulative_mean_matrix[0]
            control_trials_array = self.cumulative_trials_matrix[0]
            control_std_array = self.cumulative_se_matrix[0] * sqrt(
                control_trials_array
            )
            llr = llr_two_sample_ttest(
                control_mean_array,
                control_std_array,
                control_trials_array,
                condition_mean_array,
                condition_std_array,
                condition_trials_array,
                index_to_control_planned,
            )
            llr[0 : self.min_number_test_days - 1] = 0
            llr_matrix.append(llr)

        stop = timeit.default_timer()

        self.llr_matrix = llr_matrix
        self.llr_run_time = stop - start

    def calculate_llr(self, index_to_control_test=None):
        start = timeit.default_timer()
        llr_matrix = zeros((self.number_conditions, self.max_number_test_days))
        index_to_control_planned = (
            self.index_to_control_planned
            if index_to_control_test is None
            else index_to_control_test
        )

        eps = 1e-10
        control_mean_array = self.cumulative_mean_matrix[0] + eps
        control_trials_array = self.cumulative_trials_matrix[0] + eps
        control_std_array = (
            self.cumulative_se_matrix[0] * sqrt(control_trials_array) + eps
        )
        for k in range(1, self.number_conditions):
            condition_mean_array = self.cumulative_mean_matrix[k] + eps
            condition_trials_array = self.cumulative_trials_matrix[k] + eps
            condition_std_array = (
                self.cumulative_se_matrix[k] * sqrt(condition_trials_array) + eps
            )

            llr = llr_two_sample_ttest(
                control_mean_array,
                control_std_array,
                control_trials_array,
                condition_mean_array,
                condition_std_array,
                condition_trials_array,
                index_to_control_planned,
            )
            llr[0 : self.min_number_test_days - 1] = 0
            llr_matrix[k] = llr

        self.llr_matrix = llr_matrix
        self.llr_run_time = timeit.default_timer() - start

    def calculate_expected_loss_new(self, which):
        start = timeit.default_timer()
        expected_loss_matrix = []
        expected_gain_matrix = []

        # Ensure inputs are NumPy arrays (safe if they're already arrays)
        mean_matrix = array(self.cumulative_mean_matrix)
        se_matrix = array(self.cumulative_se_matrix)

        min_day = self.min_number_test_days - 1
        max_day = self.max_number_test_days
        n_days = max_day - min_day
        n_conditions = self.number_conditions

        # Extract control group mean and SE across valid days
        control_mean = mean_matrix[0, min_day:max_day]  # shape: [n_days]
        control_se = se_matrix[0, min_day:max_day]  # shape: [n_days]

        # Extract treatment condition stats (excluding control)
        condition_means = mean_matrix[
            1:, min_day:max_day
        ]  # shape: [n_conditions-1, n_days]
        condition_ses = se_matrix[
            1:, min_day:max_day
        ]  # shape: [n_conditions-1, n_days]

        # Compute deltas
        delta_means = condition_means - control_mean  # broadcasted over rows
        delta_ses = sqrt(condition_ses**2 + control_se**2)  # broadcasted

        # Analytical expected gain/loss
        e_gain, e_loss = expected_loss_delta_normal(
            delta_means, delta_ses, risk_type=which
        )

        # Build tuples with control risk (max gain at each day)
        if which != "gain":
            loss_tuples = [
                tuple([max(e_gain[:, i])] + list(e_loss[:, i])) for i in range(n_days)
            ]
        if which != "loss":
            gain_tuples = [
                tuple([max(e_gain[:, i])] + list(e_loss[:, i])) for i in range(n_days)
            ]

        # Initialize final matrices
        for k in range(n_conditions):
            # condition_losses = empty(self.max_number_test_days)
            # condition_gains = empty(self.max_number_test_days)

            condition_losses = full(self.max_number_test_days, 1e17)
            condition_gains = full(self.max_number_test_days, 1e17)

            for j in range(self.max_number_test_days):
                if j < self.min_number_test_days - 1:
                    pass
                    # if which != "gain":
                    #    condition_losses[j] = 1e17
                    # if which != "loss":
                    #    condition_gains[j] = 1e17
                else:
                    idx = j - (self.min_number_test_days - 1)
                    if which != "gain":
                        condition_losses[j] = loss_tuples[idx][k]
                    if which != "loss":
                        condition_gains[j] = gain_tuples[idx][k]

            if which != "gain":
                expected_loss_matrix.append(condition_losses)
            if which != "loss":
                expected_gain_matrix.append(condition_gains)

        # Store results
        if which != "gain":
            self.expected_loss_matrix = expected_loss_matrix
        if which != "loss":
            self.expected_gain_matrix = expected_gain_matrix

        self.loss_run_time = timeit.default_timer() - start

    def calculate_expected_loss(self, which, number_posterior_simulations=1000):
        start = timeit.default_timer()

        expected_loss_matrix = []
        expected_gain_matrix = []
        loss_tuples = []
        gain_tuples = []

        for j in range(self.min_number_test_days - 1, self.max_number_test_days):
            # posterior_samples_list = []
            generic_tuple_0 = []
            control_risks = []

            control_mean = self.cumulative_mean_matrix[0][j]
            control_se = self.cumulative_se_matrix[0][j]

            for k in range(1, self.number_conditions):
                condition_mean = self.cumulative_mean_matrix[k][j]
                condition_se = self.cumulative_se_matrix[k][j]

                delta_mean = condition_mean - control_mean
                delta_se = (condition_se**2 + control_se**2) ** 0.5
                delta_samples = normal(
                    delta_mean, delta_se, number_posterior_simulations
                )
                e_gain, e_loss = expected_loss_delta_vectorized(
                    delta_samples, risk_type=which
                )

                generic_tuple_0.append(e_loss)
                control_risks.append(e_gain)

            generic_tuple = tuple([builtins.max(control_risks)] + generic_tuple_0)

            if which != "gain":
                loss_tuples.append(generic_tuple)
            if which != "loss":
                gain_tuples.append(generic_tuple)

        for k in range(self.number_conditions):
            condition_losses = empty(self.max_number_test_days)
            condition_gains = empty(self.max_number_test_days)

            for j in range(self.max_number_test_days):
                if j < self.min_number_test_days - 1:
                    if which != "gain":
                        condition_losses[j] = 1e17
                    if which != "loss":
                        condition_gains[j] = 1e17
                else:
                    index = j - (self.min_number_test_days - 1)
                    if which != "gain":
                        condition_losses[j] = loss_tuples[index][k]
                    if which != "loss":
                        condition_gains[j] = gain_tuples[index][k]

            if which != "gain":
                expected_loss_matrix.append(condition_losses)
            if which != "loss":
                expected_gain_matrix.append(condition_gains)

        if which != "gain":
            self.expected_loss_matrix = expected_loss_matrix
        if which != "loss":
            self.expected_gain_matrix = expected_gain_matrix

        stop = timeit.default_timer()
        self.loss_run_time = stop - start

    def update_potential_outcomes_at_each_test_day(self):
        start = timeit.default_timer()

        cumulative_potential_sum_matrix = zeros(
            (self.number_conditions, self.max_number_test_days)
        )

        # all_trials_cumsum = cumsum(self.all_trials[::-1])[
        #    ::-1
        # ]  # Cumulative sum from the end
        for j in range(self.number_conditions):
            condition_potential_sums = array(self.cumulative_all_sums, dtype=float)

            mean_values = self.mean_matrix[j]

            # Compute dot product only once for efficiency
            # condition_potential_sums[:self.min_test_days] = 0
            for k in range(
                self.min_number_test_days - 1, self.max_number_test_days - 1
            ):
                condition_potential_sums[k] += dot(
                    mean_values[k + 1 :], self.all_trials[k + 1 :]
                )

            # Adjust potential sum by mean volume difference
            condition_potential_sums += self.condition_mean_list[j] * (
                self.max_volume_year - self.max_volume_test
            )

            cumulative_potential_sum_matrix[j] = condition_potential_sums
        self.cumulative_potential_sum_matrix = cumulative_potential_sum_matrix
        self.potential_sum_run_time = timeit.default_timer() - start

    def calculate_fixed_stopping_values(
        self, z_critical=1.65, test_direction="increase", actual_fixed_days=None
    ):
        start = timeit.default_timer()
        decision_position = self.max_number_test_days - 1
        if actual_fixed_days is not None:
            decision_position = actual_fixed_days - 1
        # z_critical = (
        #    norm.ppf(1 - alpha)
        #    if test_direction == "increase"
        #    else norm.ppf(alpha)
        # )

        if test_direction == "increase":
            stopping_array = where(self.z_scores >= z_critical)[0]
        else:
            stopping_array = where(self.z_scores <= z_critical)[0]

        winning_condition = 0
        if len(stopping_array) == 0:
            pass
        else:
            if test_direction == "increase":
                max_z_score = max(self.z_scores)
                winning_condition = where(self.z_scores == max_z_score)[0][0]
            else:
                min_z_score = min(self.z_scores)
                winning_condition = where(self.z_scores == min_z_score)[0][0]

        self.fixed_potential_sum = self.cumulative_potential_sum_matrix[
            winning_condition
        ][decision_position]
        self.fixed_winning_condition = winning_condition
        self.fixed_stopping_details_run_time = timeit.default_timer() - start

    def calculate_sequential_stopping_values(self, alpha=0.05, power=0.80):
        start = timeit.default_timer()
        (
            lower_cutoff,
            upper_cutoff,
        ) = sprt_bounds(  # Calculate SPRT Lower and Upper Bounds
            alpha=alpha,
            statistical_power=power,
        )

        first_sig_days = [self.max_number_test_days + 100]
        upper_bound_crossed_list = [False]
        lower_bound_crossed_list = [False]
        llr_list = [0]
        self.sequential_stopping_matrix = []
        for j in range(1, self.number_conditions):
            first_sig_day = None
            first_upper_day = None
            first_lower_day = None
            llr = self.llr_matrix[j]
            upper_stopping_array = where(llr >= upper_cutoff)[0]
            lower_stopping_array = where(llr <= lower_cutoff)[0]

            if len(upper_stopping_array) > 0:
                first_upper_day = upper_stopping_array[0]
            if len(lower_stopping_array) > 0:
                first_lower_day = lower_stopping_array[0]

            upper_bound_crossed = False
            lower_bound_crossed = False
            if first_upper_day is not None:
                if first_lower_day is not None:
                    if first_lower_day < first_upper_day:
                        first_sig_day = first_lower_day
                        lower_bound_crossed = True
                    else:
                        first_sig_day = first_upper_day
                        upper_bound_crossed = True
                else:
                    first_sig_day = first_upper_day
                    upper_bound_crossed = True
            else:
                if first_lower_day is not None:
                    first_sig_day = first_lower_day
                    lower_bound_crossed = True

            first_sig_days.append(first_sig_day)
            lower_bound_crossed_list.append(lower_bound_crossed)
            upper_bound_crossed_list.append(upper_bound_crossed)
            llr_list.append(
                llr[first_sig_day if first_sig_day else self.max_number_test_days - 1]
            )

        winning_conditions = where(array(upper_bound_crossed_list) == True)[0]
        winning_time_per_condition = [first_sig_days[i] for i in winning_conditions]

        final_winning_conditions = None
        final_winning_condition = None
        winning_time = None
        if len(winning_time_per_condition) > 0:
            winning_time = min(winning_time_per_condition)
            final_winning_conditions = where(first_sig_days == winning_time)[0]

        llr_initial = None
        if final_winning_conditions is not None:
            for possible_winning_condition in final_winning_conditions:
                llr_value = llr_list[possible_winning_condition]
                if llr_initial is None:
                    llr_initial = llr_value
                    final_winning_condition = possible_winning_condition
                elif llr_initial < llr_value:
                    llr_initial = llr_value
                    final_winning_condition = possible_winning_condition

        if False not in lower_bound_crossed_list[1:]:
            winning_time = max(first_sig_days[1:])
        winning_time = winning_time if winning_time else self.max_number_test_days - 1
        self.sequential_potential_sum = self.cumulative_potential_sum_matrix[
            final_winning_condition if final_winning_condition else 0
        ][winning_time]
        self.sequential_final_winning_condition = final_winning_condition
        self.sequential_winning_time = winning_time
        self.lower_bound_crossed = lower_bound_crossed_list[self.number_conditions - 1]
        self.sequential_stopping_details_run_time = timeit.default_timer() - start

    def calculate_optimal_stopping_values(self, threshold_values, type):
        self.threshold_values = threshold_values
        start = timeit.default_timer()
        winning_condition_at_each_ths = []
        winning_time_at_each_ths = []
        potential_sum_at_each_ths = []

        for ths in threshold_values:
            stopping_position = None
            all_risk_arrays = []
            for j in range(self.number_conditions):
                current_risk_array = (
                    self.expected_loss_matrix[j]
                    if type == "loss"
                    else self.expected_gain_matrix[j]
                )
                all_risk_arrays.append(current_risk_array)
                if j == 0:
                    minimum_array = current_risk_array
                else:
                    minimum_array = minimum(minimum_array, current_risk_array)
            breach_or_not_array = where(minimum_array < ths)[0]
            if len(breach_or_not_array) > 0:
                stopping_position = breach_or_not_array[0]

            final_stopping_position = (
                stopping_position
                if stopping_position
                else self.max_number_test_days - 1
            )
            for j in range(self.number_conditions):
                if (
                    minimum_array[final_stopping_position]
                    == all_risk_arrays[j][final_stopping_position]
                ):
                    winning_condition = j if stopping_position is not None else j
                    winning_condition_at_each_ths.append(winning_condition)

                    # winning_condition_breach = j if stopping_position is not None else 0
                    # winning_condition_at_each_ths_breach.append(
                    #    winning_condition_breach
                    # )

                    winning_time_at_each_ths.append(
                        self.cumulative_all_trials[final_stopping_position]
                    )

                    potential_sum_at_each_ths.append(
                        self.cumulative_potential_sum_matrix[winning_condition][
                            final_stopping_position
                        ]
                    )
                    # potential_sum_at_each_ths_breach.append(
                    #    self.cumulative_potential_sum_matrix[winning_condition_breach][
                    #        final_stopping_position
                    #    ]
                    # )
                    break
        self.winning_condition_at_each_ths = winning_condition_at_each_ths
        # self.winning_condition_at_each_ths_breach = winning_condition_at_each_ths_breach
        self.winning_time_at_each_ths = winning_time_at_each_ths
        self.potential_sum_at_each_ths = potential_sum_at_each_ths
        # self.potential_sum_at_each_ths_breach = potential_sum_at_each_ths_breach

        self.optimal_stopping_details_run_time = timeit.default_timer() - start

    def to_dict(self):
        inner_dict = {
            "winning_condition_at_each_ths": self.winning_condition_at_each_ths
        }
        return inner_dict
