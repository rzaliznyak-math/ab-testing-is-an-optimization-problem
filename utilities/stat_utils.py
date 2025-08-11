import logging

import pandas as pd
from numpy import add, divide, multiply, power, seterr, sqrt, subtract

logging.basicConfig()
logger = logging.getLogger("stat_utils")
logger.setLevel(logging.INFO)

seterr(all="raise")
logger.propagate = False

def pooled_standard_error(
    control_deviation,
    number_control_samples,
    treatment_deviation,
    number_treatment_samples,
):
    """Pooled Standard Error for two samples

    :param float_or_pandas.series control_deviation: The experimental measured deviation of control values
    :param int_or_panda.series number_control_samples: The number of samples upon which control mean is based
    :param float_or_pandas.series treatment_deviation: The experimental measured deviation of treatment values
    :param int_or_Pandas.series number_treatment_samples: The number of samples upon which treatment mean is based
    :return: standard_error
    :rtype: pandas.series
    """
    try:
        pooled_deviation = sqrt(
            divide(
                add(
                    multiply(
                        subtract(number_control_samples, 1),
                        power(control_deviation, 2),
                    ),
                    subtract(number_treatment_samples, 1)
                    * power(treatment_deviation, 2),
                ),
                subtract(add(number_control_samples, number_treatment_samples), 2),
            )
        )

        standard_error = multiply(
            pooled_deviation,
            sqrt(
                add(
                    divide(1, number_control_samples),
                    divide(1, number_treatment_samples),
                )
            ),
        )
    except FloatingPointError as e:
        logger.error(f"Error: {e} - cannot divided by zero!")
        return None
    except TypeError as e:
        logger.error(f"Error: {e} - incorrect input data type!")
        return None
    else:
        logger.info(f"Calculated pooled standard error")
        return pd.Series(standard_error)
