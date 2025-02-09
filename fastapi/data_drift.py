import pandas as pd
from scipy.stats import ks_2samp

def get_data_distribution(df):
    """
    Calculate basic statistics and quantiles for input data.
    Args:
        df (pd.DataFrame): Input data
    Returns:
        pd.DataFrame: Dataframe containing statistics for each feature.
    """
    stats = pd.DataFrame({
        'mean': df.mean(),
        'std_dev': df.std(),
        'min': df.min(),
        'max': df.max(),
        'percentile_25': df.quantile(0.25),
        'percentile_50': df.median(),
        'percentile_75': df.quantile(0.75)
    })
    return stats

def ks_test(training_data, input_data):
    """
    Apply Kolmogorov-Smirnov test to check if two datasets come from the same distribution.
    Args:
        training_data (array-like): Training dataset for comparison
        input_data (array-like): Incoming dataset to check against training data
    Returns:
        tuple: KS statistic and p-value
    """
    ks_statistic, p_value = ks_2samp(training_data, input_data)
    return ks_statistic, p_value

def check_data_drift(training_data, input_data, feature_name):
    """
    Compare distributions and check if drift is detected.
    Args:
        training_data (pd.Series): Training data for feature
        input_data (pd.Series): Incoming data for feature
        feature_name (str): The name of the feature being checked
    """
    print(f"Checking drift for feature: {feature_name}")

    ks_stat, p_val = ks_test(training_data, input_data)
    print(f"KS Statistic: {ks_stat}, P-Value: {p_val}")

    if p_val < 0.05:
        print(f"Data drift detected for feature {feature_name}. Triggering retraining pipeline...")
        return True
    else:
        print(f"No significant drift detected for feature {feature_name}. No retraining required.")
        return False



