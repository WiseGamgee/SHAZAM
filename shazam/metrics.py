import torch
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from torchmetrics.classification import (ROC, AUROC, F1Score, Recall, Precision,
                                         PrecisionRecallCurve, AveragePrecision, Accuracy)


def get_binary_metrics(y_pred_binary,
                       y_true,
                       round_dp: int = 4):
    # Convert to tensors
    y_pred_binary, y_true = torch.Tensor(y_pred_binary).long(), torch.Tensor(y_true).long()

    # Instantiate threshold metrics
    f1 = F1Score(task="binary")
    recall = Recall(task="binary")
    prec = Precision(task="binary")
    acc = Accuracy(task="binary")

    # Compute all metrics and store in dictionary
    metrics_dict = {"f1_score": round(f1(y_pred_binary, y_true).item(), round_dp),
                    "recall": round(recall(y_pred_binary, y_true).item(), round_dp),
                    "precision": round(prec(y_pred_binary, y_true).item(), round_dp),
                    "accuracy": round(acc(y_pred_binary, y_true).item(), round_dp)}
    return metrics_dict


def get_metrics(y_pred,
                y_true,
                threshold: float = None,
                round_dp: int = 4):
    # Convert to tensors
    y_pred, y_true = torch.Tensor(y_pred), torch.Tensor(y_true).long()

    # Create dictionary to store metrics
    metrics_dict = {}

    # Define threshold-less metrics
    roc = ROC(task="binary", thresholds=1000)
    auroc = AUROC(task="binary")
    pr_curve = PrecisionRecallCurve(task="binary", thresholds=1000)
    ap = AveragePrecision(task="binary")

    # Get ROC and PR curves
    roc_fpr, roc_tpr, roc_thresholds = roc(y_pred, y_true)
    pr_prec, pr_recall, pr_thresholds = pr_curve(y_pred, y_true)
    metrics_dict.update({"roc":
                             {"auc": round(auroc(y_pred, y_true).item(), round_dp),
                              "fpr": [round(i, round_dp) for i in roc_fpr.tolist()],
                              "tpr": [round(i, round_dp) for i in roc_tpr.tolist()],
                              "thresholds": [round(i, round_dp) for i in roc_thresholds.tolist()]},
                         "pr_curve":
                             {"aupr": round(ap(y_pred, y_true).item(), round_dp),
                              "random_aupr": round(torch.sum(y_true).item() / len(y_true), round_dp),
                              "precision": [round(i, round_dp) for i in pr_prec.tolist()],
                              "recall": [round(i, round_dp) for i in pr_recall.tolist()],
                              "thresholds": [round(i, round_dp) for i in pr_thresholds.tolist()]}})

    # Check if threshold is available
    if threshold:
        # Use threshold to establish binary outputs
        y_pred_binary = torch.where(y_pred < threshold, 0., 1.)

        # Instantiate threshold metrics
        f1 = F1Score(task="binary")
        recall = Recall(task="binary")
        prec = Precision(task="binary")
        acc = Accuracy(task="binary")

        # Compute all metrics and store in dictionary
        metrics_dict.update({"threshold": {"value": threshold,
                                           "f1_score": round(f1(y_pred_binary, y_true).item(), round_dp),
                                           "recall": round(recall(y_pred_binary, y_true).item(), round_dp),
                                           "precision": round(prec(y_pred_binary, y_true).item(), round_dp),
                                           "accuracy": round(acc(y_pred_binary, y_true).item(), round_dp)}})
    return metrics_dict


def get_monthly_thresholds(data,
                           n_stddevs: float = 1.64,
                           results_dir: str = None):
    """
    Calculates and plots a threshold based on the monthly mean + standard deviation,
    and prints the monthly mean and standard deviation as a dictionary.

    Args:
        data (dict): Dictionary with date (YYYY-MM-DD) keys and anomaly score values.
    """
    # Convert the dictionary to lists of day of the year and scores
    day_of_year = [datetime.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday for date_str in data.keys()]
    scores = list(data.values())

    # Create a reference year for plotting (leap year to cover all 366 days)
    base_date = datetime(2020, 1, 1)
    dates = [base_date + timedelta(days=day - 1) for day in day_of_year]

    # Create DataFrame with the dates and scores
    scores_df = pd.DataFrame({"Date": dates, "Score": scores})
    scores_df = scores_df.sort_values("Date")
    scores_df.set_index("Date", inplace=True)

    # Extract the month for each entry
    scores_df['Month'] = scores_df.index.month - 1

    # Calculate the mean and standard deviation for each month
    monthly_stats = scores_df.groupby('Month')['Score'].agg(['mean', 'std'])
    monthly_stats['Monthly_Threshold'] = monthly_stats['mean'] + n_stddevs * monthly_stats['std']
    scores_df = scores_df.join(monthly_stats, on='Month')

    if results_dir:
        # Set Seaborn style
        sns.set(style="whitegrid")

        # Plot the original scores and the monthly thresholds using Seaborn
        plt.figure(figsize=(12, 6))

        # Scatter plot for anomaly scores
        sns.scatterplot(x=scores_df.index,
                        y=scores_df['Score'],
                        marker='o',
                        color='b',
                        label='Anomaly Score')

        # Line plot for monthly thresholds
        sns.lineplot(x=scores_df.index,
                     y=scores_df['Monthly_Threshold'],
                     color='r',
                     label=f"Monthly Mean + {n_stddevs} Std Devs",
                     errorbar=None)

        # Format the x-axis to show month names only
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

        # Add labels and title
        plt.xlabel('Month')
        plt.ylabel('Anomaly Score')
        plt.title(f'Anomaly Scores with Monthly Mean + {n_stddevs} Std Dev Threshold')

        # Show grid, legend, and display the plot
        plt.grid(True)
        plt.legend()

        # Save plot to file
        plt.savefig(f"{results_dir}/monthly_thresholds_plot.png")

    # Return monthly thresholds
    return monthly_stats["Monthly_Threshold"].to_dict()


def get_circular_polynomial_thresholds(anomaly_scores, degree=3, stddev_degree=2, n_stddev=3, results_dir=None):
    """
    Fits a polynomial to anomaly scores and calculates a threshold using a polynomial fit to the monthly
    standard deviations of the residuals, considering the cyclical nature of day-of-year data.

    Parameters:
    - anomaly_scores (dict): Dictionary where keys are dates (YYYY-MM-DD) and values are anomaly scores.
    - degree (int): Degree of the polynomial to fit to the anomaly scores (default: 3).
    - stddev_degree (int): Degree of the polynomial to fit to the monthly standard deviations (default: 2).
    - n_stddev (int): Number of standard deviations to add to the polynomial fit to set the threshold (default: 3).

    Returns:
    - None: Displays a plot of the anomaly scores, polynomial fit, and threshold based on monthly standard deviations.
    """
    # Convert dates to day-of-year
    dates = list(anomaly_scores.keys())
    scores = list(anomaly_scores.values())
    day_of_year = np.array([datetime.strptime(date, "%Y-%m-%d").timetuple().tm_yday for date in dates])

    # Convert day of year to circular components (sine and cosine)
    days_normalized = day_of_year / 365.0
    sin_day = np.sin(2 * np.pi * days_normalized)
    cos_day = np.cos(2 * np.pi * days_normalized)

    # Prepare data for polynomial fitting
    X = np.vstack([sin_day, cos_day]).T
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly_features.fit_transform(X)

    # Fit polynomial to the anomaly scores
    poly_model = LinearRegression().fit(X_poly, scores)
    predicted_scores = poly_model.predict(X_poly)

    # Calculate residuals
    residuals = scores - predicted_scores

    # Calculate monthly standard deviations
    df = pd.DataFrame({"Day": day_of_year, "Residuals": residuals})
    df['Month'] = [datetime.strptime(date, "%Y-%m-%d").month for date in dates]
    monthly_stddev = df.groupby("Month")["Residuals"].std().reindex(range(1, 13), fill_value=np.nan)

    # Filter out months with NaN standard deviations
    valid_months = monthly_stddev.dropna().index
    valid_stddev = monthly_stddev.dropna().values
    middle_days = np.array([datetime(datetime.now().year, month, 15).timetuple().tm_yday for month in valid_months])

    # Filter days of the year to include only those corresponding to valid months
    valid_days = np.array(
        [day for date, day in zip(dates, day_of_year) if datetime.strptime(date, "%Y-%m-%d").month in valid_months])

    if len(valid_stddev) > 0 and len(valid_days) > 0:
        middle_days_normalized = middle_days / 365.0
        sin_middle_day = np.sin(2 * np.pi * middle_days_normalized)
        cos_middle_day = np.cos(2 * np.pi * middle_days_normalized)
        stddev_poly_features = PolynomialFeatures(degree=stddev_degree, include_bias=False)
        X_stddev_poly = stddev_poly_features.fit_transform(np.vstack([sin_middle_day, cos_middle_day]).T)
        stddev_poly_model = LinearRegression().fit(X_stddev_poly, valid_stddev)
    else:
        raise ValueError("Missing data.")

    # Predict standard deviations for all days of the year
    all_days = np.arange(0, 366)  # Days of the year (including leap year)
    all_days_normalized = all_days / 365.0
    sin_all_days = np.sin(2 * np.pi * all_days_normalized)
    cos_all_days = np.cos(2 * np.pi * all_days_normalized)
    days_components = np.vstack([sin_all_days, cos_all_days]).T
    X_days_poly = stddev_poly_features.transform(days_components)
    stddev_poly_pred = stddev_poly_model.predict(X_days_poly)

    # Predict scores for all days of the year using the fitted polynomial model
    X_all_days_poly = poly_features.transform(days_components)
    predicted_all_days_scores = poly_model.predict(X_all_days_poly)
    threshold = predicted_all_days_scores + n_stddev * stddev_poly_pred

    # Calculate R² value
    ss_total = np.sum((scores - np.mean(scores)) ** 2)
    ss_residual = np.sum((scores - predicted_scores) ** 2)
    r_squared = 1 - (ss_residual / ss_total)

    if results_dir:
        # Set Seaborn style
        sns.set(style="whitegrid")

        # Plot the results using Seaborn
        plt.figure(figsize=(12, 6))

        # Create a DataFrame for easy plotting with Seaborn
        dates = [datetime.strptime(date, '%Y-%m-%d') for date in anomaly_scores.keys()]
        df = pd.DataFrame({
            'Day of the Year': [date.timetuple().tm_yday for date in dates],
            'Anomaly Score': list(anomaly_scores.values()),
            'Year': [f"Anomaly Scores ({date.year})" for date in dates]
        })

        # Get unique years
        unique_years = sorted(df['Year'].unique())
        marker_styles = itertools.cycle(["o", "s", "D", "P", "X", "^", "*", "v", "<", ">"])  # Marker options
        year_markers = {year: next(marker_styles) for year in unique_years}

        # Plot using Seaborn, coloring by 'Year'
        sns.scatterplot(x='Day of the Year',
                        y='Anomaly Score',
                        hue='Year',
                        style='Year',
                        palette='dark',
                        markers=year_markers,
                        data=df,
                        s=100)

        # # Scatter plot for anomaly scores
        # sns.scatterplot(x=day_of_year,
        #                 y=scores,
        #                 label="Anomaly Score",
        #                 color="b")

        # # Line plot for polynomial fit
        # sns.lineplot(x=all_days,
        #              y=predicted_all_days_scores,
        #              label=f"Regression",
        #              color="g",
        #              errorbar=None)
        #
        # # Line plot for standard deviation polynomial fit
        # sns.lineplot(x=all_days,
        #              y=stddev_poly_pred,
        #              label=f"Std. Deviation Fit",
        #              color="purple",
        #              linestyle='dotted',
        #              errorbar=None)

        # Line plot for the threshold
        sns.lineplot(x=all_days,
                     y=threshold,
                     label=f"Seasonal Threshold", #(Regression + {n_stddev} Std. Dev. Fit)
                     color="r",
                     linestyle="dashed",
                     errorbar=None)

        # Customize font sizes
        AXIS_LABEL_SIZE = 18
        TICK_LABEL_SIZE = 16
        LEGEND_FONT_SIZE = 18

        # Add labels with custom font sizes
        plt.xlabel("Day of the Year", fontsize=AXIS_LABEL_SIZE)
        plt.ylabel("Anomaly Score", fontsize=AXIS_LABEL_SIZE)

        # Customize tick label sizes
        plt.xticks(fontsize=TICK_LABEL_SIZE)
        plt.yticks(fontsize=TICK_LABEL_SIZE)

        # Format the x-axis to show month names only
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

        # Customize legend
        plt.legend(fontsize=LEGEND_FONT_SIZE)
        # For more legend customization:
        # plt.legend(fontsize=LEGEND_FONT_SIZE, title='Legend Title', title_fontsize=LEGEND_FONT_SIZE + 2)

        # Save plot to file
        plt.savefig(f"{results_dir}/polynomial_thresholds_plot.png", bbox_inches='tight', dpi=300)

    # Return threshold dict
    return {i: val for i, val in enumerate(threshold)}


def apply_thresholds(scores_dict, threshold_dict, threshold_time_period='day', return_binary=True):
    """
    Converts anomaly scores to binary values based on whether they exceed the threshold,
    where thresholds can be defined by day, week, fortnight, or month.

    Args:
        scores_dict (dict): Dictionary with date (YYYY-MM-DD) keys and anomaly score values.
        threshold_dict (dict): Dictionary with keys based on granularity ('day', 'week', 'fortnight', 'month')
                               and threshold values.
        threshold_time_period (str): Defines the granularity for threshold lookup ('day', 'week', 'fortnight', 'month').

    Returns:
        dict: Dictionary with the same date keys, but binary values (1 if score exceeds threshold, 0 otherwise).
    """
    y_pred = {}

    for date_str, score in scores_dict.items():
        # Convert date string to datetime object
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")

        key = None
        if threshold_time_period == 'day':
            key = date_obj.timetuple().tm_yday - 1  # 0-based day of the year
        elif threshold_time_period == 'week':
            key = (date_obj.timetuple().tm_yday - 1) // 7  # 0-based week number
        elif threshold_time_period == 'fortnight':
            key = (date_obj.timetuple().tm_yday - 1) // 14  # 0-based fortnight number
        elif threshold_time_period == 'month':
            key = date_obj.month - 1  # 0-based month (January = 0)
        elif threshold_time_period == 'season':
            # Calculate the season of the year (0-indexed)
            month = date_obj.month
            if month in [12, 1, 2]:
                key = 0  # Summer
            elif month in [3, 4, 5]:
                key = 1  # Autumn
            elif month in [6, 7, 8]:
                key = 2  # Winter
            elif month in [9, 10, 11]:
                key = 3  # Spring
        else:
            raise ValueError(f"Invalid granularity: {threshold_time_period}. Use 'day', 'week', 'fortnight', 'month', or 'season'.")

        # Get the threshold for the current key (day, week, fortnight, or month)
        threshold = threshold_dict.get(key, None)
        try:
            score -= threshold
        except ValueError as e:
            print(f"Threshold value not found for {threshold_time_period} = {key}.")

        # Apply the threshold to determine the binary result
        if return_binary:
            y_pred[date_str] = 1 if score > 0 else 0
        else:
            y_pred[date_str] = score
    return y_pred


def plot_anomaly_scores(train_sdim_dict,
                        test_sdim_dict,
                        anomaly_label_dict,
                        poly_thresholds,
                        results_dir):
    # Set Seaborn style
    sns.set(style="whitegrid")

    # Combine train and test data into a single dictionary
    sdim_dict = {**train_sdim_dict, **test_sdim_dict}

    # Convert date strings to datetime objects and sort the dictionaries
    all_dates = sorted([datetime.strptime(date, '%Y-%m-%d') for date in sdim_dict.keys()])
    all_scores = [sdim_dict[date.strftime('%Y-%m-%d')] for date in all_dates]

    # Convert date strings to datetime objects and sort the dictionaries
    test_dates = sorted([datetime.strptime(date, '%Y-%m-%d') for date in test_sdim_dict.keys()])

    # Create figure
    plt.figure(figsize=(25, 10))
    ax = plt.gca()

    # Add shaded region for test data (faded red)
    ax.axvspan(test_dates[0], test_dates[-1], facecolor='red', alpha=0.1, label='Test Period')

    # Plot residuals over time
    sns.lineplot(x=all_dates, y=all_scores, label="SIU-Net Anomaly Scores", color='b', ax=ax)

    # Generate a continuous date range from the earliest to the latest date
    every_day = [all_dates[0] + timedelta(days=i) for i in range((all_dates[-1] - all_dates[0]).days + 1)]

    # Map the threshold values to the corresponding day of the year, accounting for year reset
    days_of_year = [(date.timetuple().tm_yday - 1) % 366 for date in every_day]  # Ensure day 0-364 for each year

    # Map the threshold values to each day of the year
    threshold_values = [poly_thresholds[day] for day in days_of_year]

    # Plot the threshold line across the entire date range
    sns.lineplot(x=every_day, y=threshold_values, label='Seasonal Threshold', linestyle="--", color='black', ax=ax)

    # Separate anomalies (1) from normal (0) using label_dict
    anomaly_dates = [datetime.strptime(date, '%Y-%m-%d') for date in anomaly_label_dict.keys() if
                     anomaly_label_dict[date] == 1]
    anomaly_scores = [sdim_dict[date.strftime('%Y-%m-%d')] for date in anomaly_dates]

    # Plot anomalies
    label_detected = False
    label_missed = False
    for date, anomaly_score in zip(anomaly_dates, anomaly_scores):
        if anomaly_score > poly_thresholds[date.timetuple().tm_yday - 1]:
            sns.scatterplot(x=[date], y=[anomaly_score], marker='o', color='green', s=50, ax=ax,
                            label='Anomaly Detected' if not label_detected else "")
            label_detected = True
        else:
            sns.scatterplot(x=[date], y=[anomaly_score], marker='x', color='red', s=100, ax=ax,
                            label='Anomaly Missed' if not label_missed else "")
            label_missed = True

    # Format the date on the x-axis to show only the start of each month and its year
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))

    # Automatically adjust the date labels to prevent overlap
    plt.gcf().autofmt_xdate()

    # Labels and title with customized fonts
    plt.xlabel('Date', fontsize=24)
    plt.ylabel('Anomaly Score', fontsize=24)
    plt.xticks(rotation=90, fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)

    # Save the plot as PNG
    plt.savefig(f"{results_dir}/sdim_plot.png", format="png")


def plot_residuals(train_sdim_dict,
                   test_sdim_dict,
                   anomaly_label_dict,
                   poly_thresholds,
                   results_dir):
    # Set Seaborn style
    sns.set(style="whitegrid")

    # Combine train and test data into a single dictionary
    sdim_dict = {**train_sdim_dict, **test_sdim_dict}

    # Convert date strings to datetime objects and sort the dictionaries
    all_dates = sorted([datetime.strptime(date, '%Y-%m-%d') for date in sdim_dict.keys()])
    all_scores = [sdim_dict[date.strftime('%Y-%m-%d')] for date in all_dates]

    # Extract test dates
    test_dates = sorted([datetime.strptime(date, '%Y-%m-%d') for date in test_sdim_dict.keys()])

    # Create plot figure and axes
    plt.figure(figsize=(25, 10))
    ax = plt.gca()

    # Highlight test period with a red shaded region
    ax.axvspan(test_dates[0], test_dates[-1], facecolor='red', alpha=0.1, label='Test Period')

    # Make residual boundary very clear
    plt.axhline(y=0,
                color='black',
                linestyle='--',
                linewidth=2)

    # Calculate day of year for each date, ensuring correct day (0-364 for each year)
    days_of_year = [(date.timetuple().tm_yday - 1) % 366 for date in all_dates]

    # Map threshold values to each day of the year
    threshold_values = [poly_thresholds[day] for day in days_of_year]

    # Calculate residuals (anomaly score - threshold)
    residual_values = np.subtract(all_scores, threshold_values)

    # Plot residuals over time
    sns.lineplot(x=all_dates, y=residual_values, label="SIU-Net Anomaly Score Residuals", color='b', ax=ax)

    # Separate anomalies (where label is 1) from normal data
    anomaly_dates = [datetime.strptime(date, '%Y-%m-%d') for date in anomaly_label_dict.keys() if anomaly_label_dict[date] == 1]
    anomaly_residuals = [sdim_dict[date.strftime('%Y-%m-%d')] - poly_thresholds[(date.timetuple().tm_yday - 1) % 366]
                         for date in anomaly_dates]

    # Plot anomalies as green crosses for positive residuals and red crosses for negative residuals
    label_detected = False
    label_missed = False
    for date, residual in zip(anomaly_dates, anomaly_residuals):
        if residual > 0:
            sns.scatterplot(x=[date], y=[residual], marker='o', color='green', s=50, ax=ax,
                            label='Anomaly Detected' if not label_detected else "")
            label_detected = True
        else:
            sns.scatterplot(x=[date], y=[residual], marker='x', color='red', s=100, ax=ax,
                            label='Anomaly Missed' if not label_missed else "")
            label_missed = True

    # Format x-axis to show the start of each month and year
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))

    # Automatically adjust the date labels to prevent overlap
    plt.gcf().autofmt_xdate()

    # Customize labels and fonts
    plt.xlabel('Date', fontsize=24)
    plt.ylabel('Residual', fontsize=24)
    plt.xticks(rotation=90, fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)

    # Save the plot as a PNG
    plt.savefig(f"{results_dir}/residual_plot.png", format="png")
