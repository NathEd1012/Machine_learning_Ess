import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse


def load_data(file_path):
    """
    Load csv file into a pd DataFrame

    :param file_path: Path to training_stats.csv
    :return: DataFrame containing data
    """
    if not os.path.isfile(file_path):
        print(f"File not found in: {file_path}")
        sys.exit(1)

    data = pd.read_csv(file_path)
    return data

def plot_event_statistics(data, stats_to_plot):
    """
    Plot selected statistics over cumulative number of training iterations

    :param data: DataFrame w data
    :param stats_to_plot: List of columns to plot
    """

    plt.figure(figsize = (14, 7))

    # Calculate cumulative training iterations
    data["Cumulative Iterations"] = data["Rounds Played"].cumsum()

    for stat in stats_to_plot:
        if stat in data.columns:
            plt.plot(data["Cumulative Iterations"], data[stat] / data["Rounds Played"], label = stat)
        else:
            print(f"Statistic {stat} not found in columns.")

    plt.title("Statistics")
    plt.xlabel("Cumulative training iterations")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()

    plt.show()

def plot_action_histogram(data, actions_to_plot, row = -1):
    """
    Plot a histogram of the action distribution of selected row
    """
    if row >= len(data) or row < -len(data):
        print(f"Invalid row index: {row}. Data only has {len(data)} rows")
        sys.exit(1)

    row_data = data.iloc[row]

    action_counts = []
    for action in actions_to_plot:
        if action in data.columns:
            action_counts.append(row_data[action])
        else:
            print(f"Action {action} not found in columns.")

    plt.figure(figsize = (10, 6))
    plt.bar(actions_to_plot, action_counts, color = 'skyblue')
    plt.title(f"Action distribution for Row {row if row >= 0 else 'Last' }")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.tight_layout()

    plt.show()

def main():
    parser = argparse.ArgumentParser(description = "Visualize training statistics.")
    parser.add_argument('--file-path', type = str, required = True, help = "Path to the training_stats.csv file.")

    # Time series
    parser.add_argument('--stats', nargs = '+', help = "Select statistics to visualize over training iterations")

    # Histogram
    parser.add_argument('--hist', nargs = '*', help = "Select action counts to visualize in histogram")
    parser.add_argument('--row', type = int, default = -1, help = "Row index for histogram. Default: Last row")

    args = parser.parse_args()

    data = load_data(args.file_path)

    # Stats plot?
    if args.stats:
        plot_event_statistics(data, args.stats)

    # Hist?
    default_actions = ["MOVED_UP", "MOVED_DOWN", "MOVED_RIGHT", "MOVED_LEFT", "BOMB_DROPPED", "WAITED", "INVALID_ACTION"]
    if args.hist is not None:
        actions_to_plot = args.hist if args.hist else default_actions
        plot_action_histogram(data, actions_to_plot, row = args.row)

if __name__ == "__main__":
    main()