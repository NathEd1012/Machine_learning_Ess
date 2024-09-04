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

def main():
    parser = argparse.ArgumentParser(description = "Visualize training statistics.")
    parser.add_argument('--file-path', type = str, required = True, help = "Path to the training_stats.csv file.")
    parser.add_argument('--stats', nargs = '+', required = True, help = "Select statistics to visualize")

    args = parser.parse_args()

    data = load_data(args.file_path)
    plot_event_statistics(data, args.stats)

if __name__ == "__main__":
    main()