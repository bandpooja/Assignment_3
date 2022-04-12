import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def hyper_parameter_CM(array, index: list, columns: list, title: str,
                       xlabel: str, ylabel: str, img_loc: str):
    df_cm = pd.DataFrame(array, index=index, columns=columns)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, cmap='gist_earth_r')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(img_loc)
    plt.close()


def line_plots(arrays: list, labels: list, xticks: list, xlabel: str,
               ylabel: str, title: str, img_loc: str):
    if len(arrays) == 2:
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(arrays[0], 'g-', label=labels[0])
        ax2.plot(arrays[1], 'b-', label=labels[0])
        ax1.legend()
        ax2.legend()
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(labels[0], color='g')
        ax2.set_ylabel(labels[1], color='b')
        plt.title(title)
    else:
        for array, label in zip(arrays, labels):
            plt.plot(array, label=label)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
    plt.xticks(range(len(arrays[0])), xticks)
    plt.savefig(img_loc, bbox_inches='tight')
    plt.close()
