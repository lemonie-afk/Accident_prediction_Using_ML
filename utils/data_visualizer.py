import matplotlib.pyplot as plt


def plot_accidents(df):

    df["Accident"].value_counts().plot(kind="bar")

    plt.title("Accident Distribution")
    plt.xlabel("Accident")
    plt.ylabel("Count")

    plt.savefig("accident_distribution.png")
