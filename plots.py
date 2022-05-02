import matplotlib.pyplot as plt

from utils import load_results


def plot_results(results):
    plt.figure(figsize=(10, 10))
    plt.plot(results["rouge1"].keys(), results["rouge1"].values(), label="ROUGE-1", color="red")
    plt.plot(results["rouge2"].keys(), results["rouge2"].values(), label="ROUGE-2", color="green")
    plt.plot(results["rougeL"].keys(), results["rougeL"].values(), label="ROUGE-L", color="blue")

    plt.xlabel("Number of documents")
    plt.ylabel("ROUGE score")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    results = load_results("test-metrics")
    plot_results(results)