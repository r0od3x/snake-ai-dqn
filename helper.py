import matplotlib.pyplot as plt

plt.ion()


def plot(scores, mean_scores):
    """Live training curve: score per game and running mean."""
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean score')
    plt.ylim(ymin=0)
    plt.legend(loc='upper left')
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], f"{mean_scores[-1]:.2f}")
    plt.show(block=False)
    plt.pause(.1)
