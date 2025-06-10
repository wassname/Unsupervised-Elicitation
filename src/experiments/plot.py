import json
from matplotlib import pyplot as plt
import os

for file in os.listdir("."):
    if file.startswith("log_"):
        print(file)
        with open(file, "r") as f:
            data = [json.loads(i) for i in f]
        data = data[:60]
        x = list(range(8, 8 + len(data)))
        y_score = [i['score'] for i in data]
        y_acc = [i['acc'] for i in data]
        # plt.subplot(1, 2, 1)
        plt.plot(x, y_score, label=f'Acc {max(y_acc):.2f}')
        plt.xlabel("# Searched Claims")
        plt.ylabel("Score")
        # plt.subplot(1, 2, 2)
        # plt.plot(x, y_acc)
        # plt.xlabel("# Searched Claims")
        # plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("log.png", dpi=500)