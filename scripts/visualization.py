import ast

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

f = open("SatMAE/outputs-checkpoint-finetune-eurosat-10pc/log.txt", "r")

dict_list = []
accuracies = []

for line in f:
    dict_list.append(ast.literal_eval(line))
    accuracies.append(ast.literal_eval(line)["test_acc1"])

print(accuracies)

plt.plot(accuracies)
plt.show()
