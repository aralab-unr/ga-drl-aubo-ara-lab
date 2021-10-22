import matplotlib.pyplot as plt
import numpy as np

title = "Success rate over Epochs"
xaxis = "Epochs"
yaxis = "Success rate"
inputTxtFileName = "sample_logs_success_rate_rollout.txt"
outputPlotFileName = "success_rate_over_epochs-1.png"

with open("plots data files by execution id/" + inputTxtFileName) as f:
    lines = f.readlines()
    y = [line.split()[0] for line in lines]

plt.plot(range(len(y)), y, 'o-')
plt.xticks(range(len(y)),y)
plt.title(title)
plt.xlabel(xaxis)
plt.ylabel(yaxis)
#plt.show()
plt.savefig(outputPlotFileName)

