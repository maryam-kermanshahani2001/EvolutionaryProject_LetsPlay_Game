import json
import matplotlib.pyplot as plt
import numpy as np

fit_res = json.load(open("fit_res.json", "r"))
num_of_generations = len(fit_res['min_score'])

x = np.zeros(num_of_generations)
for i in range(num_of_generations):
    x[i] = i


plt.plot(x, fit_res['min_score'])
plt.title("min")
plt.show()
plt.plot(x, fit_res['max_score'])
plt.title("max")
plt.show()
plt.plot(x, fit_res['avg_score'])
plt.title("avg")
plt.show()
