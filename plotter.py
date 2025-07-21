import matplotlib.pyplot as plt
import numpy as np
import json


with open("./data/methods_theory_2.json") as f:
    methods_theory_2 = json.load(f)

with open("./data/methods_theory_3.json") as f:
    methods_theory_3 = json.load(f)

with open("./data/methods_theory_5.json") as f:
    methods_theory_5 = json.load(f)

with open("./data/methods_theory_7.json") as f:
    methods_theory_7 = json.load(f)

with open("./data/methods_2.json") as f:
    methods_2 = json.load(f)

with open("./data/methods_3.json") as f:
    methods_3 = json.load(f)

with open("./data/methods_5.json") as f:
    methods_5 = json.load(f)

with open("./data/methods_7.json") as f:
    methods_7 = json.load(f)


figure, ax = plt.subplots(1, figsize=(18,10))
ax.tick_params(axis='both', which='major', labelsize=30)
ax.tick_params(axis='both', which='minor', labelsize=30)

data_points = 60
shots_base = 16000
error = 0.05

thetas = np.linspace(0, np.pi, data_points)
eplisionError = [error/2 for el in thetas]

dot_size = 80

plt.plot(thetas, methods_theory_2, label="_Hidden", alpha=0.5, color="blue")
plt.scatter(thetas, methods_2, label=r"$\alpha$ = 2", color = "blue", s=dot_size)
plt.errorbar(thetas, methods_2, yerr=eplisionError, fmt="o", capsize=1, alpha=0.2, color="blue", elinewidth=3)

plt.plot(thetas, methods_theory_3, label="_Hidden".format(3), alpha=0.5, color="orange")
plt.scatter(thetas, methods_3, label=r"$\alpha$ = 3", color = "orange", s=dot_size)
plt.errorbar(thetas, methods_3, yerr=eplisionError, fmt="o", capsize=1, alpha=0.2, color="orange", elinewidth=3)

plt.plot(thetas, methods_theory_5, label="_Hidden".format(5), alpha=0.5, color="red")
plt.scatter(thetas, methods_5, label=r"$\alpha$ = 5", color = "red", s=dot_size)
plt.errorbar(thetas, methods_5, yerr=eplisionError, fmt="o", capsize=1, alpha=0.2, color="red", elinewidth=3)

plt.plot(thetas, methods_theory_7, label="_Hidden".format(7), alpha=0.5, color="green")
plt.scatter(thetas, methods_7, label=r" $\alpha$ = 7", color = "green", s=dot_size)
plt.errorbar(thetas, methods_7, yerr=eplisionError, fmt="o", capsize=1, alpha=0.2, color="green", elinewidth=3)

plt.xlabel(r"$\theta$", fontsize="45")
plt.ylabel(r"$A_{\alpha}(\vert \psi_\theta \rangle)$", fontsize="45")
lgnd = plt.legend(loc="lower left", ncol=4, fontsize=40, bbox_to_anchor=(-0.08, -0.3), frameon=False)
lgnd.legendHandles[0]._sizes = [600]
lgnd.legendHandles[1]._sizes = [600]
lgnd.legendHandles[2]._sizes = [600]
lgnd.legendHandles[3]._sizes = [600]

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)

plt.tight_layout()
# plt.show()
plt.savefig("./graph1.png")