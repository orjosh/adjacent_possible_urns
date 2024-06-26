import powerlaw
from matplotlib import pyplot as plt
from retweetsamplemodel import RetweetSampleModel

model = RetweetSampleModel(n_in=500, n_events=10000)

degree_distribution = ["normal", [100, 25]]
# degree_distribution = ["power", [1.9]]
contacts_chance_distribution = ["power", [1.9]]
# contacts_chance_distribution = ["normal", [100, 25]]

freqs = model.generate_event_frequencies(degree_distribution, contacts_chance_distribution)

print(freqs)

# fig, ax = plt.subplots()
# ax.hist(model.get_degrees())

fig, axs = plt.subplots(nrows=2, ncols=1)

fit = powerlaw.Fit(freqs)
fit.plot_ccdf(ax=axs[1], linestyle='--', marker='o', label="Data", original_data=True)
fit.power_law.plot_ccdf(ax=axs[1], linestyle='--', linewidth='2', label=r"Power-law ($\gamma$=" + f"{fit.power_law.alpha:.2f})")

axs[0].scatter(range(len(freqs)), freqs)

for ax in axs:
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()

axs[0].set_ylabel("Frequency")
axs[0].set_xlabel("Rank")

axs[1].set_ylabel(r"$P(X\geq x)$")
axs[1].set_xlabel("Frequency")

fig.suptitle("Gaussian degree, Zipfian contact chance")
fig.set_size_inches(10, 10)
fig.savefig("test.png")