import sys
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(sys.argv[1])

# Choose how many bins you want here
num_bins = 100

mn = min(data)
mx = max(data)

print(np.percentile(data,50))
print(np.percentile(data,95))
print(np.percentile(data,99))
print(np.percentile(data,99.9))
print(np.percentile(data,99.99))

"""
# Use the histogram function to bin the data
counts, bin_edges = np.histogram(data, bins=np.logspace(np.log10(mn),np.log10(mx), num_bins))

# Now find the cdf
cdf = np.cumsum(counts/sum(counts))

zipped = zip(bin_edges[1:], cdf)
l=list(zipped)
np.savetxt(sys.argv[1] + '_cdf.csv', l, fmt='%f %f', header='Latency_nsec CDF')

# plt.plot(cdf)

# plt.rcParams['axes.formatter.useoffset'] = False

# And finally plot the cdf

# axes=plt.gca()
# axes.set_xlabel('Data')
# axes.set_ylabel('Probability')

# np.round(bin_edges, decimals=2)
# axes.set_xticklabels(bin_edges)
# plt.xticks(rotation = 15)

# plt.savefig(sys.argv[1] + '_cdf.png', dpi=300)

# for scatter plot
# x=np.arange(0,data.size)
# plt.scatter(x,data)
# plt.savefig(sys.argv[1] + '_scatter.png', dpi=300)

# plt.show()

"""

