import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# INPUTS
extension = "-1"






file_path = f"MapTesting/all_distribution_test/all_dist{extension}.csv"
data = {}
with open(file_path, 'r', newline='') as csvfile:
    data = {}
    dir_data = {}
    headers = False
    reader = csv.reader(csvfile)
    for row in reader:
        if headers is False:
            headers = row
            for header in headers[:-2]:
                data[header] = []
            dir_data[headers[-2]] = []
            dir_data[headers[-1]] = []
            continue
        for col in range(len(row)):
            if col <= 8:
                data[headers[col]].append(int(row[col]))
            else:
                dir_data[headers[col]].append(float(row[col]))

species = headers[:-2]
counts = {
    "Empty"  : [],
    "Passed" : [],
    "Object" : [],
    "Target" : []
}
label = {
    "Empty"  : 2,
    "Passed" : 1,
    "Object" : -1,
    "Target" : 3
}

for key in counts:
    for cell in data:
        counts[key].append(data[cell].count(label[key]) / len(data[cell]) * 100)


x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in counts.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    # ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Label Count [%]')
ax.set_xticks(x + width, [spec.strip('cell_') for spec in species])
ax.legend(loc='upper center', ncols=4)
ax.set_ylim(0, 50)

plt.show()


# from matplotlib import colors
# from matplotlib.ticker import PercentFormatter
# dist1 = dir_data[headers[-2]]
# dist2 = dir_data[headers[-1]]
# n_bins = 20
# fig, axs = plt.subplots(1, 2, tight_layout=True)

# # N is the count in each bin, bins is the lower-limit of the bin
# N, bins, patches = axs[0].hist(dist1, bins=n_bins)

# # We'll color code by height, but you could use any scalar
# fracs = N / N.max()

# # we need to normalize the data to 0..1 for the full range of the colormap
# norm = colors.Normalize(fracs.min(), fracs.max())

# # Now, we'll loop through our objects and set the color of each accordingly
# for thisfrac, thispatch in zip(fracs, patches):
#     color = plt.cm.viridis(norm(thisfrac))
#     thispatch.set_facecolor(color)

# # We can also normalize our inputs by the total number of counts
# axs[1].hist(dist1, bins=n_bins, density=True)

# # Now we format the y-axis to display percentage
# axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))

fig, ax = plt.subplots(tight_layout=True)
hist = ax.hist2d(dir_data[headers[-2]], dir_data[headers[-1]], cmap = 'binary', bins=20)
plt.gca().set_aspect('equal')
plt.show()
