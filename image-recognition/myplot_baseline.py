from constants import ROOT_DIR
import matplotlib.pyplot as plt
import csv
import os

red = '#cc0000'
blue = '#3366ff'
dataset = 'MNIST'
shift = 'AE_shift'
m = 0.0
tag = 'Accuracy'
line = 'train'
percentages = [0.5]
csv_folder = 'csv_runs'

with open(os.path.join(ROOT_DIR, csv_folder,
                       '{}_baseline_clean-{}_{}.csv'.format(dataset, tag, 'train')),
          newline='') as f:
    reader = csv.reader(f)
    train = list(reader)[1:]
    train = [float(x[2]) for x in train]

with open(os.path.join(ROOT_DIR, csv_folder,
                       '{}_baseline_clean-{}_{}.csv'.format(dataset, tag, 'val')),
          newline='') as f:
    reader = csv.reader(f)
    val = list(reader)[1:]
    val = [float(x[2]) for x in val]

# with open(os.path.join(ROOT_DIR, csv_folder,
#                        'run-{}_AE_shift-m={}-std=0_top0.25-tag-{}_{}.csv'.format(dataset, m, tag, 'test')),
#           newline='') as f:
#     reader = csv.reader(f)
#     top25_test = list(reader)[1:]
#     top25_test = [float(x[2]) for x in top25_test]
#
# with open(os.path.join(ROOT_DIR, csv_folder,
#                        'run-{}_AE_shift-m={}-std=0_bottom0.25-tag-{}_{}.csv'.format(dataset, m, tag, 'train')),
#           newline='') as f:
#     reader = csv.reader(f)
#     bottom25_train = list(reader)[1:]
#     bottom25_train = [float(x[2]) for x in bottom25_train]
#
# with open(os.path.join(ROOT_DIR, csv_folder,
#                        'run-{}_AE_shift-m={}-std=0_bottom0.25-tag-{}_{}.csv'.format(dataset, m, tag, 'val')),
#           newline='') as f:
#     reader = csv.reader(f)
#     bottom25_val = list(reader)[1:]
#     bottom25_val = [float(x[2]) for x in bottom25_val]
#
# with open(os.path.join(ROOT_DIR, csv_folder,
#                        'run-{}_AE_shift-m={}-std=0_bottom0.25-tag-{}_{}.csv'.format(dataset, m, tag, 'test')),
#           newline='') as f:
#     reader = csv.reader(f)
#     bottom25_test = list(reader)[1:]
#     bottom25_test = [float(x[2]) for x in bottom25_test]
#
# with open(os.path.join(ROOT_DIR, csv_folder,
#                        'run-{}_AE_shift-m={}-std=0_random0.25-tag-{}_{}.csv'.format(dataset, m, tag, 'train')),
#           newline='') as f:
#     reader = csv.reader(f)
#     random25_train = list(reader)[1:]
#     random25_train = [float(x[2]) for x in random25_train]
#
# with open(os.path.join(ROOT_DIR, csv_folder,
#                        'run-{}_AE_shift-m={}-std=0_random0.25-tag-{}_{}.csv'.format(dataset, m, tag, 'test')),
#           newline='') as f:
#     reader = csv.reader(f)
#     random25_test = list(reader)[1:]
#     random25_test = [float(x[2]) for x in random25_test]

linewidth = 2

fig = plt.figure(figsize=(12, 5))
st = fig.suptitle('Accuracy trend of baseline trained on a MNIST dataset')
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(range(len(train)), train, color=red, linewidth=linewidth)
ax1.set_title("Train")
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(range(len(val)), val, color=red, linewidth=linewidth)
ax2.set_title("Validation")
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')

fig.tight_layout()

st.set_y(0.95)
fig.subplots_adjust(top=0.85)
fig.savefig(ROOT_DIR + '/plots/{}_baseline.png'.format(dataset))
