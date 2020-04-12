from constants import ROOT_DIR
import matplotlib.pyplot as plt
import csv
import os

red = '#cc0000'
blue = '#3366ff'
dataset = 'MNIST'
m = 0.0
tag = 'Accuracy'
line = 'train'
percentage = 0.25
csv_folder = 'csv_runs'

with open(os.path.join(ROOT_DIR, csv_folder,
                       '{}_random{}-Accuracy_{}.csv'.format(dataset, percentage, 'train')),
          newline='') as f:
    reader = csv.reader(f)
    random50_train = list(reader)[1:]
    random50_train = [float(x[2]) for x in random50_train]

with open(os.path.join(ROOT_DIR, csv_folder,
                       '{}_top{}-Accuracy_{}.csv'.format(dataset, percentage, 'train')),
          newline='') as f:
    reader = csv.reader(f)
    top50_train = list(reader)[1:]
    top50_train = [float(x[2]) for x in top50_train]

with open(os.path.join(ROOT_DIR, csv_folder,
                       '{}_random{}-Accuracy_{}.csv'.format(dataset, percentage, 'test')),
          newline='') as f:
    reader = csv.reader(f)
    random50_test = list(reader)[1:]
    random50_test = [float(x[2]) for x in random50_test]

with open(os.path.join(ROOT_DIR, csv_folder,
                       '{}_top{}-Accuracy_{}.csv'.format(dataset, percentage, 'test')),
          newline='') as f:
    reader = csv.reader(f)
    top50_test = list(reader)[1:]
    top50_test = [float(x[2]) for x in top50_test]

with open(os.path.join(ROOT_DIR, csv_folder,
                       '{}_bottom{}-Accuracy_{}.csv'.format(dataset, percentage, 'test')),
          newline='') as f:
    reader = csv.reader(f)
    bottom50_test = list(reader)[1:]
    bottom50_test = [float(x[2]) for x in bottom50_test]

with open(os.path.join(ROOT_DIR, csv_folder,
                       '{}_bottom{}-Accuracy_{}.csv'.format(dataset, percentage, 'train')),
          newline='') as f:
    reader = csv.reader(f)
    bottom50_train = list(reader)[1:]
    bottom50_train = [float(x[2]) for x in bottom50_train]

linewidth = 2

fig = plt.figure(figsize=(12, 5))
st = fig.suptitle('Accuracy trend of a baseline trained on MNIST dataset\n'
                  'and finetuned on USPS dataset\n'
                  'Samples selected according to entropy-driven criterion\n'
                  )
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(range(len(top50_train)), top50_train, color=red, linewidth=linewidth, label='Top 25%')
ax1.plot(range(len(bottom50_train)), bottom50_train, color=red, linewidth=linewidth, label='Bottom 25%', linestyle='--')
ax1.plot(range(len(random50_train)), random50_train, color=blue, linewidth=linewidth, label='Random 25%', linestyle=':')
ax1.set_title("Train")
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend(loc='lower right')

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(range(len(top50_test)), top50_test, color=red, linewidth=linewidth, label='Top 25%')
ax2.plot(range(len(bottom50_test)), bottom50_test, color=red, linewidth=linewidth, label='Bottom 25%', linestyle='--')
ax2.plot(range(len(random50_test)), random50_test, color=blue, linewidth=linewidth, label='Random 25%', linestyle=':')
ax2.set_title("Test")
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend(loc='lower right')

fig.tight_layout()

st.set_y(0.95)
fig.subplots_adjust(top=0.75)
fig.savefig(ROOT_DIR + '/plots/{}_entropy_{}.png'.format(dataset, percentage))
