import matplotlib.pyplot as plt
from constants import *
import csv

dataset='CIFAR_10'
tag = 'Accuracy'
shift = 'AE_shift'
m = 2
csv_folder = 'csv_runs'
red = '#cc0000'
blue = '#3366ff'

with open(os.path.join(ROOT_DIR, csv_folder,
                       'run-{}_jacobian_4-tag-{}_{}.csv'.format(dataset, tag, 'train')),
          newline='') as f:
    reader = csv.reader(f)
    jac_train = list(reader)[1:]
    jac_train = [float(x[2]) for x in jac_train]

with open(os.path.join(ROOT_DIR, csv_folder,
                       'run-{}_jacobian_4-tag-{}_{}.csv'.format(dataset, tag, 'val')),
          newline='') as f:
    reader = csv.reader(f)
    jac_val = list(reader)[1:]
    jac_val = [float(x[2]) for x in jac_val]

with open(os.path.join(ROOT_DIR, csv_folder,
                       'run-{}_random_jacobian_4-tag-{}_{}.csv'.format(dataset, tag, 'train')),
          newline='') as f:
    reader = csv.reader(f)
    rnd_train = list(reader)[1:]
    rnd_train = [float(x[2]) for x in rnd_train]

with open(os.path.join(ROOT_DIR, csv_folder,
                       'run-{}_random_jacobian_4-tag-{}_{}.csv'.format(dataset, tag, 'val')),
          newline='') as f:
    reader = csv.reader(f)
    rnd_val = list(reader)[1:]
    rnd_val = [float(x[2]) for x in rnd_val]

linewidth = 2

fig = plt.figure(figsize=(12, 5))
st = fig.suptitle('Accuracy trend of a baseline pre-trained on a clean CIFAR 10 dataset\n '
                  'and finetuned on a distorted version\n with embedding shift = {}\n'
                  'Samples selected according to the differential criterion'.format(m))

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(range(len(jac_train)), jac_train, label='Differential', color=red, linewidth=linewidth)
ax1.plot(range(len(rnd_train)), rnd_train, label='Random', color=blue, linestyle=':', linewidth=linewidth)
ax1.set_title("Train")
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend(loc='lower right')

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(range(len(jac_val)), jac_val, label='Differential', color=red, linewidth=linewidth)
ax2.plot(range(len(rnd_val)), rnd_val, label='Random', color=blue, linestyle=':', linewidth=linewidth)
ax2.set_title("Validation")
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend(loc='lower right')

fig.tight_layout()

st.set_y(0.95)
fig.subplots_adjust(top=0.75)
fig.savefig(ROOT_DIR + '/plots/{}_{}{}_50_jac.png'.format(dataset, shift, m))
plt.show()