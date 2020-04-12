from constants import ROOT_DIR
import matplotlib.pyplot as plt
import csv

red = '#cc0000'
blue = '#3366ff'

s_0 = [0.45, 0.65]
i_0 = 0.5
r_0 = 0.0
beta = 0.8
gamma = 0.2

filename_log_loss_scratch = ROOT_DIR + '\\csv\\s_0={}-i_0={:.1f}-r_0={}_beta={}_gamma={}_noise_0.pt_scratch-Loss_Log-train.csv'.format(
    s_0, i_0, r_0, beta, gamma)
filename_log_loss_finetuned = ROOT_DIR + '\\csv\\s_0={}-i_0={:.1f}-r_0={}_beta={}_gamma={}_noise_0.pt_finetuned-Loss_Log-train.csv'.format(
    s_0, i_0, r_0, beta, gamma)

with open(filename_log_loss_scratch) as f:
    reader = csv.reader(f)
    scratch = list(reader)[1:]
    scratch = [float(x[2]) for x in scratch]
    scratch = scratch[:150]

with open(filename_log_loss_finetuned) as f:
    reader = csv.reader(f)
    finetuned = list(reader)[1:]
    finetuned = [float(x[2]) for x in finetuned]
    finetuned = finetuned[:150]

plt.figure(figsize=(10, 5))
plt.title('Loss trend - model trained on bundle of initial conditions\n'
          'S(0) bundle = {}\n'
          'Beta = {} | Gamma = {}'.format(s_0, beta, gamma))
plt.plot(range(len(scratch)), scratch, color=red, label='from scratch')
plt.plot(range(len(finetuned)), finetuned, color=blue, label='finetuned')
plt.xlabel('Epochs')
plt.ylabel('LogLoss')
plt.legend(loc='upper right')
plt.savefig(ROOT_DIR + '/plots/logloss_s_0={}-beta={}-gamma={}.png'.format(s_0, beta, gamma))
plt.show()
