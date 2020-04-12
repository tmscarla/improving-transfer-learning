from matplotlib import animation, rc
import matplotlib.pyplot as plt
import numpy as np

rc('animation', html='html5')


class Animator:

    def __init__(self, X_noisy_train_s, y_train, prop, property_perc, alternate,
                 n_epochs=100, interval=100):
        self.X_noisy_train_s = X_noisy_train_s
        self.y_train = y_train
        self.prop = prop
        self.property_perc = property_perc
        self.alternate = alternate
        self.n_epochs = n_epochs
        self.interval = interval

        self.reset_fig()

    def reset_fig(self):
        self.fig, self.ax = plt.subplots()
        self.fig.set_dpi(100)
        self.fig.set_size_inches(10, 6)

        max_x0, min_x0 = max(self.X_noisy_train_s[:, 0]), min(self.X_noisy_train_s[:, 0])
        max_x1, min_x1 = max(self.X_noisy_train_s[:, 1]), min(self.X_noisy_train_s[:, 1])
        self.ax.set_xlim((min_x0 - 0.2, max_x0 + 0.2))
        self.ax.set_ylim((min_x1 - 0.2, max_x1 + 0.2))

    def calculate_db(self, weight1, weight2, bias):
        slope = -weight1 / weight2
        x_vals = np.array(self.ax.get_xlim())
        y_vals = -bias / weight2 + slope * x_vals
        return x_vals, y_vals

    def init_d(self):
        self.ax.legend(loc='upper left')
        self.ax.set_title(
            "Decision boundaries evolution\n property = {} | selected {}% |"
            " alternate = {}".format(self.prop, int(self.property_perc * 100), self.alternate))

        self.ax.scatter(self.X_noisy_train_s[:, 0], self.X_noisy_train_s[:, 1], marker='o', c=self.y_train,
                        s=40, alpha=0.3, edgecolor='k')

        for j in range(len(self.lines)):
            self.lines[j].set_data([], [])

        return tuple(self.lines)

    def animate_d(self, i):
        label = "Iter: {}\n Train Accuracy: ".format(i)
        # label = "Iter: {}\n Train Accuracy: random - {:.6f} | top - {:.6f} | bottom - {:.6f} |" \
        #         " all -  {:.6f}\n Test Accuracy: random - {:.6f} | top - {:.6f} | bottom - {:.6f} |" \
        #         " all - {:.6f}".format(i,

        # Calculate and plot lines
        for j in range(len(self.dbs)):
            self.lines[j].set_data(*self.calculate_db(*self.dbs[j][i]))

        self.ax.set_xlabel(label)

        return tuple(self.lines)

    def run(self, dbs, train_accs, test_accs, labels, colors, markers=None):
        assert len({len(dbs), len(train_accs), len(test_accs), len(labels), len(colors)}) == 1

        self.dbs, self.train_accs, self.test_accs = dbs, train_accs, test_accs
        self.labels, self.colors, self.markers = labels, colors, markers
        self.lines = []
        if markers is None:
            markers = ['--'] * len(self.lines)

        for i in range(len(dbs)):
            self.lines.append(self.ax.plot([], [], markers[i], lw=2, color=colors[i], label=labels[i])[0])

        anim_d = animation.FuncAnimation(self.fig, self.animate_d, init_func=self.init_d,
                                         frames=self.n_epochs, interval=self.interval, blit=True)

        return anim_d
