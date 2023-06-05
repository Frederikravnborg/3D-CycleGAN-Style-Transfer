import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

def visualize_loss():
    filename = "output/loss_06.05.10.59.18.csv"

    # read data from csv file using pandas
    meta = pd.read_csv(filename, nrows=1)
    data = pd.read_csv(filename ,skiprows=1)
    print(meta)

    # extract data from dataframe
    epoch = data['epoch']
    G_loss = []
    cycle_loss = []
    D_loss = []

    indexes = []
    step = 0
    for i in range (len(epoch)):
        indexes = data.index.get_indexer(data.query(f'epoch == {step}').index)
        G_loss.append(data['G_loss'][indexes].mean())
        cycle_loss.append(data['cycle_loss'][indexes].mean())
        D_loss.append(data['D_loss'][indexes].mean())
        step += 1

    # generate 2 plots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(G_loss, label='G_loss')
    # ax1.plot(cycle_loss, label='cycle_loss')
    ax1.legend()
    # add labels
    ax1.set(xlabel='Epoch', ylabel='Loss', title='Losses')
    ax2.plot(D_loss, label='D_loss')
    ax2.legend()
    ax2.set(xlabel='Epoch', ylabel='Loss', title='discriminator loss')
    plt.show()


if __name__ == '__main__':
    visualize_loss()
