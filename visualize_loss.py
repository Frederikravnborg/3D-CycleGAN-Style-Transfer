import csv
import matplotlib.pyplot as plt
import numpy as np

def visualize_loss():
    with open('loss.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        G_loss = []
        cycle_loss = []
        D_loss = []
        for row in reader:
            G_loss.append(float(row['G_loss']))
            cycle_loss.append(float(row['cycle_loss']))
            D_loss.append(float(row['D_loss']))
    #generate 2 plots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Loss')
    ax1.plot(G_loss, label='G_loss')
    ax1.plot(cycle_loss, label='cycle_loss')
    ax1.legend()
    ax2.plot(D_loss, label='D_loss')
    ax2.legend()
    plt.show()
    

if __name__ == '__main__':
    visualize_loss()
