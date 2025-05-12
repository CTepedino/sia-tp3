

import matplotlib.pyplot as plt

if __name__ == "__main__":

    linear_errs = [45.75026784705061, 40.90452265130926, 40.982959664391196, 41.778139760862324, 50.565365695267495]
    linear_lrs = [1e-06,1e-05, 0.0001, 0.001, 0.01]

    sigmoid_errs = [0.9320630519380739, 0.9310229505425544, 0.9309191486345094, 187.79060767996242, 931.0263225882493]
    sigmoid_lrs = [0.01, 0.001, 0.0001, 1e-05, 1e-06]

    tanh_errs = [0.9429304425079049, 0.9326799148539113, 0.9310915111910337, 1.5901470294430446, 1.590213098459928]
    tanh_lrs = [0.01, 0.001, 0.0001, 1e-05, 1e-06]

    plt.figure(figsize=(10, 6))
    plt.plot(linear_lrs, linear_errs, label='Linear', marker='o')
    plt.plot(sigmoid_lrs, sigmoid_errs, label='Sigmoid', marker='o')
    plt.plot(tanh_lrs, tanh_errs, label='Tanh', marker='o')


    plt.xscale('log')

    plt.xlabel('Learning Rate')
    plt.ylabel('Error promedio minimo')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.gca().invert_xaxis()

    plt.show()

