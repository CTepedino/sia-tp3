

import matplotlib.pyplot as plt

if __name__ == "__main__":

    linear_errs = [40.89669593692596, 40.904528131948645, 40.982959664391196, 41.778139760862324, 50.565365695267495]
    linear_lrs = [1e-06,1e-05, 0.0001, 0.001, 0.01]

    sigmoid_errs = [0.9320630519380739, 0.9310229505425544, 0.9309191486345094, 0.9309087736450816, 0.9309079886345094]
    sigmoid_lrs = [0.01, 0.001, 0.0001, 1e-05, 1e-06]

    tanh_errs = [1.717341076342256, 1.6021465410849494, 1.5912325802934344, 1.5901470294430446, 1.590213098459928]
    tanh_lrs = [0.01, 0.001, 0.0001, 1e-05, 1e-06]

    plt.figure(figsize=(10, 6))
    plt.plot(linear_lrs, linear_errs, label='Linear', marker='o')
    plt.plot(sigmoid_lrs, sigmoid_errs, label='Sigmoid', marker='o')
    plt.plot(tanh_lrs, tanh_errs, label='Tanh', marker='o')


    plt.xscale('log')

    plt.xlabel('Learning Rate (escala logaritmica)')
    plt.ylabel('Error promedio minimo')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.gca().invert_xaxis()

    plt.show()

