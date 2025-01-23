import matplotlib.pyplot as plt

class Visualization:
    @staticmethod
    def plot_signal(signal, title="Signal", xlabel="Time", ylabel="Amplitude"):
        plt.figure(figsize=(10, 5))
        plt.plot(signal)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()