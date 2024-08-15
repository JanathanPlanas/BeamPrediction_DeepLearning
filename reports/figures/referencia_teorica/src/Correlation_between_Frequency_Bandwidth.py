import matplotlib.pyplot as plt

frequencies = [5, 30, 300, 600]  # Em GHz
bandwidths = [0.1, 1, 10, 50]  # Em GHz

plt.figure(figsize=(10, 6))
plt.plot(frequencies, bandwidths, marker='o', linestyle='-', color='b')
plt.title('Correlation between Frequency and Bandwidth')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Bandwith (GHz)')
plt.grid(True)
plt.xticks([0, 100, 200, 300, 400, 500, 600])
plt.yticks([0, 10, 20, 30, 40, 50])
plt.show()