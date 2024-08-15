if True:
    import matplotlib.pyplot as plt

    frequencies = ['5.9 GHz', 'mmWave (30-300 GHz)', 'sub-Terahertz (>300 GHz)']
    range_values = [500, 200, 100]  
    bandwidth_values = [50, 1000, 3000]  

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Frequency Bands')
    ax1.set_ylabel('Range (m)', color=color)
    ax1.bar(frequencies, range_values, color=color, alpha=0.6)
    ax1.tick_params(axis='y',labelcolor=color)

   
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Bandwidth (MHz)', color=color)  
    ax2.plot(frequencies, bandwidth_values, color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)

    
    plt.title('Comparison of Frequency Bands for V2V Communication')
    fig.tight_layout()  
    plt.show()