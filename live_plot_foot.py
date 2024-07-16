import matplotlib
matplotlib.use('GTK4Agg') 
print(matplotlib.get_backend())

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


# File paths
file_paths = {
    'FL_foot': 'FL_foot.csv',
    'FR_foot': 'FR_foot.csv',
    'RL_foot': 'RL_foot.csv',
    'RR_foot': 'RR_foot.csv',
}

# Initialize the plot
fig, ax = plt.subplots()

# Initialize lines for each file
lines = {}
for label in file_paths.keys():
    line, = ax.plot([], [], lw=2, label=label)
    lines[label] = line

# Set up the plot labels and limits
ax.set_xlabel('Iteration')
ax.set_ylabel('Foot height [m]')
ax.legend(loc='best')

ax.set_xlim(-0.1, 100.1)
ax.set_ylim(0.25, 0.50)

def init():
    for line in lines.values():
        line.set_data([], [])
    return lines.values()

def update(frame):
    try:
        for label, file_path in file_paths.items():
            if os.path.exists(file_path):
                # Load the data from CSV file
                data = np.loadtxt(file_path, delimiter=',')
                # Select only the third line (row index 2)
                data_to_plot = data
                x = np.arange(len(data_to_plot))
                y = data_to_plot
                lines[label].set_data(x, y)

            else:
                print(f"File {file_path} does not exist.")
        
        # Adjust x and y limits based on the data
        all_y_data = [lines[label].get_ydata() for label in file_paths.keys()]
        if all_y_data:
            y_min = min(-0.05,min(np.min(y) - 0.02 for y in all_y_data))
            y_max = max(0.15,max(np.max(y) + 0.02 for y in all_y_data))
            ax.set_xlim(0, len(data_to_plot) - 1)
            ax.set_ylim(y_min, y_max)

    except Exception as e:
        print(f"Error reading or processing the file: {e}")

    return lines.values()

# Create the animation
ani = animation.FuncAnimation(
    fig,
    update,
    init_func=init,
    blit=True,
    interval=100,
    cache_frame_data=False,
)

plt.show()

