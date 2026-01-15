import matplotlib.pyplot as plt
import csv
import os

def plot_action_logs(file_path="action_logs/throttle_steering_commands.csv", save_path="action_logs/action_commands_plot.png"):
    """
    Reads the throttle and steering commands from a CSV file and plots them over time.
    """
    throttle_commands = []
    steering_commands = []

    # Read data from CSV file
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for idx, row in enumerate(reader, 1):
            throttle_commands.append(float(row[0]))
            steering_commands.append(float(row[1]))

    timestamps = list(range(1, len(throttle_commands) + 1))

    # Plotting
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(timestamps, throttle_commands, label='Throttle Command', color='blue')
    plt.xlabel('Step Number')
    plt.ylabel('Throttle (rad/s)')
    plt.title('Throttle Commands Over Time')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(timestamps, steering_commands, label='Steering Command', color='orange')
    plt.xlabel('Step Number')
    plt.ylabel('Steering Angle (radians)')
    plt.title('Steering Commands Over Time')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    os.makedirs("action_logs", exist_ok=True)
    plot_action_logs()