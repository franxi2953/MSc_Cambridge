import serial
import matplotlib.pyplot as plt
import time

# Set up serial communication
ser = serial.Serial('COM8', 115200, timeout=1)
time.sleep(2)  # Wait for the serial connection to initialize

# Lists to store the data
amplified_voltages = []
offsets = []

# Enable interactive mode in matplotlib
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(offsets, amplified_voltages, 'r-')  # Red line plot
plt.xlabel('Offset')
plt.ylabel('Amplified Voltage')
plt.title('Real-time Plot of Amplified Voltage vs Offset')

try:
    while True:
        ser.write(b'$')
        line = ser.readline().decode('utf-8').strip()
        parts = line.split(',')
        if len(parts) >= 3:
            try:
                amplified_voltage = float(parts[0].split(':')[1])
                offset = float(parts[2].split(':')[1])

                # Append data to the lists
                amplified_voltages.append(amplified_voltage)
                offsets.append(offset)

                # Update the plot
                line.set_xdata(offsets)
                line.set_ydata(amplified_voltages)
                ax.relim()  # Recalculate limits
                ax.autoscale_view(True, True, True)  # Rescale the plot
                fig.canvas.draw()
                fig.canvas.flush_events()

            except ValueError:
                print(f"Error parsing line: {line}")

        time.sleep(1)  # Short delay

except KeyboardInterrupt:
    print("Program interrupted by the user")a

finally:
    ser.close()
    plt.ioff()  # Disable interactive mode
    plt.show()  # Show the final plot
