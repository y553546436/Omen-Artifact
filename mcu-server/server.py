import serial
import struct
import numpy as np
import sys
from rich.progress import track
from time import sleep
import argparse


parser = argparse.ArgumentParser(description='MCU Serial Test')
parser.add_argument('-D', '--device', type=str, help='Serial device', required=True)
parser.add_argument('-s', '--start', type=int, help='Test start index')
parser.add_argument('-e', '--end', type=int, help='Test end index')
parser.add_argument('-n', '--number', type=int, help='Number of tests')
parser.add_argument('-O', '--output', type=str, help='Output file', default='serial_test_data.csv')
parser.add_argument('-d', '--dataset', type=str, help='Dataset name', required=True)
args = parser.parse_args()

serial_device = args.device

# Initialize UART connection
ser = serial.Serial(serial_device, 115200, timeout=1)

# Define the 2D array of floating points
x_test = np.load(f'data/{args.dataset}/test.npy')
y_test = np.load(f'data/{args.dataset}/testlbl.npy')
n = x_test.shape[0]

print("Data loaded")
print("Number of test data: ", n)
print("Data shape: ", x_test.shape)
print("Data type: ", x_test.dtype)
if args.start is not None:
    print("Start index: ", args.start)
if args.end is not None:
    print("End index: ", args.end)
if args.number is not None:
    print("Number of tests: ", args.number)
print("Serial device: ", serial_device)

print("Serial Test Start")

print("Data format: (Normal Inferred Class, Normal Time, Omen Dim Use, Omen Inferred Class, Omen Time, Absolute Inferred Class, Absolute Time, Diff Inferred Class, Diff Time, Mean Inferred Class, Mean Time, Actual Class)")

data = []
success = True
interrupted = False
try:
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    test_start = args.start if args.start is not None else 0
    test_end = args.end if args.end is not None else test_start + args.number if args.number is not None else n
    for i in track(range(test_start, test_end, 1), description="Testing..."):
        print("Test ", i)
        # Flatten the 2D array to 1D
        flattened_data = x_test[i].flatten()

        # Send the data
        test_feature = struct.pack('f' * len(flattened_data), *flattened_data)
        print(f'Sending {len(test_feature)} bytes')
        ser.write(test_feature)
        
        # Receive data
        while ser.in_waiting < 4 * 14:
            sleep(0.1)
        received_data = ser.read(4 * 14)
        print(f'Received {len(received_data)} bytes')
        # convert to uint32
        received_data = list(struct.unpack('14I', received_data))
        received_data[1] = received_data[1] / 10
        received_data[4] = received_data[4] / 10
        received_data[7] = received_data[7] / 10
        received_data[10] = received_data[10] / 10
        received_data[13] = received_data[13] / 10
        data.append(received_data + [int(y_test[i])])
        print(data[-1])
except KeyboardInterrupt:
    print("Test interrupted")
    success = False
    interrupted = True
except Exception as e:
    print(e)
    print("Test failed")
    success = False
finally:
    # Close the UART connection
    # clear the buffer before closing
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    ser.close()

    if not interrupted:
        with open(args.output, 'w') as f:
            for item in data:
                f.write(",".join(str(n) for n in item) + "\n")
        print(f"Serial Test End. Data saved in {args.output}")

    metrics = np.array(data)
    normal_time = metrics[:, 1]
    omen_time = metrics[:, 4]
    absolute_time = metrics[:, 7]
    diff_time = metrics[:, 10]
    mean_time = metrics[:, 13]
    avg_omen_dim = np.mean(metrics[:, 2])
    avg_absolute_dim = np.mean(metrics[:, 5])
    avg_diff_dim = np.mean(metrics[:, 8])
    avg_mean_dim = np.mean(metrics[:, 11])
    print("Normal Time Mean: ", np.mean(normal_time))
    print("Normal Time Std: ", np.std(normal_time))
    print("Omen Time Mean: ", np.mean(omen_time))
    print("Omen Time Std: ", np.std(omen_time))
    print("Absolute Time Mean: ", np.mean(absolute_time))
    print("Absolute Time Std: ", np.std(absolute_time))
    print("Diff Time Mean: ", np.mean(diff_time))
    print("Diff Time Std: ", np.std(diff_time))
    print("Mean Time Mean: ", np.mean(mean_time))
    print("Mean Time Std: ", np.std(mean_time))
    print("Normal Accuracy: ", np.mean(metrics[:, 0] == metrics[:, -1]))
    print("Omen Accuracy: ", np.mean(metrics[:, 3] == metrics[:, -1]))
    print("Absolute Accuracy: ", np.mean(metrics[:, 6] == metrics[:, -1]))
    print("Diff Accuracy: ", np.mean(metrics[:, 9] == metrics[:, -1]))
    print("Mean Accuracy: ", np.mean(metrics[:, 12] == metrics[:, -1]))
    print("Average Omen Dim Used: ", avg_omen_dim)
    print("Average Absolute Dim Used: ", avg_absolute_dim)
    print("Average Diff Dim Used: ", avg_diff_dim)
    print("Average Mean Dim Used: ", avg_mean_dim)
    print("Number of tests: ", len(metrics))
    if not success:
        print("Test failed with errors")
        exit(1)
