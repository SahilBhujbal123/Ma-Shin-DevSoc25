import subprocess
import time
import psutil
import os

class InfiniteBen
chmarker:
    def __init__(self, file_path):
        """Initialize the benchmarker with the file to test."""
        self.file_path = file_path
        self.process = None
        self.start_time = None
        self.cpu_usage = []
        self.memory_usage = []

    def run(self, duration=None):
        """
        Run the file and monitor performance.
        If duration is None, run indefinitely and provide real-time updates.
        If duration is specified, run for that duration and provide a final report.
        """
        print(f"Benchmarking file: {self.file_path}")
        self.start_time = time.time()

        # Start the process
        self.process = subprocess.Popen(self.file_path, shell=True)

        # Monitor CPU and memory usage
        try:
            while True:
                if self.process.poll() is not None:  # Process has ended
                    break

                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.Process(self.process.pid).memory_info().rss / (1024 * 1024)  # In MB
                self.cpu_usage.append(cpu_percent)
                self.memory_usage.append(memory_info)

                # Print real-time stats
                print(f"\n--- Real-Time Stats ---")
                print(f"CPU Usage: {cpu_percent:.2f}%")
                print(f"Memory Usage: {memory_info:.2f} MB")

                # Stop if duration is specified
                if duration and time.time() - self.start_time >= duration:
                    break

                time.sleep(1)  # Adjust sampling interval

        except KeyboardInterrupt:
            print("Benchmarking stopped by user.")

    def report(self):
        """Generate a performance report."""
        if not self.process:
            print("No process was run.")
            return

        execution_time = time.time() - self.start_time
        avg_cpu_usage = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        max_memory_usage = max(self.memory_usage) if self.memory_usage else 0

        print("\n--- Benchmark Report ---")
        print(f"File: {self.file_path}")
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"Average CPU Usage: {avg_cpu_usage:.2f}%")
        print(f"Max Memory Usage: {max_memory_usage:.2f} MB")

class FiniteBenchmarker:
    def __init__(self, file_path):
        """Initialize the benchmarker with the file to test."""
        self.file_path = file_path
        self.process = None
        self.start_time = None
        self.end_time = None
        self.cpu_usage = []
        self.memory_usage = []

    def run(self):
        """Run the file and monitor performance."""
        print(f"Benchmarking file: {self.file_path}")
        self.start_time = time.time()

        # Start the process
        self.process = subprocess.Popen(self.file_path, shell=True)

        # Monitor CPU and memory usage
        while self.process.poll() is None:  # While the process is running
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.Process(self.process.pid).memory_info().rss / (1024 * 1024)  # In MB
            self.cpu_usage.append(cpu_percent)
            self.memory_usage.append(memory_info)
            time.sleep(0.1)  # Adjust sampling interval

        self.end_time = time.time()

    def report(self):
        """Generate a performance report."""
        if not self.process:
            print("No process was run.")
            return

        execution_time = self.end_time - self.start_time
        avg_cpu_usage = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        max_memory_usage = max(self.memory_usage) if self.memory_usage else 0

        print("\n--- Benchmark Report ---")
        print(f"File: {self.file_path}")
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"Average CPU Usage: {avg_cpu_usage:.2f}%")
        print(f"Max Memory Usage: {max_memory_usage:.2f} MB")

'''
# Example Usage
if __name__ == "__main__":
    # Benchmark a Python script indefinitely
    benchmarker = InfiniteBenchmarker("python3 your_long_running_script.py")
    benchmarker.run()  # Run indefinitely with real-time updates

    # Benchmark a Python script for a specific duration
    # benchmarker = InfiniteBenchmarker("python3 your_long_running_script.py")
    # benchmarker.run(duration=30)  # Run for 30 seconds
    # benchmarker.report()
'''