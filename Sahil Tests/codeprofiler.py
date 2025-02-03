import time
import psutil
import tracemalloc
import pynvml
import subprocess

class CodeProfiler:
    def __init__(self, command):
    
        self.command = command
        self.start_time = None
        self.end_time = None
        self.cpu_start = None
        self.cpu_end = None
        self.memory_start = None
        self.memory_end = None
        self.gpu_start = None
        self.gpu_end = None
        tracemalloc.start()
        pynvml.nvmlInit()

    def get_gpu_usage(self):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        except:
            return "N/A"

    def start(self):
        
        self.start_time = time.perf_counter()
        self.cpu_start = psutil.cpu_percent(interval=None)
        self.memory_start = psutil.virtual_memory().used
        self.gpu_start = self.get_gpu_usage()
        tracemalloc.reset_peak()

    def stop(self):
        
        self.end_time = time.perf_counter()
        self.cpu_end = psutil.cpu_percent(interval=None)
        self.memory_end = psutil.virtual_memory().used
        self.gpu_end = self.get_gpu_usage()

    def report(self):
        exec_time = self.end_time - self.start_time
        cpu_usage = self.cpu_end - self.cpu_start
        mem_usage = (self.memory_end - self.memory_start) / (1024 * 1024)  # MB
        peak_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)  # MB
        gpu_usage = self.gpu_end

        print("\n--- Code Profiling Report ---")
        print(f"Execution Time: {exec_time:.4f} sec")
        print(f"CPU Usage Change: {cpu_usage:.2f}%")
        print(f"Memory Used: {mem_usage:.2f} MB")
        print(f"Peak Memory Usage: {peak_mem:.2f} MB")
        print(f"GPU Usage: {gpu_usage}%")
        tracemalloc.stop()

    def run_and_profile(self):
        """Run a script or executable and monitor performance."""
        self.start()
        process = subprocess.Popen(self.command, shell=True)
        process.wait()  # Wait for script to finish
        self.stop()
        self.report()

# Example Usage:
profiler = CodeProfiler("python temp.py")  # Run a Python file
# profiler = CodeProfiler("./a.out")  # Run a compiled C++ program
# profiler = CodeProfiler("java MyProgram")  # Run a Java program

profiler.run_and_profile()
