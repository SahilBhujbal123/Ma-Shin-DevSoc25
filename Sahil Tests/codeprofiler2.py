import time
import psutil
import os
import tracemalloc
import pynvml

class CodeProfiler:
    def __init__(self):
        """Initialize the profiler."""
        self.process = psutil.Process(os.getpid())  # Track this script's process
        tracemalloc.start()
        pynvml.nvmlInit()
    
    def get_gpu_usage(self):
        """Get GPU memory used by this process instead of total system GPU usage."""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Get GPU handle
            pid = os.getpid()  # Get current process ID
            gpu_mem_used = 0

            # Get all running processes using the GPU
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if proc.pid == pid:  # Match our process ID
                    gpu_mem_used = proc.usedGpuMemory / (1024 * 1024)  # Convert to MB

            return gpu_mem_used  # Returns GPU memory used in MB

        except:
            return "N/A"

    def start(self):
        """Start tracking process-specific CPU & memory usage."""
        self.start_time = time.perf_counter()
        self.cpu_start = self.process.cpu_times().user + self.process.cpu_times().system  # CPU time
        self.memory_start = self.process.memory_info().rss / (1024 * 1024)  # Memory in MB
        tracemalloc.reset_peak()

    def stop(self):
        """Stop tracking and calculate process-specific usage."""
        self.end_time = time.perf_counter()
        self.cpu_end = self.process.cpu_times().user + self.process.cpu_times().system
        self.memory_end = self.process.memory_info().rss / (1024 * 1024)
        self.gpu_usage = self.get_gpu_usage()

    def report(self):
        """Generate a profiling report."""
        exec_time = self.end_time - self.start_time
        cpu_usage = self.cpu_end - self.cpu_start  # CPU time spent only by this script
        mem_usage = self.memory_end - self.memory_start  # Memory used only by this script
        peak_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)  # Peak memory usage
        gpu_usage = self.gpu_usage

        print("\n--- Code Profiling Report ---")
        print(f"Execution Time: {exec_time:.4f} sec")
        print(f"CPU Time Used: {cpu_usage:.4f} sec")  # More stable than %
        print(f"Memory Used: {mem_usage:.2f} MB")
        print(f"Peak Memory Usage: {peak_mem:.2f} MB")
        print(f"GPU Memory Used: {gpu_usage} MB")  # Now tracks per-process GPU memory
        tracemalloc.stop()

# Example Usage:
profiler = CodeProfiler()
profiler.start()

# --- Your actual code ---
array = []
for i in range(100021):
    array.append(i)
# ------------------------

profiler.stop()
profiler.report()

pynvml.nvmlShutdown()
