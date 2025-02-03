// benchmark_industry.cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <complex>
#include <algorithm>
#include <thread>
#include <random>
#include <numeric>
#include <cstddef>
#include <functional>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <limits>

#ifdef _OPENMP
  #include <omp.h>
#endif

// OS-specific includes for affinity and CPU settings
#ifdef _WIN32
  #include <windows.h>
#else
  // Linux (and macOS may support some of these, but this code is tested on Linux)
  #include <sched.h>
  #include <unistd.h>
#endif

using namespace std;
using namespace chrono;

// Dummy declaration to resolve std::byte ambiguity.
std::byte myByte;

// ==================== Global Configuration ====================
struct Config {
    // Problem sizes
    int gemm_size    = 512;
    int fft_size     = 1 << 18;   // 262144 points
    int sort_size    = 1 << 24;   // ~16.7 million elements
    int fib_number   = 40;
    int mandel_size  = 2048;
    int stream_size  = 1 << 24;
    int sieve_limit  = 10000000;  // for prime sieve benchmark
    int encrypt_size = 1 << 26;   // about 67 million bytes

    // Benchmark parameters
    int warmup_runs   = 2;
    int measured_runs = 5;

    // CPU settings
    bool multi_thread  = true;  // true = multi-core; false = single-core
    int  affinity_core = 0;      // pin main thread to this core (if supported)

    // Thermal throttling (Linux only)
    double max_temp = 85.0;      // Celsius; if exceeded, pause benchmarking
};

Config config;

// ==================== Utility Functions ====================

// Run a given benchmark lambda function for 'runs' iterations and return the median time.
double run_benchmark(function<double(void)> benchFunc, int runs) {
    vector<double> times;
    for (int i = 0; i < runs; i++) {
        double t = benchFunc();
        times.push_back(t);
    }
    sort(times.begin(), times.end());
    return times[runs / 2];
}

// Set core affinity for the current thread.
bool set_core_affinity(int core_id) {
#ifdef _WIN32
    HANDLE thread = GetCurrentThread();
    DWORD_PTR mask = (1ULL << core_id);
    DWORD_PTR result = SetThreadAffinityMask(thread, mask);
    return (result != 0);
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    int result = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    return (result == 0);
#endif
}

// Attempt to force the CPU governor to "performance" (Linux only).
void set_performance_governor() {
#ifndef _WIN32
    int numCpus = sysconf(_SC_NPROCESSORS_CONF);
    for (int i = 0; i < numCpus; i++) {
        ostringstream path;
        path << "/sys/devices/system/cpu/cpu" << i << "/cpufreq/scaling_governor";
        ofstream governorFile(path.str());
        if (governorFile.is_open()) {
            governorFile << "performance";
            governorFile.close();
        } else {
            cerr << "Warning: Could not set CPU " << i << " governor to performance. (May need root privileges)\n";
        }
    }
#else
    // On Windows, dynamic frequency scaling is controlled by power settings.
    cout << "Note: On Windows, please ensure your power plan is set to High Performance.\n";
#endif
}

// Read CPU temperature (Linux only). Returns temperature in Celsius.
// On Windows (or if unavailable) returns -1.
double get_cpu_temperature() {
#ifndef _WIN32
    ifstream tempFile("/sys/class/thermal/thermal_zone0/temp");
    if (tempFile.is_open()) {
        double temp;
        tempFile >> temp;
        tempFile.close();
        return temp / 1000.0; // Convert from millidegrees to degrees
    }
#endif
    return -1.0;
}

// Thermal monitor thread function: if temperature exceeds threshold, pause briefly.
void thermal_monitor(double max_temp) {
#ifndef _WIN32
    while (true) {
        double temp = get_cpu_temperature();
        if (temp > 0 && temp > max_temp) {
            cout << "Thermal Warning: CPU temperature " << temp 
                 << "°C exceeds threshold of " << max_temp 
                 << "°C. Pausing benchmarks for cooling...\n";
            this_thread::sleep_for(seconds(5));
        } else {
            this_thread::sleep_for(seconds(1));
        }
    }
#else
    // On Windows, we do not implement thermal monitoring here.
    while (false) { }
#endif
}

// ==================== Benchmark Workloads ====================

// GEMM Benchmark (Blocked matrix multiplication)
double benchmark_gemm(int N) {
    vector<vector<double>> A(N, vector<double>(N, 1.0));
    vector<vector<double>> B(N, vector<double>(N, 1.0));
    vector<vector<double>> C(N, vector<double>(N, 0.0));
    auto start = high_resolution_clock::now();
    const int block_size = 32;
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(dynamic) if(config.multi_thread)
#endif
    for (int i = 0; i < N; i += block_size) {
        for (int j = 0; j < N; j += block_size) {
            for (int k = 0; k < N; k += block_size) {
                for (int ii = i; ii < min(i + block_size, N); ++ii) {
                    for (int kk = k; kk < min(k + block_size, N); ++kk) {
                        for (int jj = j; jj < min(j + block_size, N); ++jj) {
                            C[ii][jj] += A[ii][kk] * B[kk][jj];
                        }
                    }
                }
            }
        }
    }
    auto end = high_resolution_clock::now();
    return duration<double>(end - start).count();
}

// FFT Benchmark (Cooley-Tukey)
void fft(vector<complex<double>>& x) {
    size_t N = x.size();
    if (N <= 1) return;
    vector<complex<double>> even(N / 2), odd(N / 2);
    for (size_t i = 0; i < N / 2; ++i) {
        even[i] = x[2 * i];
        odd[i]  = x[2 * i + 1];
    }
    fft(even);
    fft(odd);
    for (size_t k = 0; k < N / 2; ++k) {
        complex<double> t = polar(1.0, -2 * M_PI * k / N) * odd[k];
        x[k] = even[k] + t;
        x[k + N / 2] = even[k] - t;
    }
}

double benchmark_fft(int N) {
    vector<complex<double>> signal(N);
    for (int i = 0; i < N; ++i)
        signal[i] = complex<double>(cos(2 * M_PI * i / N), 0);
    auto start = high_resolution_clock::now();
    fft(signal);
    auto end = high_resolution_clock::now();
    return duration<double>(end - start).count();
}

// MergeSort Benchmark
void merge(vector<int>& arr, int l, int m, int r) {
    vector<int> temp(r - l + 1);
    int i = l, j = m + 1, k = 0;
    while (i <= m && j <= r) {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }
    while (i <= m) temp[k++] = arr[i++];
    while (j <= r) temp[k++] = arr[j++];
    for (int p = 0; p < k; p++)
        arr[l + p] = temp[p];
}

void merge_sort(vector<int>& arr, int l, int r) {
    if (l >= r) return;
    int m = l + (r - l) / 2;
    merge_sort(arr, l, m);
    merge_sort(arr, m + 1, r);
    merge(arr, l, m, r);
}

double benchmark_sort(int N) {
    vector<int> arr(N);
    mt19937 rng(42);
    uniform_int_distribution<int> dist(1, N * 10);
    generate(arr.begin(), arr.end(), [&]() { return dist(rng); });
    auto start = high_resolution_clock::now();
    merge_sort(arr, 0, N - 1);
    auto end = high_resolution_clock::now();
    return duration<double>(end - start).count();
}

// Recursive Fibonacci (not parallelized)
int fib(int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}

double benchmark_fib(int n) {
    auto start = high_resolution_clock::now();
    volatile int result = fib(n);
    auto end = high_resolution_clock::now();
    return duration<double>(end - start).count();
}

// Mandelbrot Benchmark
double benchmark_mandelbrot(int size) {
    vector<int> output(size * size);
    const int max_iter = 1000;
    const double x_min = -2.0, x_max = 1.0;
    const double y_min = -1.2, y_max = 1.2;
    auto start = high_resolution_clock::now();
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) if(config.multi_thread)
#endif
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            double zx = 0.0, zy = 0.0;
            double cx = x_min + (x_max - x_min) * x / size;
            double cy = y_min + (y_max - y_min) * y / size;
            int iter = 0;
            while (zx * zx + zy * zy < 4.0 && iter < max_iter) {
                double tmp = zx * zx - zy * zy + cx;
                zy = 2.0 * zx * zy + cy;
                zx = tmp;
                iter++;
            }
            output[y * size + x] = iter;
        }
    }
    auto end = high_resolution_clock::now();
    return duration<double>(end - start).count();
}

// STREAM Triad Benchmark
double benchmark_stream(int N) {
    vector<double> a(N), b(N), c(N);
    const double scalar = 3.0;
    iota(a.begin(), a.end(), 0.0);
    iota(b.begin(), b.end(), 1.0);
    iota(c.begin(), c.end(), 2.0);
    auto start = high_resolution_clock::now();
#ifdef _OPENMP
    #pragma omp parallel for if(config.multi_thread)
#endif
    for (int i = 0; i < N; i++) {
        a[i] = b[i] + scalar * c[i];
    }
    auto end = high_resolution_clock::now();
    return duration<double>(end - start).count();
}

// --- Additional Workloads ---

// Prime Sieve Benchmark: Count primes up to 'limit' using the Sieve of Eratosthenes.
double benchmark_prime_sieve(int limit) {
    auto start = high_resolution_clock::now();
    vector<bool> is_prime(limit + 1, true);
    if (limit >= 0) {
        is_prime[0] = is_prime[1] = false;
    }
    for (int i = 2; i * i <= limit; i++) {
        if (is_prime[i]) {
            for (int j = i * i; j <= limit; j += i) {
                is_prime[j] = false;
            }
        }
    }
    int count = 0;
    for (bool prime : is_prime)
        if (prime) count++;
    auto end = high_resolution_clock::now();
    return duration<double>(end - start).count();
}

// XOR Encryption Benchmark: Perform XOR on a large buffer repeatedly.
double benchmark_encryption(int size) {
    vector<unsigned char> data(size, 0);
    for (int i = 0; i < size; i++)
        data[i] = static_cast<unsigned char>(i % 256);
    unsigned char key = 0xAA;
    auto start = high_resolution_clock::now();
    for (int pass = 0; pass < 10; pass++) {
#ifdef _OPENMP
        #pragma omp parallel for if(config.multi_thread)
#endif
        for (int i = 0; i < size; i++) {
            data[i] ^= key;
        }
    }
    auto end = high_resolution_clock::now();
    return duration<double>(end - start).count();
}

// ==================== Main Function ====================
int main() {
    // Set core affinity.
    if (!set_core_affinity(config.affinity_core)) {
        cerr << "Warning: Unable to set core affinity.\n";
    } else {
        cout << "Core affinity set to core " << config.affinity_core << ".\n";
    }

    // Attempt to force performance governor (Linux only).
    set_performance_governor();

    // Start thermal monitor in a separate thread (only active on Linux).
#ifndef _WIN32
    thread thermalThread(thermal_monitor, config.max_temp);
    thermalThread.detach();
#else
    cout << "Thermal monitoring is not implemented on Windows.\n";
#endif

    cout << "\n=== Comprehensive CPU Benchmark ===\n";

    // Warmup runs.
    cout << "Performing warmup runs...\n";
    for (int i = 0; i < config.warmup_runs; i++) {
        benchmark_gemm(256);
        benchmark_fft(1024);
        benchmark_sort(1 << 16);
        benchmark_fib(20);
        benchmark_mandelbrot(512);
        benchmark_stream(1 << 20);
        benchmark_prime_sieve(100000);
        benchmark_encryption(1 << 20);
    }

    // Run measured benchmarks (5 times each, using median).
    double gemm_time   = run_benchmark([&](){ return benchmark_gemm(config.gemm_size); }, config.measured_runs);
    double fft_time    = run_benchmark([&](){ return benchmark_fft(config.fft_size); }, config.measured_runs);
    double sort_time   = run_benchmark([&](){ return benchmark_sort(config.sort_size); }, config.measured_runs);
    double fib_time    = run_benchmark([&](){ return benchmark_fib(config.fib_number); }, config.measured_runs);
    double mandel_time = run_benchmark([&](){ return benchmark_mandelbrot(config.mandel_size); }, config.measured_runs);
    double stream_time = run_benchmark([&](){ return benchmark_stream(config.stream_size); }, config.measured_runs);
    double sieve_time  = run_benchmark([&](){ return benchmark_prime_sieve(config.sieve_limit); }, config.measured_runs);
    double encrypt_time= run_benchmark([&](){ return benchmark_encryption(config.encrypt_size); }, config.measured_runs);

    // Output individual benchmark times.
    cout << "GEMM (" << config.gemm_size << "x" << config.gemm_size << "): " << gemm_time << " s\n";
    cout << "FFT (" << config.fft_size << " points): " << fft_time << " s\n";
    cout << "MergeSort (" << config.sort_size << " elements): " << sort_time << " s\n";
    cout << "Fibonacci (n=" << config.fib_number << "): " << fib_time << " s\n";
    cout << "Mandelbrot (" << config.mandel_size << "x" << config.mandel_size << "): " << mandel_time << " s\n";
    cout << "STREAM Triad (" << config.stream_size << " elements): " << stream_time << " s\n";
    cout << "Prime Sieve (up to " << config.sieve_limit << "): " << sieve_time << " s\n";
    cout << "XOR Encryption (" << config.encrypt_size << " bytes, 10 passes): " << encrypt_time << " s\n";

    // Convert times to performance (reciprocal, higher is better).
    auto perf = [](double t) -> double { return (t > 0) ? (1.0 / t) : 0.0; };
    double perf_gemm    = perf(gemm_time);
    double perf_fft     = perf(fft_time);
    double perf_sort    = perf(sort_time);
    double perf_fib     = perf(fib_time);
    double perf_mandel  = perf(mandel_time);
    double perf_stream  = perf(stream_time);
    double perf_sieve   = perf(sieve_time);
    double perf_encrypt = perf(encrypt_time);

    // Weights for each benchmark (these should ideally sum to 1).
    const double w_gemm    = 0.20;
    const double w_fft     = 0.10;
    const double w_sort    = 0.10;
    const double w_fib     = 0.10;
    const double w_mandel  = 0.15;
    const double w_stream  = 0.10;
    const double w_sieve   = 0.15;
    const double w_encrypt = 0.10;

    double final_score = w_gemm * perf_gemm + w_fft * perf_fft + w_sort * perf_sort +
                         w_fib * perf_fib + w_mandel * perf_mandel + w_stream * perf_stream +
                         w_sieve * perf_sieve + w_encrypt * perf_encrypt;

    cout << "\nFinal Performance Score: " << final_score << "\n";
    cout << "Press Enter to exit...";
    cin.get();
    return 0;
}
