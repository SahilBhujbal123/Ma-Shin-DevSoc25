#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <numeric>
#include <future>
#include <sstream>
#include <iomanip>
#include <array>
#include <cstdint>
#include <fstream>

// ---------------------
// Simple SHA-256 Implementation
// ---------------------
namespace sha256 {
    constexpr std::array<uint32_t, 64> K = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    inline uint32_t rotright(uint32_t value, unsigned int count) {
        return (value >> count) | (value << (32 - count));
    }

    inline uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) {
        return (x & y) ^ (~x & z);
    }

    inline uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) {
        return (x & y) ^ (x & z) ^ (y & z);
    }

    inline uint32_t Sigma0(uint32_t x) {
        return rotright(x, 2) ^ rotright(x, 13) ^ rotright(x, 22);
    }

    inline uint32_t Sigma1(uint32_t x) {
        return rotright(x, 6) ^ rotright(x, 11) ^ rotright(x, 25);
    }

    inline uint32_t sigma0(uint32_t x) {
        return rotright(x, 7) ^ rotright(x, 18) ^ (x >> 3);
    }

    inline uint32_t sigma1(uint32_t x) {
        return rotright(x, 17) ^ rotright(x, 19) ^ (x >> 10);
    }

    void sha256_transform(uint32_t state[8], const uint8_t block[64]) {
        uint32_t m[64];
        for (unsigned int i = 0; i < 16; ++i) {
            m[i] = (block[i * 4] << 24) | (block[i * 4 + 1] << 16) | (block[i * 4 + 2] << 8) | block[i * 4 + 3];
        }
        for (unsigned int i = 16; i < 64; ++i) {
            m[i] = sigma1(m[i - 2]) + m[i - 7] + sigma0(m[i - 15]) + m[i - 16];
        }

        uint32_t a = state[0], b = state[1], c = state[2], d = state[3],
                 e = state[4], f = state[5], g = state[6], h = state[7];

        for (unsigned int i = 0; i < 64; ++i) {
            uint32_t T1 = h + Sigma1(e) + Ch(e, f, g) + K[i] + m[i];
            uint32_t T2 = Sigma0(a) + Maj(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + T1;
            d = c;
            c = b;
            b = a;
            a = T1 + T2;
        }

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }

    void sha256(const uint8_t* data, size_t length, uint8_t hash[32]) {
        uint32_t state[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                               0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
        uint64_t bitlen = length * 8;
        uint8_t buffer[64];
        size_t buffer_len = 0;

        for (size_t i = 0; i < length; ++i) {
            buffer[buffer_len++] = data[i];
            if (buffer_len == 64) {
                sha256_transform(state, buffer);
                buffer_len = 0;
            }
        }

        buffer[buffer_len++] = 0x80;
        if (buffer_len > 56) {
            while (buffer_len < 64) buffer[buffer_len++] = 0;
            sha256_transform(state, buffer);
            buffer_len = 0;
        }

        while (buffer_len < 56) buffer[buffer_len++] = 0;
        for (int i = 7; i >= 0; --i)
            buffer[buffer_len++] = (bitlen >> (i * 8)) & 0xff;
        sha256_transform(state, buffer);

        for (int i = 0; i < 8; ++i) {
            hash[i * 4] = (state[i] >> 24) & 0xff;
            hash[i * 4 + 1] = (state[i] >> 16) & 0xff;
            hash[i * 4 + 2] = (state[i] >> 8) & 0xff;
            hash[i * 4 + 3] = state[i] & 0xff;
        }
    }
} // namespace sha256

// ---------------------
// Baseline Reference Values (Arbitrary)
// ---------------------
constexpr double BASELINE_MATRIX_GFLOPS    = 20.0;   // GFLOPS (higher is better)
constexpr double BASELINE_SORT_TIME        = 0.5;    // seconds (lower is better)
constexpr double BASELINE_PRIME_TIME       = 1.0;    // seconds (lower is better)
constexpr double BASELINE_FP_TIME          = 1.0;    // seconds (lower is better)
constexpr double BASELINE_MEMCPY_GBPS      = 4.0;    // GB/s (higher is better)
constexpr double BASELINE_CRYPTO_TIME      = 0.5;    // seconds (lower is better)
constexpr double BASELINE_MULTICORE_TIME   = 0.5;    // seconds (lower is better)
constexpr double BASELINE_IMAGE_TIME       = 0.3;    // seconds (lower is better)
constexpr double BASELINE_PHYSICS_TIME     = 0.5;    // seconds (lower is better)
constexpr double BASELINE_ML_TIME          = 1.0;    // seconds (lower is better)

// Use a relative path so the file is saved in the same directory as the executable.
const std::string RESULTS_PATH = "results.txt";

// Number of iterations (runs) per benchmark.
constexpr int NUM_RUNS = 10;

// ---------------------
// Benchmark 1: Matrix Multiplication (600x600 matrices)
// ---------------------
double runMatrixMultiplicationTest() {
    const int m = 600, n = 600, k = 600;
    std::vector<double> A(m * k);
    std::vector<double> B(k * n);
    std::vector<double> C(m * n, 0.0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    for (auto &a : A) a = dis(gen);
    for (auto &b : B) b = dis(gen);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
    auto finish = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(finish - start).count();
    double gflops = (2.0 * m * n * k) / (elapsed * 1e9);
    return gflops;
}

double runMatrixMultiplicationAggregate() {
    double sumGFLOPS = 0.0;
    for (int i = 0; i < NUM_RUNS; i++) {
        sumGFLOPS += runMatrixMultiplicationTest();
    }
    double avgGFLOPS = sumGFLOPS / NUM_RUNS;
    std::cout << "[Matrix Multiplication] Average GFLOPS: " << avgGFLOPS << "\n";
    return avgGFLOPS;
}

// ---------------------
// Benchmark 2: Sorting (20 million integers)
// ---------------------
double runSortingTest() {
    const size_t size = 20000000;
    std::vector<int> arr(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 1000000);
    for (auto &val : arr)
        val = dis(gen);

    auto start = std::chrono::high_resolution_clock::now();
    std::sort(arr.begin(), arr.end());
    auto finish = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(finish - start).count();
}

double runSortingAggregate() {
    double totalTime = 0.0;
    for (int i = 0; i < NUM_RUNS; i++) {
        totalTime += runSortingTest();
    }
    double avgTime = totalTime / NUM_RUNS;
    std::cout << "[Sorting] Average Time: " << avgTime << " seconds\n";
    return avgTime;
}

// ---------------------
// Benchmark 3: Prime Sieve (limit = 20 million)
// ---------------------
double runPrimeSieveTest() {
    int limit = 20000000;
    std::vector<bool> isPrime(limit + 1, true);
    if (limit >= 0) isPrime[0] = false;
    if (limit >= 1) isPrime[1] = false;

    auto start = std::chrono::high_resolution_clock::now();
    for (int p = 2; p * p <= limit; p++) {
        if (isPrime[p]) {
            for (int multiple = p * p; multiple <= limit; multiple += p) {
                isPrime[multiple] = false;
            }
        }
    }
    auto finish = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(finish - start).count();
}

double runPrimeSieveAggregate() {
    double totalTime = 0.0;
    for (int i = 0; i < NUM_RUNS; i++) {
        totalTime += runPrimeSieveTest();
    }
    double avgTime = totalTime / NUM_RUNS;
    std::cout << "[Prime Sieve] Average Time: " << avgTime << " seconds\n";
    return avgTime;
}

// ---------------------
// Benchmark 4: Floating-Point Operations (20 million iterations)
// ---------------------
double runFloatingPointOpsTest() {
    size_t iterations = 20000000;
    double result = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; i++) {
        result += std::sin(i) * std::cos(i);
    }
    auto finish = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(finish - start).count();
}

double runFloatingPointOpsAggregate() {
    double totalTime = 0.0;
    for (int i = 0; i < NUM_RUNS; i++) {
        totalTime += runFloatingPointOpsTest();
    }
    double avgTime = totalTime / NUM_RUNS;
    std::cout << "[Floating-Point Ops] Average Time: " << avgTime << " seconds\n";
    return avgTime;
}

// ---------------------
// Benchmark 5: Memory Copy Throughput (200 MB buffer, 100 iterations)
// ---------------------
double runMemoryCopyTest() {
    size_t bufferSize = 200 * 1024 * 1024;
    size_t iterations = 100;
    std::vector<char> src(bufferSize, 'a');
    std::vector<char> dest(bufferSize, 0);

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; i++) {
        std::memcpy(dest.data(), src.data(), bufferSize);
    }
    auto finish = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(finish - start).count();
    double throughputGBps = (double(bufferSize) * iterations) / (elapsed * 1e9);
    return throughputGBps;
}

double runMemoryCopyAggregate() {
    double totalThroughput = 0.0;
    for (int i = 0; i < NUM_RUNS; i++) {
        totalThroughput += runMemoryCopyTest();
    }
    double avgThroughput = totalThroughput / NUM_RUNS;
    std::cout << "[Memory Copy] Average Throughput: " << avgThroughput << " GB/s\n";
    return avgThroughput;
}

// ---------------------
// Benchmark 6: Cryptographic Hash (SHA-256) (10 MB data, 100 iterations)
// ---------------------
double runCryptoHashTest() {
    size_t dataSize = 10 * 1024 * 1024;
    size_t iterations = 100;
    std::vector<unsigned char> data(dataSize, 'x');
    std::vector<unsigned char> hash(32);

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; i++) {
        sha256::sha256(data.data(), dataSize, hash.data());
    }
    auto finish = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(finish - start).count();
}

double runCryptoHashAggregate() {
    double totalTime = 0.0;
    for (int i = 0; i < NUM_RUNS; i++) {
        totalTime += runCryptoHashTest();
    }
    double avgTime = totalTime / NUM_RUNS;
    std::cout << "[Crypto Hash] Average Time: " << avgTime << " seconds\n";
    return avgTime;
}

// ---------------------
// Benchmark 7: Multi-Core Summation (200 million elements)
// ---------------------
double runMultiCoreTest() {
    size_t arraySize = 200000000;
    size_t numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;

    std::vector<double> data(arraySize);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    for (auto &val : data)
        val = dis(gen);

    auto worker = [&data](size_t start, size_t end) -> double {
        return std::accumulate(data.begin() + start, data.begin() + end, 0.0);
    };

    std::vector<std::future<double>> futures;
    size_t blockSize = arraySize / numThreads;
    auto startTime = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < numThreads; i++) {
        size_t begin = i * blockSize;
        size_t end = (i == numThreads - 1) ? arraySize : begin + blockSize;
        futures.push_back(std::async(std::launch::async, worker, begin, end));
    }
    double sumMulti = 0.0;
    for (auto &f : futures)
        sumMulti += f.get();
    auto finishTime = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(finishTime - startTime).count();
}

double runMultiCoreAggregate() {
    double totalTime = 0.0;
    for (int i = 0; i < NUM_RUNS; i++) {
        totalTime += runMultiCoreTest();
    }
    double avgTime = totalTime / NUM_RUNS;
    std::cout << "[Multi-Core Summation] Average Time: " << avgTime << " seconds\n";
    return avgTime;
}

// ---------------------
// Benchmark 8: Image Processing (Blur Filter on a 4K image)
// ---------------------
double runImageProcessingTest() {
    int width = 3840, height = 2160;
    std::vector<unsigned char> image(width * height);
    std::vector<unsigned char> output(image.size(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);
    for (auto &pixel : image)
        pixel = static_cast<unsigned char>(dis(gen));

    auto start = std::chrono::high_resolution_clock::now();
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sum = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    sum += image[(y + dy) * width + (x + dx)];
                }
            }
            output[y * width + x] = static_cast<unsigned char>(sum / 9);
        }
    }
    auto finish = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(finish - start).count();
}

double runImageProcessingAggregate() {
    double totalTime = 0.0;
    for (int i = 0; i < NUM_RUNS; i++) {
        totalTime += runImageProcessingTest();
    }
    double avgTime = totalTime / NUM_RUNS;
    std::cout << "[Image Processing] Average Time: " << avgTime << " seconds\n";
    return avgTime;
}

// ---------------------
// Benchmark 9: Physics Simulation (N-Body Simulation)
// ---------------------
struct Particle {
    double x, y;
    double vx, vy;
};

double runPhysicsSimulationTest() {
    int numParticles = 5000;
    int iterations = 200;
    std::vector<Particle> particles(numParticles);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> posDis(-100.0, 100.0);
    std::uniform_real_distribution<double> velDis(-1.0, 1.0);
    for (auto &p : particles) {
        p.x = posDis(gen);
        p.y = posDis(gen);
        p.vx = velDis(gen);
        p.vy = velDis(gen);
    }
    const double G = 6.67430e-11;
    const double dt = 0.01;

    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < numParticles; i++) {
            double fx = 0.0, fy = 0.0;
            for (int j = 0; j < numParticles; j++) {
                if (i == j) continue;
                double dx = particles[j].x - particles[i].x;
                double dy = particles[j].y - particles[i].y;
                double distSq = dx * dx + dy * dy + 1e-10;
                double force = G / distSq;
                fx += force * dx / std::sqrt(distSq);
                fy += force * dy / std::sqrt(distSq);
            }
            particles[i].vx += fx * dt;
            particles[i].vy += fy * dt;
        }
        for (auto &p : particles) {
            p.x += p.vx * dt;
            p.y += p.vy * dt;
        }
    }
    auto finish = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(finish - start).count();
}

double runPhysicsSimulationAggregate() {
    double totalTime = 0.0;
    for (int i = 0; i < NUM_RUNS; i++) {
        totalTime += runPhysicsSimulationTest();
    }
    double avgTime = totalTime / NUM_RUNS;
    std::cout << "[Physics Simulation] Average Time: " << avgTime << " seconds\n";
    return avgTime;
}

// ---------------------
// Benchmark 10: Machine Learning (Logistic Regression)
// (20000 samples, 1000 features, 250 iterations)
// ---------------------
double sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

double runMachineLearningTest() {
    size_t numSamples = 20000;
    size_t numFeatures = 1000;
    int iterations = 250;
    double learningRate = 0.1;

    std::vector<std::vector<double>> X(numSamples, std::vector<double>(numFeatures));
    std::vector<int> y(numSamples, 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> featureDis(-1.0, 1.0);
    std::uniform_int_distribution<int> labelDis(0, 1);
    for (size_t i = 0; i < numSamples; i++) {
        for (size_t j = 0; j < numFeatures; j++) {
            X[i][j] = featureDis(gen);
        }
        y[i] = labelDis(gen);
    }
    std::vector<double> weights(numFeatures, 0.0);

    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        std::vector<double> gradients(numFeatures, 0.0);
        for (size_t i = 0; i < numSamples; i++) {
            double z = std::inner_product(X[i].begin(), X[i].end(), weights.begin(), 0.0);
            double prediction = sigmoid(z);
            double error = prediction - y[i];
            for (size_t j = 0; j < numFeatures; j++) {
                gradients[j] += error * X[i][j];
            }
        }
        for (size_t j = 0; j < numFeatures; j++) {
            weights[j] -= learningRate * gradients[j] / numSamples;
        }
    }
    auto finish = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(finish - start).count();
}

double runMachineLearningAggregate() {
    double totalTime = 0.0;
    for (int i = 0; i < NUM_RUNS; i++) {
        totalTime += runMachineLearningTest();
    }
    double avgTime = totalTime / NUM_RUNS;
    std::cout << "[Machine Learning] Average Time: " << avgTime << " seconds\n";
    return avgTime;
}

// ---------------------
// Function to Save Results to a CSV-like File in the Current Directory
// ---------------------
void saveResultsToFile(
    double overallScore,
    const std::vector<std::pair<std::string, double>>& scoreEntries,
    const std::string& filename
) {
    // Open the file for writing (will be created in the current directory)
    std::ofstream outFile(filename, std::ios::out | std::ios::trunc);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not create results file at " << filename << "!" << std::endl;
        return;
    }

    // Write CSV-like lines in the format: "Label",Value
    outFile << "\"Matrix Multiplication:\"" << scoreEntries[0].second << "\n";
    outFile << "\"Sorting\":" << scoreEntries[1].second << "\n";
    outFile << "\"Prime Sieve\":" << scoreEntries[2].second << "\n";
    outFile << "\"Floating-Point Operations\":" << scoreEntries[3].second << "\n";
    outFile << "\"Memory Copy\":" << scoreEntries[4].second << "\n";
    outFile << "\"Crypto Hash\":" << scoreEntries[5].second << "\n";
    outFile << "\"Multi-Core Summation\":" << scoreEntries[6].second << "\n";
    outFile << "\"Image Processing\":" << scoreEntries[7].second << "\n";
    outFile << "\"Physics Simulation\":" << scoreEntries[8].second << "\n";
    outFile << "\"Machine Learning\":" << scoreEntries[9].second << "\n";
    outFile << "\"Overall Score\":" << overallScore << "\n";

    outFile.flush();
    outFile.close();
    std::cout << "Results successfully saved to: " << filename << std::endl;
}

// ---------------------
// MAIN: Run All Benchmarks, Compute Scores, and Save Results
// ---------------------
int main() {
    std::cout << "Extended CPU Benchmark Suite (Parallel, Aggregated over " 
              << NUM_RUNS << " runs per test)" << "\n";
    std::cout << "-------------------------------------------------------------\n\n";

    // Launch each aggregated benchmark concurrently.
    auto futMatrix = std::async(std::launch::async, runMatrixMultiplicationAggregate);
    auto futSort   = std::async(std::launch::async, runSortingAggregate);
    auto futPrime  = std::async(std::launch::async, runPrimeSieveAggregate);
    auto futFP     = std::async(std::launch::async, runFloatingPointOpsAggregate);
    auto futMemcpy = std::async(std::launch::async, runMemoryCopyAggregate);
    auto futCrypto = std::async(std::launch::async, runCryptoHashAggregate);
    auto futMulti  = std::async(std::launch::async, runMultiCoreAggregate);
    auto futImage  = std::async(std::launch::async, runImageProcessingAggregate);
    auto futPhys   = std::async(std::launch::async, runPhysicsSimulationAggregate);
    auto futML     = std::async(std::launch::async, runMachineLearningAggregate);

    // Get aggregated metrics.
    double avgGFLOPS    = futMatrix.get();   // Higher is better.
    double avgSortTime  = futSort.get();     // Lower is better.
    double avgPrimeTime = futPrime.get();    // Lower is better.
    double avgFPTime    = futFP.get();       // Lower is better.
    double avgMemcpyGB  = futMemcpy.get();     // Higher is better.
    double avgCryptoTime= futCrypto.get();     // Lower is better.
    double avgMultiTime = futMulti.get();      // Lower is better.
    double avgImageTime = futImage.get();      // Lower is better.
    double avgPhysicsTime = futPhys.get();     // Lower is better.
    double avgMLTime    = futML.get();         // Lower is better.

    // Compute normalized scores.
    double scoreMatrix = avgGFLOPS / BASELINE_MATRIX_GFLOPS;
    double scoreSort   = BASELINE_SORT_TIME / avgSortTime;
    double scorePrime  = BASELINE_PRIME_TIME / avgPrimeTime;
    double scoreFP     = BASELINE_FP_TIME / avgFPTime;
    double scoreMemcpy = avgMemcpyGB / BASELINE_MEMCPY_GBPS;
    double scoreCrypto = BASELINE_CRYPTO_TIME / avgCryptoTime;
    double scoreMulti  = BASELINE_MULTICORE_TIME / avgMultiTime;
    double scoreImage  = BASELINE_IMAGE_TIME / avgImageTime;
    double scorePhysics= BASELINE_PHYSICS_TIME / avgPhysicsTime;
    double scoreML     = BASELINE_ML_TIME / avgMLTime;

    std::cout << "\nNormalized Scores (Reference = 1.0):\n";
    std::cout << "  Matrix Multiplication: " << scoreMatrix << "\n";
    std::cout << "  Sorting:               " << scoreSort   << "\n";
    std::cout << "  Prime Sieve:           " << scorePrime  << "\n";
    std::cout << "  Floating-Point Ops:    " << scoreFP     << "\n";
    std::cout << "  Memory Copy:           " << scoreMemcpy << "\n";
    std::cout << "  Crypto Hash:           " << scoreCrypto << "\n";
    std::cout << "  Multi-Core Sum:        " << scoreMulti  << "\n";
    std::cout << "  Image Processing:      " << scoreImage  << "\n";
    std::cout << "  Physics Simulation:    " << scorePhysics<< "\n";
    std::cout << "  Machine Learning:      " << scoreML     << "\n";

    // Compute overall score as the geometric mean.
    std::vector<double> scores = { scoreMatrix, scoreSort, scorePrime, scoreFP,
                                   scoreMemcpy, scoreCrypto, scoreMulti,
                                   scoreImage, scorePhysics, scoreML };
    double product = 1.0;
    for (double s : scores)
        product *= s;
    double overallScore = std::pow(product, 1.0 / scores.size());
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nOverall CPU Benchmark Score: " << overallScore << "\n";

    // Assemble score entries for saving.
    std::vector<std::pair<std::string, double>> scoreEntries = {
        {"Matrix Multiplication", scoreMatrix},
        {"Sorting", scoreSort},
        {"Prime Sieve", scorePrime},
        {"Floating-Point Operations", scoreFP},
        {"Memory Copy", scoreMemcpy},
        {"Crypto Hash", scoreCrypto},
        {"Multi-Core Summation", scoreMulti},
        {"Image Processing", scoreImage},
        {"Physics Simulation", scorePhysics},
        {"Machine Learning", scoreML}
    };

    // Save the results to a CSV-like file in the current directory.
    saveResultsToFile(overallScore, scoreEntries, RESULTS_PATH);
    std::cout << "\nResults saved to: " << RESULTS_PATH << "\n";

    std::cout << "Press Enter to exit...";
    //std::cin.get();
    return 0;
}
