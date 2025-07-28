#include <windows.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <thread>
#include <complex>
#include <omp.h>

// Define M_PI if not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class FFTStressTester {
private:
    std::ofstream logFile;
    std::mt19937 rng;
    int numThreads;
    int activeThreads;  // n-2 threads for computation
    
    // Get current timestamp as string
    std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
    
    // Generate random complex vector
    std::vector<std::complex<double>> generateRandomVector(int size) {
        std::vector<std::complex<double>> vector(size);
        std::mt19937 local_rng(std::chrono::steady_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        
        for (int i = 0; i < size; ++i) {
            double real = dist(local_rng);
            double imag = dist(local_rng);
            vector[i] = std::complex<double>(real, imag);
        }
        
        return vector;
    }
    
    // Calculate FLOPS for FFT computation
    double calculateFFTFlops(int n) {
        // FFT complexity is O(n log n) with approximately 5n log2(n) operations
        if (n <= 1) return 0.0;
        return 5.0 * n * std::log2(n);
    }
    
    // FFT result structure
    struct FFTResult {
        std::vector<std::complex<double>> forward;
        std::vector<std::complex<double>> inverse;
        bool success;
        double computationIntensity;
        int threadId;
        int vectorId;
        double maxMagnitude;
        double totalEnergy;
    };
    
    // Bit reversal for FFT
    void bitReverse(std::vector<std::complex<double>>& data) {
        int n = data.size();
        int j = 0;
        
        for (int i = 1; i < n; ++i) {
            int bit = n >> 1;
            while (j & bit) {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            
            if (i < j) {
                std::swap(data[i], data[j]);
            }
        }
    }
    
    // Cooley-Tukey FFT algorithm (radix-2)
    void performFFT(std::vector<std::complex<double>>& data, bool inverse = false) {
        int n = data.size();
        
        // Bit reversal
        bitReverse(data);
        
        // FFT computation
        for (int len = 2; len <= n; len <<= 1) {
            double angle = 2.0 * M_PI / len * (inverse ? 1 : -1);
            std::complex<double> wlen(std::cos(angle), std::sin(angle));
            
            for (int i = 0; i < n; i += len) {
                std::complex<double> w(1.0, 0.0);
                for (int j = 0; j < len / 2; ++j) {
                    std::complex<double> u = data[i + j];
                    std::complex<double> v = data[i + j + len / 2] * w;
                    data[i + j] = u + v;
                    data[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
        
        // Normalize for inverse FFT
        if (inverse) {
            for (auto& x : data) {
                x /= n;
            }
        }
    }
    
    // Pad vector to next power of 2
    int nextPowerOf2(int n) {
        int power = 1;
        while (power < n) {
            power <<= 1;
        }
        return power;
    }
    
    // Calculate vector magnitude
    double calculateMagnitude(const std::vector<std::complex<double>>& data) {
        double maxMag = 0.0;
        for (const auto& val : data) {
            double mag = std::abs(val);
            if (mag > maxMag) {
                maxMag = mag;
            }
        }
        return maxMag;
    }
    
    // Calculate total energy
    double calculateEnergy(const std::vector<std::complex<double>>& data) {
        double energy = 0.0;
        for (const auto& val : data) {
            energy += std::norm(val);
        }
        return energy;
    }
    
    // Independent FFT computation for each thread
    FFTResult performIndependentFFT(const std::vector<std::complex<double>>& inputVector, int threadId, int vectorId) {
        FFTResult result;
        result.threadId = threadId;
        result.vectorId = vectorId;
        result.computationIntensity = 0.0;
        result.maxMagnitude = 0.0;
        result.totalEnergy = 0.0;
        
        try {
            int originalSize = inputVector.size();
            int paddedSize = nextPowerOf2(originalSize);
            
            // Create padded vector for FFT
            std::vector<std::complex<double>> paddedVector(paddedSize);
            for (int i = 0; i < originalSize; ++i) {
                paddedVector[i] = inputVector[i];
            }
            for (int i = originalSize; i < paddedSize; ++i) {
                paddedVector[i] = std::complex<double>(0.0, 0.0);
            }
            
            // Perform multiple FFT operations for intensive computation
            std::vector<std::complex<double>> workingVector = paddedVector;
            
            // Forward FFT
            performFFT(workingVector, false);
            result.forward = workingVector;
            
            // Calculate magnitude and energy
            result.maxMagnitude = calculateMagnitude(workingVector);
            result.totalEnergy = calculateEnergy(workingVector);
            
            // Inverse FFT
            performFFT(workingVector, true);
            result.inverse = workingVector;
            
            // Additional intensive FFT operations for stress testing
            for (int iter = 0; iter < 10; ++iter) {
                std::vector<std::complex<double>> tempVector = paddedVector;
                
                // Add some noise to make computation more intensive
                std::mt19937 local_rng(std::chrono::steady_clock::now().time_since_epoch().count() + threadId * 1000 + iter);
                std::uniform_real_distribution<double> noise(-0.01, 0.01);
                
                for (auto& val : tempVector) {
                    val += std::complex<double>(noise(local_rng), noise(local_rng));
                }
                
                // Forward FFT
                performFFT(tempVector, false);
                
                // Inverse FFT
                performFFT(tempVector, true);
                
                // Accumulate computation intensity
                result.computationIntensity += calculateEnergy(tempVector);
            }
            
            // Additional computation for intensity measurement
            result.computationIntensity += result.totalEnergy;
            
            result.success = true;
        }
        catch (...) {
            result.success = false;
        }
        
        return result;
    }
    
    // Perform stress test with independent thread computations
    void performStressTest(int vectorSize, const std::string& testName) {
        std::cout << "Starting " << testName << " (Vector size: " << vectorSize << ")" << std::endl;
        std::cout << "Using " << activeThreads << " independent threads (out of " << numThreads << " available)" << std::endl;
        
        // Generate multiple random vectors - one for each active thread
        std::vector<std::vector<std::complex<double>>> vectors(activeThreads);
        std::cout << "  Generating " << activeThreads << " random complex vectors..." << std::endl;
        
        for (int i = 0; i < activeThreads; ++i) {
            vectors[i] = generateRandomVector(vectorSize);
        }
        
        // Calculate theoretical FLOPS per thread
        int paddedSize = nextPowerOf2(vectorSize);
        double theoreticalFlopsPerThread = calculateFFTFlops(paddedSize) * 12; // 12 FFT operations per thread
        double totalTheoreticalFlops = theoreticalFlopsPerThread * activeThreads;
        
        std::cout << "  Starting independent FFT computations on " << activeThreads << " threads..." << std::endl;
        std::cout << "  Padded vector size: " << paddedSize << " (next power of 2)" << std::endl;
        
        // Storage for results from each thread
        std::vector<FFTResult> results(activeThreads);
        
        // Record start time
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Perform independent FFT computations in parallel
        #pragma omp parallel for num_threads(activeThreads)
        for (int i = 0; i < activeThreads; ++i) {
            int threadId = omp_get_thread_num();
            results[i] = performIndependentFFT(vectors[i], threadId, i);
        }
        
        // Record end time
        auto endTime = std::chrono::high_resolution_clock::now();
        
        // Calculate elapsed time
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        double elapsedSeconds = duration.count() / 1000000.0;
        
        // Calculate performance metrics
        double totalActualFlops = totalTheoreticalFlops / elapsedSeconds;
        double totalGflops = totalActualFlops / 1e9;
        
        // Count successful computations and gather statistics
        int successCount = 0;
        double totalComputationIntensity = 0.0;
        double maxMagnitude = 0.0;
        double totalEnergy = 0.0;
        
        for (const auto& result : results) {
            if (result.success) {
                successCount++;
                totalComputationIntensity += result.computationIntensity;
                if (result.maxMagnitude > maxMagnitude) {
                    maxMagnitude = result.maxMagnitude;
                }
                totalEnergy += result.totalEnergy;
            }
        }
        
        // Log results
        std::cout << "  Vector Size: " << vectorSize << " (padded to " << paddedSize << ")" << std::endl;
        std::cout << "  Active Threads: " << activeThreads << std::endl;
        std::cout << "  Successful Computations: " << successCount << "/" << activeThreads << std::endl;
        std::cout << "  Total Elapsed Time: " << std::fixed << std::setprecision(6) << elapsedSeconds << " seconds" << std::endl;
        std::cout << "  Total Performance: " << std::fixed << std::setprecision(3) << totalGflops << " GFLOPS" << std::endl;
        std::cout << "  Average Performance per Thread: " << std::fixed << std::setprecision(3) << totalGflops / activeThreads << " GFLOPS" << std::endl;
        std::cout << "  Total Theoretical FLOPS: " << std::scientific << std::setprecision(2) << totalTheoreticalFlops << std::endl;
        std::cout << "  Total Computation Intensity: " << std::scientific << std::setprecision(2) << totalComputationIntensity << std::endl;
        std::cout << "  Max FFT Magnitude: " << std::scientific << std::setprecision(2) << maxMagnitude << std::endl;
        std::cout << "  Total Energy: " << std::scientific << std::setprecision(2) << totalEnergy << std::endl;
        std::cout << std::endl;
        
        // Write to log file
        logFile << "=== " << testName << " ===" << std::endl;
        logFile << "Timestamp: " << getCurrentTimestamp() << std::endl;
        logFile << "Vector Size: " << vectorSize << " (padded to " << paddedSize << ")" << std::endl;
        logFile << "Hardware Threads: " << numThreads << std::endl;
        logFile << "Active Threads: " << activeThreads << std::endl;
        logFile << "Successful Computations: " << successCount << "/" << activeThreads << std::endl;
        logFile << "Total Elapsed Time: " << std::fixed << std::setprecision(6) << elapsedSeconds << " seconds" << std::endl;
        logFile << "Total Performance: " << std::fixed << std::setprecision(3) << totalGflops << " GFLOPS" << std::endl;
        logFile << "Average Performance per Thread: " << std::fixed << std::setprecision(3) << totalGflops / activeThreads << " GFLOPS" << std::endl;
        logFile << "Total Theoretical FLOPS: " << std::scientific << std::setprecision(2) << totalTheoreticalFlops << std::endl;
        logFile << "Total Computation Intensity: " << std::scientific << std::setprecision(2) << totalComputationIntensity << std::endl;
        logFile << "Max FFT Magnitude: " << std::scientific << std::setprecision(2) << maxMagnitude << std::endl;
        logFile << "Total Energy: " << std::scientific << std::setprecision(2) << totalEnergy << std::endl;
        
        // Log individual thread results
        logFile << "Individual Thread Results:" << std::endl;
        for (int i = 0; i < activeThreads; ++i) {
            const auto& result = results[i];
            logFile << "  Thread " << result.threadId << " (Vector " << result.vectorId << "): ";
            if (result.success) {
                logFile << "SUCCESS - Max magnitude: " << std::scientific << std::setprecision(3) << result.maxMagnitude;
                logFile << ", Energy: " << std::scientific << std::setprecision(3) << result.totalEnergy;
            } else {
                logFile << "FAILED";
            }
            logFile << std::endl;
        }
        
        logFile << "----------------------------------------" << std::endl;
        logFile << std::endl;
        logFile.flush();
    }
    
public:
    FFTStressTester() : rng(std::chrono::steady_clock::now().time_since_epoch().count()) {
        // Detect hardware threads
        numThreads = std::thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 8; // Fallback
        
        // Use n-2 threads for computation (reserve 2 for system)
        activeThreads = (std::max)(1, numThreads - 2);
        
        omp_set_num_threads(activeThreads);
        omp_set_dynamic(0); // Disable dynamic thread adjustment
        
        std::cout << "Independent FFT Stress Tester initialized" << std::endl;
        std::cout << "Hardware threads detected: " << numThreads << std::endl;
        std::cout << "Active computation threads: " << activeThreads << std::endl;
    }
    
    // Initialize log file
    bool initializeLog(const std::string& filename) {
        logFile.open(filename, std::ios::out | std::ios::app);
        if (!logFile.is_open()) {
            std::cerr << "Error: Cannot create log file " << filename << std::endl;
            return false;
        }
        
        logFile << "========================================" << std::endl;
        logFile << "Infplane Independent FFT Stress Test Log" << std::endl;
        logFile << "Test Session Started: " << getCurrentTimestamp() << std::endl;
        logFile << "Hardware Threads: " << numThreads << std::endl;
        logFile << "Active Computation Threads: " << activeThreads << std::endl;
        #ifdef _OPENMP
        logFile << "OpenMP Version: " << _OPENMP << std::endl;
        #else
        logFile << "OpenMP: Not Available" << std::endl;
        #endif
        logFile << "========================================" << std::endl;
        logFile << std::endl;
        
        return true;
    }
    
    // Run all stress tests
    void runAllTests() {
        std::cout << "=== Infplane Independent FFT Stress Tester ===" << std::endl;
        std::cout << "High-Intensity CPU Stress Test using Independent FFT Computations" << std::endl;
        std::cout << "Each thread performs complete FFT operations on its own vector" << std::endl;
        std::cout << "Testing vector sizes: 2097152, 4194304, 8388608" << std::endl;
        std::cout << "Target CPU utilization: >90%" << std::endl;
        std::cout << std::endl;
        
        // Test 1: 2097152 elements (2^21)
        performStressTest(2097152, "Small Vector Independent FFT Test");
        
        // Test 2: 4194304 elements (2^22)
        performStressTest(4194304, "Medium Vector Independent FFT Test");
        
        // Test 3: 8388608 elements (2^23)
        std::cout << "WARNING: 8388608 element vector test will consume significant CPU resources!" << std::endl;
        std::cout << "This test will utilize " << activeThreads << " CPU cores at maximum capacity." << std::endl;
        std::cout << "Press any key to continue or Ctrl+C to abort..." << std::endl;
        system("pause");
        
        try {
            performStressTest(8388608, "Large Vector Independent FFT Test");
        }
        catch (const std::exception& e) {
            std::cout << "Error during large vector test: " << e.what() << std::endl;
            logFile << "Large Vector Independent FFT Test: FAILED - " << e.what() << std::endl;
        }
        catch (...) {
            std::cout << "Unknown error during large vector test" << std::endl;
            logFile << "Large Vector Independent FFT Test: FAILED - Unknown error" << std::endl;
        }
        
        // Write summary
        logFile << "========================================" << std::endl;
        logFile << "Test Session Completed: " << getCurrentTimestamp() << std::endl;
        logFile << "========================================" << std::endl;
        
        std::cout << "All independent FFT tests completed! Check the log file for detailed results." << std::endl;
        std::cout << "Monitor Task Manager to verify high CPU utilization during tests." << std::endl;
    }
    
    // Close log file
    void closeLog() {
        if (logFile.is_open()) {
            logFile.close();
        }
    }
};

int main() {
    // Set console to support Unicode
    SetConsoleOutputCP(CP_UTF8);
    
    std::cout << "Initializing independent FFT stress tester..." << std::endl;
    
    FFTStressTester tester;
    
    // Initialize log file
    std::string logFilename = "infplane_FFT_stress.log";
    if (!tester.initializeLog(logFilename)) {
        std::cerr << "Failed to initialize log file. Exiting." << std::endl;
        return 1;
    }
    
    std::cout << "Log file: " << logFilename << std::endl;
    std::cout << std::endl;
    
    try {
        // Run all stress tests
        tester.runAllTests();
    }
    catch (const std::exception& e) {
        std::cerr << "Error during stress test: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown error during stress test" << std::endl;
        return 1;
    }
    
    // Close log file
    tester.closeLog();
    
    std::cout << "Press any key to exit..." << std::endl;
    system("pause");
    
    return 0;
} 