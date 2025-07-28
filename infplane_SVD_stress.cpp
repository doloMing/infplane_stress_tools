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
#include <omp.h>

class SVDStressTester {
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
    
    // Generate random matrix
    std::vector<std::vector<double>> generateRandomMatrix(int rows, int cols) {
        std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
        std::mt19937 local_rng(std::chrono::steady_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix[i][j] = dist(local_rng);
            }
        }
        
        return matrix;
    }
    
    // Calculate FLOPS for SVD decomposition
    double calculateSVDFlops(int m, int n) {
        int minDim = (std::min)(m, n);
        return 8.0 * minDim * m * n + 4.0 * minDim * minDim * minDim;
    }
    
    // SVD result structure
    struct SVDResult {
        std::vector<std::vector<double>> U;
        std::vector<double> S;
        std::vector<std::vector<double>> V;
        bool success;
        double computationIntensity;
        int threadId;
        int matrixId;
    };
    
    // Matrix transpose
    std::vector<std::vector<double>> matrixTranspose(const std::vector<std::vector<double>>& A) {
        int m = static_cast<int>(A.size());
        int n = static_cast<int>(A[0].size());
        std::vector<std::vector<double>> AT(n, std::vector<double>(m));
        
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                AT[j][i] = A[i][j];
            }
        }
        
        return AT;
    }
    
    // Matrix multiplication
    std::vector<std::vector<double>> matrixMultiply(
        const std::vector<std::vector<double>>& A,
        const std::vector<std::vector<double>>& B) {
        
        int m = static_cast<int>(A.size());
        int n = static_cast<int>(B[0].size());
        int p = static_cast<int>(A[0].size());
        
        std::vector<std::vector<double>> C(m, std::vector<double>(n, 0.0));
        
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                double sum = 0.0;
                for (int k = 0; k < p; ++k) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
        
        return C;
    }
    
    // Vector normalization
    void normalizeVector(std::vector<double>& v) {
        double norm = 0.0;
        for (double val : v) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        
        if (norm > 1e-10) {
            for (double& val : v) {
                val /= norm;
            }
        }
    }
    
    // Independent SVD computation for each thread
    SVDResult performIndependentSVD(const std::vector<std::vector<double>>& matrix, int threadId, int matrixId) {
        SVDResult result;
        result.threadId = threadId;
        result.matrixId = matrixId;
        result.computationIntensity = 0.0;
        
        try {
            int m = matrix.size();
            int n = matrix[0].size();
            int minDim = (std::min)(m, n);
            
            // Initialize result matrices
            result.U.resize(m, std::vector<double>(minDim, 0.0));
            result.S.resize(minDim, 0.0);
            result.V.resize(n, std::vector<double>(minDim, 0.0));
            
            // Step 1: Compute A^T * A
            auto AT = matrixTranspose(matrix);
            auto ATA = matrixMultiply(AT, matrix);
            
            // Step 2: Power iteration for each singular vector
            for (int k = 0; k < minDim; ++k) {
                // Initialize random vector
                std::vector<double> v(n, 0.0);
                std::mt19937 local_rng(std::chrono::steady_clock::now().time_since_epoch().count() + threadId * 1000 + k);
                std::uniform_real_distribution<double> dist(-1.0, 1.0);
                
                for (int i = 0; i < n; ++i) {
                    v[i] = dist(local_rng);
                }
                
                // Power iteration
                for (int iter = 0; iter < 50; ++iter) {
                    std::vector<double> newV(n, 0.0);
                    
                    // Matrix-vector multiplication: ATA * v
                    for (int i = 0; i < n; ++i) {
                        double sum = 0.0;
                        for (int j = 0; j < n; ++j) {
                            sum += ATA[i][j] * v[j];
                        }
                        newV[i] = sum;
                    }
                    
                    normalizeVector(newV);
                    v = newV;
                }
                
                // Store right singular vector
                for (int i = 0; i < n; ++i) {
                    result.V[i][k] = v[i];
                }
                
                // Compute singular value: sqrt(v^T * ATA * v)
                double singularValue = 0.0;
                for (int i = 0; i < n; ++i) {
                    double temp = 0.0;
                    for (int j = 0; j < n; ++j) {
                        temp += ATA[i][j] * v[j];
                    }
                    singularValue += v[i] * temp;
                }
                result.S[k] = std::sqrt(std::abs(singularValue));
                
                // Compute corresponding left singular vector: u = A * v / sigma
                if (result.S[k] > 1e-10) {
                    std::vector<double> u(m, 0.0);
                    for (int i = 0; i < m; ++i) {
                        double sum = 0.0;
                        for (int j = 0; j < n; ++j) {
                            sum += matrix[i][j] * v[j];
                        }
                        u[i] = sum / result.S[k];
                    }
                    
                    // Store left singular vector
                    for (int i = 0; i < m; ++i) {
                        result.U[i][k] = u[i];
                    }
                }
            }
            
            // Additional computation for intensity
            double matrixNorm = 0.0;
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    matrixNorm += matrix[i][j] * matrix[i][j];
                }
            }
            result.computationIntensity = std::sqrt(matrixNorm);
            
            result.success = true;
        }
        catch (...) {
            result.success = false;
        }
        
        return result;
    }
    
    // Perform stress test with independent thread computations
    void performStressTest(int dimension, const std::string& testName) {
        std::cout << "Starting " << testName << " (" << dimension << "x" << dimension << ")" << std::endl;
        std::cout << "Using " << activeThreads << " independent threads (out of " << numThreads << " available)" << std::endl;
        
        // Generate multiple random matrices - one for each active thread
        std::vector<std::vector<std::vector<double>>> matrices(activeThreads);
        std::cout << "  Generating " << activeThreads << " random matrices..." << std::endl;
        
        for (int i = 0; i < activeThreads; ++i) {
            matrices[i] = generateRandomMatrix(dimension, dimension);
        }
        
        // Calculate theoretical FLOPS per thread
        double theoreticalFlopsPerThread = calculateSVDFlops(dimension, dimension);
        double totalTheoreticalFlops = theoreticalFlopsPerThread * activeThreads;
        
        std::cout << "  Starting independent SVD computations on " << activeThreads << " threads..." << std::endl;
        
        // Storage for results from each thread
        std::vector<SVDResult> results(activeThreads);
        
        // Record start time
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Perform independent SVD computations in parallel
        #pragma omp parallel for num_threads(activeThreads)
        for (int i = 0; i < activeThreads; ++i) {
            int threadId = omp_get_thread_num();
            results[i] = performIndependentSVD(matrices[i], threadId, i);
        }
        
        // Record end time
        auto endTime = std::chrono::high_resolution_clock::now();
        
        // Calculate elapsed time
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        double elapsedSeconds = duration.count() / 1000000.0;
        
        // Calculate performance metrics
        double totalActualFlops = totalTheoreticalFlops / elapsedSeconds;
        double totalGflops = totalActualFlops / 1e9;
        
        // Count successful computations
        int successCount = 0;
        double totalComputationIntensity = 0.0;
        
        for (const auto& result : results) {
            if (result.success) {
                successCount++;
                totalComputationIntensity += result.computationIntensity;
            }
        }
        
        // Log results
        std::cout << "  Matrix Dimension: " << dimension << "x" << dimension << std::endl;
        std::cout << "  Active Threads: " << activeThreads << std::endl;
        std::cout << "  Successful Computations: " << successCount << "/" << activeThreads << std::endl;
        std::cout << "  Total Elapsed Time: " << std::fixed << std::setprecision(6) << elapsedSeconds << " seconds" << std::endl;
        std::cout << "  Total Performance: " << std::fixed << std::setprecision(3) << totalGflops << " GFLOPS" << std::endl;
        std::cout << "  Average Performance per Thread: " << std::fixed << std::setprecision(3) << totalGflops / activeThreads << " GFLOPS" << std::endl;
        std::cout << "  Total Theoretical FLOPS: " << std::scientific << std::setprecision(2) << totalTheoreticalFlops << std::endl;
        std::cout << "  Total Computation Intensity: " << std::scientific << std::setprecision(2) << totalComputationIntensity << std::endl;
        std::cout << std::endl;
        
        // Write to log file
        logFile << "=== " << testName << " ===" << std::endl;
        logFile << "Timestamp: " << getCurrentTimestamp() << std::endl;
        logFile << "Matrix Dimension: " << dimension << "x" << dimension << std::endl;
        logFile << "Hardware Threads: " << numThreads << std::endl;
        logFile << "Active Threads: " << activeThreads << std::endl;
        logFile << "Successful Computations: " << successCount << "/" << activeThreads << std::endl;
        logFile << "Total Elapsed Time: " << std::fixed << std::setprecision(6) << elapsedSeconds << " seconds" << std::endl;
        logFile << "Total Performance: " << std::fixed << std::setprecision(3) << totalGflops << " GFLOPS" << std::endl;
        logFile << "Average Performance per Thread: " << std::fixed << std::setprecision(3) << totalGflops / activeThreads << " GFLOPS" << std::endl;
        logFile << "Total Theoretical FLOPS: " << std::scientific << std::setprecision(2) << totalTheoreticalFlops << std::endl;
        logFile << "Total Computation Intensity: " << std::scientific << std::setprecision(2) << totalComputationIntensity << std::endl;
        
        // Log individual thread results
        logFile << "Individual Thread Results:" << std::endl;
        for (int i = 0; i < activeThreads; ++i) {
            const auto& result = results[i];
            logFile << "  Thread " << result.threadId << " (Matrix " << result.matrixId << "): ";
            if (result.success) {
                logFile << "SUCCESS - First 3 singular values: ";
                int maxValues = (std::min)(3, static_cast<int>(result.S.size()));
                for (int j = 0; j < maxValues; ++j) {
                    logFile << std::fixed << std::setprecision(6) << result.S[j] << " ";
                }
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
    SVDStressTester() : rng(std::chrono::steady_clock::now().time_since_epoch().count()) {
        // Detect hardware threads
        numThreads = std::thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 8; // Fallback
        
        // Use n-2 threads for computation (reserve 2 for system)
        activeThreads = (std::max)(1, numThreads - 2);
        
        omp_set_num_threads(activeThreads);
        omp_set_dynamic(0); // Disable dynamic thread adjustment
        
        std::cout << "Independent SVD Stress Tester initialized" << std::endl;
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
        logFile << "Infplane Independent SVD Stress Test Log" << std::endl;
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
        std::cout << "=== Infplane Independent SVD Stress Tester ===" << std::endl;
        std::cout << "High-Intensity CPU Stress Test using Independent SVD Computations" << std::endl;
        std::cout << "Each thread performs complete SVD on its own matrix" << std::endl;
        std::cout << "Testing dimensions: 500, 1000, 1500" << std::endl;
        std::cout << "Target CPU utilization: >90%" << std::endl;
        std::cout << std::endl;
        
        // Test 1: 500x500 matrix
        performStressTest(500, "Small Matrix Independent SVD Test");
        
        // Test 2: 1000x1000 matrix
        performStressTest(1000, "Medium Matrix Independent SVD Test");
        
        // Test 3: 1500x1500 matrix
        std::cout << "WARNING: 1500x1500 matrix test will consume significant CPU resources!" << std::endl;
        std::cout << "This test will utilize " << activeThreads << " CPU cores at maximum capacity." << std::endl;
        std::cout << "Press any key to continue or Ctrl+C to abort..." << std::endl;
        system("pause");
        
        try {
            performStressTest(1500, "Large Matrix Independent SVD Test");
        }
        catch (const std::exception& e) {
            std::cout << "Error during large matrix test: " << e.what() << std::endl;
            logFile << "Large Matrix Independent SVD Test: FAILED - " << e.what() << std::endl;
        }
        catch (...) {
            std::cout << "Unknown error during large matrix test" << std::endl;
            logFile << "Large Matrix Independent SVD Test: FAILED - Unknown error" << std::endl;
        }
        
        // Write summary
        logFile << "========================================" << std::endl;
        logFile << "Test Session Completed: " << getCurrentTimestamp() << std::endl;
        logFile << "========================================" << std::endl;
        
        std::cout << "All independent SVD tests completed! Check the log file for detailed results." << std::endl;
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
    
    std::cout << "Initializing independent SVD stress tester..." << std::endl;
    
    SVDStressTester tester;
    
    // Initialize log file
    std::string logFilename = "infplane_SVD_stress.log";
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