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

class SGDStressTester {
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
    
    // Generate random matrix (samples x features)
    std::vector<std::vector<double>> generateRandomMatrix(int rows, int cols) {
        std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
        std::mt19937 local_rng(std::chrono::steady_clock::now().time_since_epoch().count());
        std::normal_distribution<double> dist(0.0, 1.0);
        
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix[i][j] = dist(local_rng);
            }
        }
        
        return matrix;
    }
    
    // Generate random vector
    std::vector<double> generateRandomVector(int size) {
        std::vector<double> vector(size);
        std::mt19937 local_rng(std::chrono::steady_clock::now().time_since_epoch().count());
        std::normal_distribution<double> dist(0.0, 1.0);
        
        for (int i = 0; i < size; ++i) {
            vector[i] = dist(local_rng);
        }
        
        return vector;
    }
    
    // Calculate FLOPS for SGD computation
    double calculateSGDFlops(int samples, int features, int iterations) {
        // Each SGD iteration: forward pass + backward pass + weight update
        // Forward pass: samples * features multiplications
        // Backward pass: samples * features multiplications for gradient
        // Weight update: features operations
        // Total per iteration: 2 * samples * features + features
        double flopsPerIteration = 2.0 * samples * features + features;
        return flopsPerIteration * iterations;
    }
    
    // SGD result structure
    struct SGDResult {
        std::vector<double> finalWeights;
        double finalLoss;
        double initialLoss;
        bool success;
        double computationIntensity;
        int threadId;
        int matrixId;
        int iterations;
        double convergenceRate;
        std::vector<double> lossHistory;
    };
    
    // Matrix-vector multiplication
    std::vector<double> matrixVectorMultiply(const std::vector<std::vector<double>>& matrix, 
                                           const std::vector<double>& vector) {
        int rows = matrix.size();
        int cols = matrix[0].size();
        std::vector<double> result(rows, 0.0);
        
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        
        return result;
    }
    
    // Calculate mean squared error loss
    double calculateMSELoss(const std::vector<double>& predictions, const std::vector<double>& targets) {
        double loss = 0.0;
        int n = predictions.size();
        
        for (int i = 0; i < n; ++i) {
            double diff = predictions[i] - targets[i];
            loss += diff * diff;
        }
        
        return loss / n;
    }
    
    // Calculate gradient for linear regression
    std::vector<double> calculateGradient(const std::vector<std::vector<double>>& X, 
                                        const std::vector<double>& predictions,
                                        const std::vector<double>& targets) {
        int samples = X.size();
        int features = X[0].size();
        std::vector<double> gradient(features, 0.0);
        
        for (int j = 0; j < features; ++j) {
            for (int i = 0; i < samples; ++i) {
                double error = predictions[i] - targets[i];
                gradient[j] += (2.0 / samples) * error * X[i][j];
            }
        }
        
        return gradient;
    }
    
    // Independent SGD computation for each thread
    SGDResult performIndependentSGD(const std::vector<std::vector<double>>& X,
                                   const std::vector<double>& Y,
                                   int threadId, int matrixId) {
        SGDResult result;
        result.threadId = threadId;
        result.matrixId = matrixId;
        result.computationIntensity = 0.0;
        result.iterations = 500;
        result.convergenceRate = 0.0;
        result.success = false;
        
        try {
            int samples = X.size();
            int features = X[0].size();
            
            // Initialize weights randomly
            std::vector<double> weights(features);
            std::mt19937 local_rng(std::chrono::steady_clock::now().time_since_epoch().count() + threadId * 1000);
            std::normal_distribution<double> weight_dist(0.0, 0.1);
            
            for (int i = 0; i < features; ++i) {
                weights[i] = weight_dist(local_rng);
            }
            
            // SGD parameters
            double learningRate = 0.01;
            double momentum = 0.9;
            std::vector<double> velocity(features, 0.0);
            
            // Store loss history for convergence analysis
            result.lossHistory.reserve(result.iterations);
            
            // Calculate initial loss
            std::vector<double> predictions = matrixVectorMultiply(X, weights);
            result.initialLoss = calculateMSELoss(predictions, Y);
            
            // SGD optimization loop
            for (int iter = 0; iter < result.iterations; ++iter) {
                // Forward pass
                predictions = matrixVectorMultiply(X, weights);
                
                // Calculate loss
                double currentLoss = calculateMSELoss(predictions, Y);
                result.lossHistory.push_back(currentLoss);
                
                // Calculate gradient
                std::vector<double> gradient = calculateGradient(X, predictions, Y);
                
                // Update weights with momentum
                for (int j = 0; j < features; ++j) {
                    velocity[j] = momentum * velocity[j] - learningRate * gradient[j];
                    weights[j] += velocity[j];
                }
                
                // Accumulate computation intensity
                result.computationIntensity += currentLoss;
                
                // Adaptive learning rate (simple decay)
                if (iter > 0 && iter % 100 == 0) {
                    learningRate *= 0.95;
                }
                
                // Additional intensive computations for stress testing
                if (iter % 10 == 0) {
                    // Perform extra matrix operations for CPU stress
                    std::vector<double> tempWeights = weights;
                    for (int extra = 0; extra < 5; ++extra) {
                        std::vector<double> tempPredictions = matrixVectorMultiply(X, tempWeights);
                        std::vector<double> tempGradient = calculateGradient(X, tempPredictions, Y);
                        
                        // Accumulate computation intensity
                        result.computationIntensity += calculateMSELoss(tempPredictions, Y);
                        
                        // Modify temp weights slightly
                        for (int k = 0; k < features; ++k) {
                            tempWeights[k] += 0.001 * tempGradient[k];
                        }
                    }
                }
            }
            
            // Final evaluation
            predictions = matrixVectorMultiply(X, weights);
            result.finalLoss = calculateMSELoss(predictions, Y);
            result.finalWeights = weights;
            
            // Calculate convergence rate
            if (result.initialLoss > 0.0) {
                result.convergenceRate = (result.initialLoss - result.finalLoss) / result.initialLoss;
            }
            
            result.success = true;
        }
        catch (...) {
            result.success = false;
        }
        
        return result;
    }
    
    // Perform stress test with independent thread computations
    void performStressTest(int matrixSize, const std::string& testName) {
        std::cout << "Starting " << testName << " (Matrix size: " << matrixSize << "x" << matrixSize << ")" << std::endl;
        std::cout << "Using " << activeThreads << " independent threads (out of " << numThreads << " available)" << std::endl;
        
        // Generate multiple random matrices and vectors - one for each active thread
        std::vector<std::vector<std::vector<double>>> matrices(activeThreads);
        std::vector<std::vector<double>> targets(activeThreads);
        
        std::cout << "  Generating " << activeThreads << " random matrices and target vectors..." << std::endl;
        
        for (int i = 0; i < activeThreads; ++i) {
            matrices[i] = generateRandomMatrix(matrixSize, matrixSize);
            targets[i] = generateRandomVector(matrixSize);
        }
        
        // Calculate theoretical FLOPS per thread
        double theoreticalFlopsPerThread = calculateSGDFlops(matrixSize, matrixSize, 500);
        double totalTheoreticalFlops = theoreticalFlopsPerThread * activeThreads;
        
        std::cout << "  Starting independent SGD optimizations on " << activeThreads << " threads..." << std::endl;
        std::cout << "  SGD iterations per thread: 500" << std::endl;
        std::cout << "  Problem: minimize ||WX - Y||^2 where W is " << matrixSize << "x" << matrixSize << std::endl;
        
        // Storage for results from each thread
        std::vector<SGDResult> results(activeThreads);
        
        // Record start time
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Perform independent SGD optimizations in parallel
        #pragma omp parallel for num_threads(activeThreads)
        for (int i = 0; i < activeThreads; ++i) {
            int threadId = omp_get_thread_num();
            results[i] = performIndependentSGD(matrices[i], targets[i], threadId, i);
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
        double avgInitialLoss = 0.0;
        double avgFinalLoss = 0.0;
        double avgConvergenceRate = 0.0;
        
        for (const auto& result : results) {
            if (result.success) {
                successCount++;
                totalComputationIntensity += result.computationIntensity;
                avgInitialLoss += result.initialLoss;
                avgFinalLoss += result.finalLoss;
                avgConvergenceRate += result.convergenceRate;
            }
        }
        
        if (successCount > 0) {
            avgInitialLoss /= successCount;
            avgFinalLoss /= successCount;
            avgConvergenceRate /= successCount;
        }
        
        // Log results
        std::cout << "  Matrix Size: " << matrixSize << "x" << matrixSize << std::endl;
        std::cout << "  Active Threads: " << activeThreads << std::endl;
        std::cout << "  Successful Optimizations: " << successCount << "/" << activeThreads << std::endl;
        std::cout << "  Total Elapsed Time: " << std::fixed << std::setprecision(6) << elapsedSeconds << " seconds" << std::endl;
        std::cout << "  Total Performance: " << std::fixed << std::setprecision(3) << totalGflops << " GFLOPS" << std::endl;
        std::cout << "  Average Performance per Thread: " << std::fixed << std::setprecision(3) << totalGflops / activeThreads << " GFLOPS" << std::endl;
        std::cout << "  Total Theoretical FLOPS: " << std::scientific << std::setprecision(2) << totalTheoreticalFlops << std::endl;
        std::cout << "  Total Computation Intensity: " << std::scientific << std::setprecision(2) << totalComputationIntensity << std::endl;
        std::cout << "  Average Initial Loss: " << std::scientific << std::setprecision(3) << avgInitialLoss << std::endl;
        std::cout << "  Average Final Loss: " << std::scientific << std::setprecision(3) << avgFinalLoss << std::endl;
        std::cout << "  Average Convergence Rate: " << std::fixed << std::setprecision(3) << avgConvergenceRate * 100 << "%" << std::endl;
        std::cout << std::endl;
        
        // Write to log file
        logFile << "=== " << testName << " ===" << std::endl;
        logFile << "Timestamp: " << getCurrentTimestamp() << std::endl;
        logFile << "Matrix Size: " << matrixSize << "x" << matrixSize << std::endl;
        logFile << "Hardware Threads: " << numThreads << std::endl;
        logFile << "Active Threads: " << activeThreads << std::endl;
        logFile << "SGD Iterations: 500" << std::endl;
        logFile << "Successful Optimizations: " << successCount << "/" << activeThreads << std::endl;
        logFile << "Total Elapsed Time: " << std::fixed << std::setprecision(6) << elapsedSeconds << " seconds" << std::endl;
        logFile << "Total Performance: " << std::fixed << std::setprecision(3) << totalGflops << " GFLOPS" << std::endl;
        logFile << "Average Performance per Thread: " << std::fixed << std::setprecision(3) << totalGflops / activeThreads << " GFLOPS" << std::endl;
        logFile << "Total Theoretical FLOPS: " << std::scientific << std::setprecision(2) << totalTheoreticalFlops << std::endl;
        logFile << "Total Computation Intensity: " << std::scientific << std::setprecision(2) << totalComputationIntensity << std::endl;
        logFile << "Average Initial Loss: " << std::scientific << std::setprecision(3) << avgInitialLoss << std::endl;
        logFile << "Average Final Loss: " << std::scientific << std::setprecision(3) << avgFinalLoss << std::endl;
        logFile << "Average Convergence Rate: " << std::fixed << std::setprecision(3) << avgConvergenceRate * 100 << "%" << std::endl;
        
        // Log individual thread results
        logFile << "Individual Thread Results:" << std::endl;
        for (int i = 0; i < activeThreads; ++i) {
            const auto& result = results[i];
            logFile << "  Thread " << result.threadId << " (Matrix " << result.matrixId << "): ";
            if (result.success) {
                logFile << "SUCCESS - Initial Loss: " << std::scientific << std::setprecision(3) << result.initialLoss;
                logFile << ", Final Loss: " << std::scientific << std::setprecision(3) << result.finalLoss;
                logFile << ", Convergence: " << std::fixed << std::setprecision(2) << result.convergenceRate * 100 << "%";
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
    SGDStressTester() : rng(std::chrono::steady_clock::now().time_since_epoch().count()) {
        // Detect hardware threads
        numThreads = std::thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 8; // Fallback
        
        // Use n-2 threads for computation (reserve 2 for system)
        activeThreads = (std::max)(1, numThreads - 2);
        
        omp_set_num_threads(activeThreads);
        omp_set_dynamic(0); // Disable dynamic thread adjustment
        
        std::cout << "Independent SGD Stress Tester initialized" << std::endl;
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
        logFile << "Infplane Independent SGD Stress Test Log" << std::endl;
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
        std::cout << "=== Infplane Independent SGD Stress Tester ===" << std::endl;
        std::cout << "High-Intensity CPU Stress Test using Independent SGD Optimizations" << std::endl;
        std::cout << "Each thread optimizes: minimize ||WX - Y||^2 using SGD" << std::endl;
        std::cout << "Testing matrix sizes: 500x500, 1000x1000, 5000x5000" << std::endl;
        std::cout << "SGD iterations per optimization: 500" << std::endl;
        std::cout << "Target CPU utilization: >90%" << std::endl;
        std::cout << std::endl;
        
        // Test 1: 500x500 matrices
        performStressTest(500, "Small Matrix Independent SGD Test");
        
        // Test 2: 1000x1000 matrices
        performStressTest(1000, "Medium Matrix Independent SGD Test");
        
        // Test 3: 5000x5000 matrices
        std::cout << "WARNING: 5000x5000 matrix test will consume significant CPU and memory resources!" << std::endl;
        std::cout << "This test will utilize " << activeThreads << " CPU cores at maximum capacity." << std::endl;
        std::cout << "Each thread will optimize a 5000x5000 matrix for 500 iterations." << std::endl;
        std::cout << "Press any key to continue or Ctrl+C to abort..." << std::endl;
        system("pause");
        
        try {
            performStressTest(5000, "Large Matrix Independent SGD Test");
        }
        catch (const std::exception& e) {
            std::cout << "Error during large matrix test: " << e.what() << std::endl;
            logFile << "Large Matrix Independent SGD Test: FAILED - " << e.what() << std::endl;
        }
        catch (...) {
            std::cout << "Unknown error during large matrix test" << std::endl;
            logFile << "Large Matrix Independent SGD Test: FAILED - Unknown error" << std::endl;
        }
        
        // Write summary
        logFile << "========================================" << std::endl;
        logFile << "Test Session Completed: " << getCurrentTimestamp() << std::endl;
        logFile << "========================================" << std::endl;
        
        std::cout << "All independent SGD tests completed! Check the log file for detailed results." << std::endl;
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
    
    std::cout << "Initializing independent SGD stress tester..." << std::endl;
    
    SGDStressTester tester;
    
    // Initialize log file
    std::string logFilename = "infplane_SGD_stress.log";
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