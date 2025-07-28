#include <windows.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <regex>
#include <thread>
#include <winhttp.h>
#include <algorithm>

#pragma comment(lib, "winhttp.lib")

class LLMPerformanceTester {
private:
    std::ofstream logFile;
    std::string ollamaBaseUrl;
    std::string testPrompt;
    
    // Get current timestamp as string
    std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
    
    // Convert string to wide string
    std::wstring stringToWString(const std::string& str) {
        if (str.empty()) return std::wstring();
        int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
        std::wstring wstrTo(size_needed, 0);
        MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
        return wstrTo;
    }
    
    // Convert wide string to string
    std::string wstringToString(const std::wstring& wstr) {
        if (wstr.empty()) return std::string();
        int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
        std::string strTo(size_needed, 0);
        WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
        return strTo;
    }
    
    // Escape JSON string
    std::string escapeJsonString(const std::string& str) {
        std::string escaped;
        for (char c : str) {
            switch (c) {
                case '"': escaped += "\\\""; break;
                case '\\': escaped += "\\\\"; break;
                case '\b': escaped += "\\b"; break;
                case '\f': escaped += "\\f"; break;
                case '\n': escaped += "\\n"; break;
                case '\r': escaped += "\\r"; break;
                case '\t': escaped += "\\t"; break;
                default: escaped += c; break;
            }
        }
        return escaped;
    }
    
    // Make HTTP request to Ollama API with improved error handling
    std::string makeHttpRequest(const std::string& method, const std::string& endpoint, const std::string& data = "") {
        HINTERNET hSession = NULL;
        HINTERNET hConnect = NULL;
        HINTERNET hRequest = NULL;
        std::string response;
        
        try {
            std::cout << "    Making " << method << " request to " << endpoint << std::endl;
            
            // Initialize WinHTTP with detailed error checking
            hSession = WinHttpOpen(L"LLM Performance Tester/1.0",
                                 WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                 WINHTTP_NO_PROXY_NAME,
                                 WINHTTP_NO_PROXY_BYPASS, 0);
            
            if (!hSession) {
                DWORD dwError = GetLastError();
                throw std::runtime_error("Failed to initialize WinHTTP session, error: " + std::to_string(dwError));
            }
            
            // Set extended timeouts for LLM inference (up to 10 minutes)
            DWORD dwConnectTimeout = 30000;      // 30 seconds to connect
            DWORD dwSendTimeout = 60000;         // 60 seconds to send
            DWORD dwReceiveTimeout = 600000;     // 10 minutes to receive (for slow models)
            DWORD dwResolveTimeout = 30000;      // 30 seconds to resolve
            
            if (!WinHttpSetTimeouts(hSession, dwResolveTimeout, dwConnectTimeout, dwSendTimeout, dwReceiveTimeout)) {
                DWORD dwError = GetLastError();
                std::cout << "    Warning: Failed to set timeouts, error: " << dwError << std::endl;
            }
            
            // Connect to localhost:11434 (default Ollama port)
            hConnect = WinHttpConnect(hSession, L"localhost", 11434, 0);
            if (!hConnect) {
                DWORD dwError = GetLastError();
                throw std::runtime_error("Failed to connect to Ollama server on localhost:11434, error: " + std::to_string(dwError));
            }
            
            // Create request
            std::wstring wEndpoint = stringToWString(endpoint);
            std::wstring wMethod = stringToWString(method);
            
            hRequest = WinHttpOpenRequest(hConnect, wMethod.c_str(), wEndpoint.c_str(),
                                        NULL, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, 0);
            
            if (!hRequest) {
                DWORD dwError = GetLastError();
                throw std::runtime_error("Failed to create HTTP request, error: " + std::to_string(dwError));
            }
            
            // Set headers for JSON content
            if (!data.empty()) {
                std::wstring headers = L"Content-Type: application/json\r\nAccept: application/json\r\n";
                if (!WinHttpAddRequestHeaders(hRequest, headers.c_str(), -1, WINHTTP_ADDREQ_FLAG_ADD)) {
                    DWORD dwError = GetLastError();
                    std::cout << "    Warning: Failed to add request headers, error: " << dwError << std::endl;
                }
            }
            
            // Send request with detailed logging
            std::cout << "    Sending request with " << data.length() << " bytes of data" << std::endl;
            if (!data.empty()) {
                std::cout << "    Request data: " << data.substr(0, 100) << (data.length() > 100 ? "..." : "") << std::endl;
            }
            
            BOOL bResults = WinHttpSendRequest(hRequest,
                                             WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                                             data.empty() ? WINHTTP_NO_REQUEST_DATA : (LPVOID)data.c_str(),
                                             data.length(), data.length(), 0);
            
            if (!bResults) {
                DWORD dwError = GetLastError();
                std::string errorMsg = "Failed to send HTTP request, error code: " + std::to_string(dwError);
                if (dwError == ERROR_WINHTTP_TIMEOUT) {
                    errorMsg += " (Request timeout)";
                } else if (dwError == ERROR_WINHTTP_CANNOT_CONNECT) {
                    errorMsg += " (Cannot connect to server)";
                } else if (dwError == ERROR_WINHTTP_CONNECTION_ERROR) {
                    errorMsg += " (Connection error)";
                }
                throw std::runtime_error(errorMsg);
            }
            
            std::cout << "    Request sent successfully, waiting for response..." << std::endl;
            
            // Receive response with timeout handling
            bResults = WinHttpReceiveResponse(hRequest, NULL);
            if (!bResults) {
                DWORD dwError = GetLastError();
                std::string errorMsg = "Failed to receive HTTP response, error code: " + std::to_string(dwError);
                if (dwError == ERROR_WINHTTP_TIMEOUT) {
                    errorMsg += " (Response timeout - model may be taking too long)";
                } else if (dwError == ERROR_WINHTTP_CONNECTION_ERROR) {
                    errorMsg += " (Connection lost during response)";
                }
                throw std::runtime_error(errorMsg);
            }
            
            // Check status code
            DWORD dwStatusCode = 0;
            DWORD dwSize = sizeof(dwStatusCode);
            WinHttpQueryHeaders(hRequest, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                               WINHTTP_HEADER_NAME_BY_INDEX, &dwStatusCode, &dwSize, WINHTTP_NO_HEADER_INDEX);
            
            std::cout << "    HTTP Status Code: " << dwStatusCode << std::endl;
            
            if (dwStatusCode != 200) {
                // Try to read error response
                std::string errorResponse;
                DWORD dwErrorSize = 0;
                do {
                    dwErrorSize = 0;
                    WinHttpQueryDataAvailable(hRequest, &dwErrorSize);
                    if (dwErrorSize > 0) {
                        std::vector<char> errorBuffer(dwErrorSize + 1);
                        DWORD dwDownloaded = 0;
                        if (WinHttpReadData(hRequest, &errorBuffer[0], dwErrorSize, &dwDownloaded)) {
                            errorResponse.append(&errorBuffer[0], dwDownloaded);
                        }
                    }
                } while (dwErrorSize > 0);
                
                throw std::runtime_error("HTTP request failed with status code: " + std::to_string(dwStatusCode) + 
                                        ", error response: " + errorResponse);
            }
            
            // Read response data with progress indication
            DWORD dwTotalSize = 0;
            DWORD dwDownloaded = 0;
            int progressDots = 0;
            
            do {
                dwSize = 0;
                if (!WinHttpQueryDataAvailable(hRequest, &dwSize)) {
                    DWORD dwError = GetLastError();
                    std::cout << "    Warning: Failed to query available data, error: " << dwError << std::endl;
                    break;
                }
                
                if (dwSize == 0) {
                    break;
                }
                
                std::vector<char> buffer(dwSize + 1);
                ZeroMemory(&buffer[0], dwSize + 1);
                
                if (!WinHttpReadData(hRequest, (LPVOID)&buffer[0], dwSize, &dwDownloaded)) {
                    DWORD dwError = GetLastError();
                    std::cout << "    Warning: Failed to read data, error: " << dwError << std::endl;
                    break;
                }
                
                if (dwDownloaded > 0) {
                    response.append(&buffer[0], dwDownloaded);
                    dwTotalSize += dwDownloaded;
                    
                    // Show progress for long responses
                    if (dwTotalSize > progressDots * 1024) {
                        std::cout << ".";
                        progressDots++;
                        if (progressDots % 50 == 0) {
                            std::cout << " " << dwTotalSize << " bytes" << std::endl << "    ";
                        }
                    }
                }
                
            } while (dwSize > 0);
            
            if (progressDots > 0) {
                std::cout << std::endl;
            }
            
            std::cout << "    Response received: " << dwTotalSize << " bytes total" << std::endl;
            
            if (dwTotalSize == 0) {
                throw std::runtime_error("Received empty response from server");
            }
            
        }
        catch (const std::exception& e) {
            std::cerr << "    HTTP request error: " << e.what() << std::endl;
            logFile << "HTTP Error: " << e.what() << std::endl;
        }
        
        // Cleanup
        if (hRequest) WinHttpCloseHandle(hRequest);
        if (hConnect) WinHttpCloseHandle(hConnect);
        if (hSession) WinHttpCloseHandle(hSession);
        
        return response;
    }
    
    // Parse JSON response (improved implementation with simple string parsing)
    std::vector<std::string> parseModelList(const std::string& jsonResponse) {
        std::vector<std::string> models;
        
        if (jsonResponse.empty()) {
            std::cout << "    Warning: Empty JSON response" << std::endl;
            return models;
        }
        
        std::cout << "    Parsing model list from JSON response" << std::endl;
        
        // Simple string-based JSON parsing for model names (avoiding regex memory issues)
        size_t pos = 0;
        while ((pos = jsonResponse.find("\"name\":", pos)) != std::string::npos) {
            pos += 7; // Skip "name":
            
            // Skip whitespace
            while (pos < jsonResponse.length() && (jsonResponse[pos] == ' ' || jsonResponse[pos] == '\t')) {
                pos++;
            }
            
            // Check for opening quote
            if (pos < jsonResponse.length() && jsonResponse[pos] == '"') {
                pos++; // Skip opening quote
                size_t endPos = jsonResponse.find('"', pos);
                if (endPos != std::string::npos) {
                    std::string modelName = jsonResponse.substr(pos, endPos - pos);
                    models.push_back(modelName);
                    std::cout << "    Found model: " << modelName << std::endl;
                    pos = endPos + 1;
                } else {
                    break;
                }
            } else {
                pos++;
            }
        }
        
        return models;
    }
    
    // Extract JSON string value using simple string parsing
    std::string extractJsonString(const std::string& json, const std::string& key) {
        std::string searchKey = "\"" + key + "\":";
        size_t pos = json.find(searchKey);
        if (pos == std::string::npos) {
            return "";
        }
        
        pos += searchKey.length();
        
        // Skip whitespace
        while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) {
            pos++;
        }
        
        // Check for opening quote
        if (pos >= json.length() || json[pos] != '"') {
            return "";
        }
        
        pos++; // Skip opening quote
        
        // Find closing quote, handling escaped quotes
        std::string result;
        while (pos < json.length()) {
            if (json[pos] == '\\' && pos + 1 < json.length()) {
                // Handle escape sequences
                char nextChar = json[pos + 1];
                switch (nextChar) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case 'n': result += '\n'; break;
                    case 'r': result += '\r'; break;
                    case 't': result += '\t'; break;
                    case 'b': result += '\b'; break;
                    case 'f': result += '\f'; break;
                    default: result += nextChar; break;
                }
                pos += 2;
            } else if (json[pos] == '"') {
                // Found closing quote
                break;
            } else {
                result += json[pos];
                pos++;
            }
        }
        
        return result;
    }
    
    // Count tokens in response (improved word-based estimation)
    int countTokens(const std::string& text) {
        if (text.empty()) return 0;
        
        int tokenCount = 0;
        bool inWord = false;
        
        for (char c : text) {
            if (std::isalnum(c) || c == '_') {
                if (!inWord) {
                    tokenCount++;
                    inWord = true;
                }
            } else {
                inWord = false;
                // Count punctuation as tokens
                if (std::ispunct(c)) {
                    tokenCount++;
                }
            }
        }
        
        // Ensure at least 1 token
        return (tokenCount > 0) ? tokenCount : 1;
    }
    
    // Test model performance with improved error handling
    struct ModelPerformance {
        std::string modelName;
        bool available;
        double responseTime;
        int totalTokens;
        double tokensPerSecond;
        std::string errorMessage;
        std::string response;
    };
    
    ModelPerformance testModelPerformance(const std::string& modelName) {
        ModelPerformance result;
        result.modelName = modelName;
        result.available = false;
        result.responseTime = 0.0;
        result.totalTokens = 0;
        result.tokensPerSecond = 0.0;
        
        std::cout << "Testing model: " << modelName << std::endl;
        
        try {
            // Create properly formatted JSON request
            std::string escapedPrompt = escapeJsonString(testPrompt);
            std::string jsonRequest = "{"
                "\"model\":\"" + modelName + "\","
                "\"prompt\":\"" + escapedPrompt + "\","
                "\"stream\":false,"
                "\"options\":{\"temperature\":0.7}"
                "}";
            
            std::cout << "    Request size: " << jsonRequest.length() << " bytes" << std::endl;
            
            // Record start time
            auto startTime = std::chrono::high_resolution_clock::now();
            
            // Make request to Ollama API
            std::string response = makeHttpRequest("POST", "/api/generate", jsonRequest);
            
            // Record end time
            auto endTime = std::chrono::high_resolution_clock::now();
            
            if (response.empty()) {
                result.errorMessage = "Empty response from API";
                return result;
            }
            
            // Calculate response time
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            result.responseTime = duration.count() / 1000.0;
            
            std::cout << "    Raw response length: " << response.length() << " characters" << std::endl;
            std::cout << "    Response time: " << std::fixed << std::setprecision(2) << result.responseTime << " seconds" << std::endl;
            
            // Parse response to extract generated text using simple string parsing
            std::string generatedText = extractJsonString(response, "response");
            
            if (!generatedText.empty()) {
                result.response = generatedText;
                
                std::cout << "    Generated text length: " << result.response.length() << " characters" << std::endl;
                std::cout << "    Generated text preview: " << result.response.substr(0, 100) << "..." << std::endl;
                
                // Count tokens
                result.totalTokens = countTokens(result.response);
                
                // Calculate tokens per second
                if (result.responseTime > 0) {
                    result.tokensPerSecond = result.totalTokens / result.responseTime;
                }
                
                result.available = true;
                
                std::cout << "    Tokens generated: " << result.totalTokens << std::endl;
                std::cout << "    Speed: " << std::fixed << std::setprecision(2) << result.tokensPerSecond << " tokens/second" << std::endl;
            } else {
                result.errorMessage = "Failed to parse response - no 'response' field found";
                std::cout << "    Error: Could not find response field in JSON" << std::endl;
                std::cout << "    Response preview: " << response.substr(0, 200) << "..." << std::endl;
                
                // Check for error field in response
                std::string errorMsg = extractJsonString(response, "error");
                if (!errorMsg.empty()) {
                    result.errorMessage = "Server error: " + errorMsg;
                }
            }
            
        }
        catch (const std::exception& e) {
            result.errorMessage = e.what();
            std::cout << "    Exception: " << e.what() << std::endl;
        }
        
        return result;
    }
    
    // Check if Ollama is running with improved detection
    bool isOllamaRunning() {
        std::cout << "Checking if Ollama server is accessible..." << std::endl;
        std::string response = makeHttpRequest("GET", "/api/tags");
        bool running = !response.empty() && response.find("models") != std::string::npos;
        std::cout << "Ollama server " << (running ? "is accessible" : "is not accessible") << std::endl;
        return running;
    }

public:
    LLMPerformanceTester() {
        ollamaBaseUrl = "http://localhost:11434";
        testPrompt = "Please introduce yourself.";
        
        std::cout << "LLM Performance Tester initialized" << std::endl;
        std::cout << "Ollama Base URL: " << ollamaBaseUrl << std::endl;
        std::cout << "Test Prompt: " << testPrompt << std::endl;
    }
    
    // Initialize log file
    bool initializeLog(const std::string& filename) {
        logFile.open(filename, std::ios::out | std::ios::app);
        if (!logFile.is_open()) {
            std::cerr << "Error: Cannot create log file " << filename << std::endl;
            return false;
        }
        
        logFile << "========================================" << std::endl;
        logFile << "Infplane LLM Performance Test Log" << std::endl;
        logFile << "Test Session Started: " << getCurrentTimestamp() << std::endl;
        logFile << "Ollama Base URL: " << ollamaBaseUrl << std::endl;
        logFile << "Test Prompt: " << testPrompt << std::endl;
        logFile << "========================================" << std::endl;
        logFile << std::endl;
        
        return true;
    }
    
    // Run performance tests
    void runPerformanceTests() {
        std::cout << "=== Infplane LLM Performance Tester ===" << std::endl;
        std::cout << "Testing local LLM inference speed using Ollama" << std::endl;
        std::cout << "Test prompt: \"" << testPrompt << "\"" << std::endl;
        std::cout << std::endl;
        
        // Check if Ollama is running
        std::cout << "Checking Ollama server status..." << std::endl;
        if (!isOllamaRunning()) {
            std::cout << "ERROR: Ollama server is not running!" << std::endl;
            std::cout << "Please start Ollama server first: ollama serve" << std::endl;
            logFile << "ERROR: Ollama server is not running" << std::endl;
            return;
        }
        std::cout << "Ollama server is running" << std::endl;
        std::cout << std::endl;
        
        // Get list of available models
        std::cout << "Scanning for available models..." << std::endl;
        std::string modelsResponse = makeHttpRequest("GET", "/api/tags");
        
        if (modelsResponse.empty()) {
            std::cout << "ERROR: Failed to get model list from Ollama" << std::endl;
            logFile << "ERROR: Failed to get model list from Ollama" << std::endl;
            return;
        }
        
        std::vector<std::string> models = parseModelList(modelsResponse);
        
        if (models.empty()) {
            std::cout << "No models found in Ollama" << std::endl;
            std::cout << "Please download models first: ollama pull <model_name>" << std::endl;
            logFile << "No models found in Ollama" << std::endl;
            return;
        }
        
        std::cout << "Found " << models.size() << " models:" << std::endl;
        for (const auto& model : models) {
            std::cout << "  - " << model << std::endl;
        }
        std::cout << std::endl;
        
        // Test each model
        std::vector<ModelPerformance> results;
        
        for (const auto& model : models) {
            ModelPerformance result = testModelPerformance(model);
            results.push_back(result);
            
            // Display immediate results
            if (result.available) {
                std::cout << "    Response time: " << std::fixed << std::setprecision(2) << result.responseTime << " seconds" << std::endl;
                std::cout << "    Total tokens: " << result.totalTokens << std::endl;
                std::cout << "    Speed: " << std::fixed << std::setprecision(2) << result.tokensPerSecond << " tokens/second" << std::endl;
                std::cout << "    SUCCESS" << std::endl;
            } else {
                std::cout << "    ERROR: " << result.errorMessage << std::endl;
            }
            std::cout << std::endl;
            
            // Small delay between tests
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        
        // Generate summary report
        generateSummaryReport(results);
        
        std::cout << "Performance testing completed! Check the log file for detailed results." << std::endl;
    }
    
    // Generate summary report
    void generateSummaryReport(const std::vector<ModelPerformance>& results) {
        std::cout << "=== Performance Summary ===" << std::endl;
        
        // Log to file
        logFile << "=== Performance Test Results ===" << std::endl;
        logFile << "Test completed at: " << getCurrentTimestamp() << std::endl;
        logFile << std::endl;
        
        // Sort results by tokens per second (descending)
        std::vector<ModelPerformance> sortedResults = results;
        std::sort(sortedResults.begin(), sortedResults.end(), 
                 [](const ModelPerformance& a, const ModelPerformance& b) {
                     return a.tokensPerSecond > b.tokensPerSecond;
                 });
        
        int successCount = 0;
        double totalTokensPerSecond = 0.0;
        
        // Display results table
        std::cout << std::left << std::setw(25) << "Model Name" 
                  << std::setw(15) << "Status"
                  << std::setw(15) << "Time (s)"
                  << std::setw(12) << "Tokens"
                  << std::setw(15) << "Speed (t/s)" << std::endl;
        std::cout << std::string(82, '-') << std::endl;
        
        logFile << std::left << std::setw(25) << "Model Name" 
                << std::setw(15) << "Status"
                << std::setw(15) << "Time (s)"
                << std::setw(12) << "Tokens"
                << std::setw(15) << "Speed (t/s)" << std::endl;
        logFile << std::string(82, '-') << std::endl;
        
        for (const auto& result : sortedResults) {
            std::string status = result.available ? "SUCCESS" : "FAILED";
            
            std::cout << std::left << std::setw(25) << result.modelName
                      << std::setw(15) << status
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.responseTime
                      << std::setw(12) << result.totalTokens
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.tokensPerSecond << std::endl;
            
            logFile << std::left << std::setw(25) << result.modelName
                    << std::setw(15) << status
                    << std::setw(15) << std::fixed << std::setprecision(2) << result.responseTime
                    << std::setw(12) << result.totalTokens
                    << std::setw(15) << std::fixed << std::setprecision(2) << result.tokensPerSecond << std::endl;
            
            if (result.available) {
                successCount++;
                totalTokensPerSecond += result.tokensPerSecond;
            }
        }
        
        std::cout << std::string(82, '-') << std::endl;
        logFile << std::string(82, '-') << std::endl;
        
        // Calculate statistics
        double averageSpeed = successCount > 0 ? totalTokensPerSecond / successCount : 0.0;
        
        std::cout << "Total models tested: " << results.size() << std::endl;
        std::cout << "Successful tests: " << successCount << std::endl;
        std::cout << "Failed tests: " << (results.size() - successCount) << std::endl;
        std::cout << "Average speed: " << std::fixed << std::setprecision(2) << averageSpeed << " tokens/second" << std::endl;
        
        logFile << std::endl;
        logFile << "=== Summary Statistics ===" << std::endl;
        logFile << "Total models tested: " << results.size() << std::endl;
        logFile << "Successful tests: " << successCount << std::endl;
        logFile << "Failed tests: " << (results.size() - successCount) << std::endl;
        logFile << "Average speed: " << std::fixed << std::setprecision(2) << averageSpeed << " tokens/second" << std::endl;
        
        if (successCount > 0) {
            const auto& fastest = sortedResults[0];
            std::cout << "Fastest model: " << fastest.modelName << " (" << std::fixed << std::setprecision(2) << fastest.tokensPerSecond << " t/s)" << std::endl;
            logFile << "Fastest model: " << fastest.modelName << " (" << std::fixed << std::setprecision(2) << fastest.tokensPerSecond << " t/s)" << std::endl;
        }
        
        // Log detailed responses
        logFile << std::endl;
        logFile << "=== Detailed Responses ===" << std::endl;
        for (const auto& result : results) {
            logFile << "Model: " << result.modelName << std::endl;
            if (result.available) {
                logFile << "Response: " << result.response.substr(0, 200) << "..." << std::endl;
            } else {
                logFile << "Error: " << result.errorMessage << std::endl;
            }
            logFile << std::endl;
        }
        
        logFile << "========================================" << std::endl;
        logFile << "Test Session Completed: " << getCurrentTimestamp() << std::endl;
        logFile << "========================================" << std::endl;
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
    
    std::cout << "Initializing LLM performance tester..." << std::endl;
    
    LLMPerformanceTester tester;
    
    // Initialize log file
    std::string logFilename = "infplane_LLM_performance.log";
    if (!tester.initializeLog(logFilename)) {
        std::cerr << "Failed to initialize log file. Exiting." << std::endl;
        return 1;
    }
    
    std::cout << "Log file: " << logFilename << std::endl;
    std::cout << std::endl;
    
    try {
        // Run performance tests
        tester.runPerformanceTests();
    }
    catch (const std::exception& e) {
        std::cerr << "Error during performance test: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown error during performance test" << std::endl;
        return 1;
    }
    
    // Close log file
    tester.closeLog();
    
    std::cout << "Press any key to exit..." << std::endl;
    system("pause");
    
    return 0;
} 