#include "perception.hpp"
#include "logger.hpp"
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <algorithm>

// ✅ OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// ✅ Tesseract includes
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#ifdef _WIN32
#include <windows.h>
#include <gdiplus.h>
#pragma comment(lib, "gdiplus.lib")
#endif

namespace GRIM {
    namespace Perception {

        static bool initialized = false;

        // ✅ Tesseract OCR engine
        static tesseract::TessBaseAPI* g_tessEngine = nullptr;

        // ✅ YOLO model components (optional - loaded on demand)
        static cv::dnn::Net g_yoloNet;
        static std::vector<std::string> g_yoloClasses;
        static bool g_yoloLoaded = false;

#ifdef _WIN32
        static ULONG_PTR gdiplusToken = 0;

        // Helper to capture screen to bitmap
        static bool captureScreenRegion(int x, int y, int width, int height,
            std::vector<BYTE>& buffer, int& outWidth, int& outHeight) {
            // ...existing code...
            HDC hdcScreen = GetDC(nullptr);
            HDC hdcMem = CreateCompatibleDC(hdcScreen);

            if (x < 0 || y < 0 || width < 0 || height < 0) {
                x = 0;
                y = 0;
                width = GetSystemMetrics(SM_CXSCREEN);
                height = GetSystemMetrics(SM_CYSCREEN);
            }

            outWidth = width;
            outHeight = height;

            HBITMAP hBitmap = CreateCompatibleBitmap(hdcScreen, width, height);
            HBITMAP hOldBitmap = (HBITMAP)SelectObject(hdcMem, hBitmap);

            BitBlt(hdcMem, 0, 0, width, height, hdcScreen, x, y, SRCCOPY);

            BITMAPINFO bmi = {};
            bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
            bmi.bmiHeader.biWidth = width;
            bmi.bmiHeader.biHeight = -height;
            bmi.bmiHeader.biPlanes = 1;
            bmi.bmiHeader.biBitCount = 32;
            bmi.bmiHeader.biCompression = BI_RGB;

            buffer.resize(width * height * 4);
            GetDIBits(hdcMem, hBitmap, 0, height, buffer.data(), &bmi, DIB_RGB_COLORS);

            SelectObject(hdcMem, hOldBitmap);
            DeleteObject(hBitmap);
            DeleteDC(hdcMem);
            ReleaseDC(nullptr, hdcScreen);

            return !buffer.empty();
        }

        // ✅ NEW: Convert Windows BGRA buffer to OpenCV Mat
        static cv::Mat bufferToMat(const std::vector<BYTE>& buffer, int width, int height) {
            if (buffer.empty() || width <= 0 || height <= 0) {
                return cv::Mat();
            }

            // Windows bitmap is BGRA, convert to BGR for OpenCV
            cv::Mat mat(height, width, CV_8UC4, (void*)buffer.data());
            cv::Mat bgr;
            cv::cvtColor(mat, bgr, cv::COLOR_BGRA2BGR);
            return bgr;
        }

#endif

        // Simple brightness-based text detection
        static std::string simpleTextDetection(const std::vector<BYTE>& buffer, int width, int height) {
            // ...existing code...
            if (buffer.empty() || width <= 0 || height <= 0) {
                return "[Error] Invalid image data";
            }

            int brightPixels = 0;
            int darkPixels = 0;

            for (size_t i = 0; i < buffer.size(); i += 4) {
                BYTE b = buffer[i];
                BYTE g = buffer[i + 1];
                BYTE r = buffer[i + 2];

                int brightness = (r + g + b) / 3;
                if (brightness > 200) {
                    brightPixels++;
                }
                else if (brightness < 55) {
                    darkPixels++;
                }
            }

            int totalPixels = width * height;
            float brightRatio = (float)brightPixels / totalPixels;
            float darkRatio = (float)darkPixels / totalPixels;

            std::ostringstream result;
            result << "[Perception] Screen capture: " << width << "x" << height << "\n";
            result << "Bright pixels: " << (int)(brightRatio * 100) << "%\n";
            result << "Dark pixels: " << (int)(darkRatio * 100) << "%\n";

            if (brightRatio > 0.5) {
                result << "Analysis: Bright screen (likely text on light background)";
            }
            else if (darkRatio > 0.5) {
                result << "Analysis: Dark screen (likely dark theme or video)";
            }
            else {
                result << "Analysis: Mixed content (images/UI elements)";
            }

            return result.str();
        }

        // Detect potential text regions by contrast
        static std::string detectTextRegions(const std::vector<BYTE>& buffer, int width, int height) {
            // ...existing code...
            if (buffer.empty() || width <= 0 || height <= 0) {
                return "[Error] Invalid image data";
            }

            const int sampleSize = 20;
            int textLikeRegions = 0;

            for (int y = 0; y < height - 1; y += sampleSize) {
                for (int x = 0; x < width - 1; x += sampleSize) {
                    int idx = (y * width + x) * 4;
                    if (idx + width * 4 + 4 >= (int)buffer.size()) continue;

                    int curr = (buffer[idx] + buffer[idx + 1] + buffer[idx + 2]) / 3;
                    int right = (buffer[idx + 4] + buffer[idx + 5] + buffer[idx + 6]) / 3;
                    int down = (buffer[idx + width * 4] + buffer[idx + width * 4 + 1] + buffer[idx + width * 4 + 2]) / 3;

                    if (abs(curr - right) > 100 || abs(curr - down) > 100) {
                        textLikeRegions++;
                    }
                }
            }

            int totalSamples = (width / sampleSize) * (height / sampleSize);
            float textDensity = (float)textLikeRegions / totalSamples;

            std::ostringstream result;
            result << "[Perception] Text region detection:\n";
            result << "High-contrast regions: " << textLikeRegions << " / " << totalSamples << "\n";
            result << "Text density: " << (int)(textDensity * 100) << "%\n";

            if (textDensity > 0.3) {
                result << "Analysis: High text content detected (document/webpage/code)";
            }
            else if (textDensity > 0.1) {
                result << "Analysis: Moderate text content (UI with text)";
            }
            else {
                result << "Analysis: Low text content (image/video/graphics)";
            }

            return result.str();
        }

        // ✅ NEW: Initialize Tesseract OCR
        static bool initTesseract() {
            if (g_tessEngine != nullptr) {
                return true; // Already initialized
            }

            g_tessEngine = new tesseract::TessBaseAPI();

            // Try to initialize with English language data
              // Tesseract expects tessdata folder in: D:/G.R.I.M/resources/tessdata
            const char* tessdataPath = "D:/G.R.I.M/resources";

            if (g_tessEngine->Init(tessdataPath, "eng") != 0) {
                LOG_ERROR("Perception", "Failed to initialize Tesseract OCR - check tessdata folder");
                delete g_tessEngine;
                g_tessEngine = nullptr;
                return false;
            }

            // Set page segmentation mode (PSM_AUTO = automatic)
            g_tessEngine->SetPageSegMode(tesseract::PSM_AUTO);

            LOG_DEBUG("Perception", "Tesseract OCR initialized successfully");
            return true;
        }

        // ✅ NEW: Load YOLO model (optional - can be slow on first load)
        static bool loadYOLO() {
            if (g_yoloLoaded) {
                return true;
            }

            try {
                // YOLO model files should be in: D:/G.R.I.M/resources/models/yolo/
                std::string modelConfig = "D:/G.R.I.M/resources/models/yolo/yolov3.cfg";
                std::string modelWeights = "D:/G.R.I.M/resources/models/yolo/yolov3.weights";
                std::string classesFile = "D:/G.R.I.M/resources/models/yolo/coco.names";

                // Check if files exist
                std::ifstream configTest(modelConfig);
                std::ifstream weightsTest(modelWeights);
                std::ifstream classesTest(classesFile);

                if (!configTest.good() || !weightsTest.good() || !classesTest.good()) {
                    LOG_DEBUG("Perception", "YOLO model files not found - object detection will use basic color detection");
                    return false;
                }

                // Load class names
                g_yoloClasses.clear();
                std::string line;
                std::ifstream ifs(classesFile);
                while (std::getline(ifs, line)) {
                    g_yoloClasses.push_back(line);
                }

                // Load YOLO network
                g_yoloNet = cv::dnn::readNetFromDarknet(modelConfig, modelWeights);
                g_yoloNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                g_yoloNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

                g_yoloLoaded = true;
                LOG_DEBUG("Perception", "YOLO model loaded successfully (" +
                    std::to_string(g_yoloClasses.size()) + " classes)");
                return true;

            }
            catch (const cv::Exception& e) {
                LOG_ERROR("Perception", std::string("Failed to load YOLO model: ") + e.what());
                return false;
            }
        }

        void init() {
            if (initialized) {
                LOG_DEBUG("Perception", "Already initialized");
                return;
            }

            LOG_PHASE("Initializing Perception System", true);

#ifdef _WIN32
            // Initialize GDI+
            Gdiplus::GdiplusStartupInput gdiplusStartupInput;
            Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, nullptr);
            LOG_DEBUG("Perception", "GDI+ initialized for screen capture");
#else
            LOG_DEBUG("Perception", "Non-Windows platform - screen capture not implemented");
#endif

            // ✅ Initialize Tesseract OCR
            if (initTesseract()) {
                LOG_DEBUG("Perception", "OCR engine ready");
            }
            else {
                LOG_DEBUG("Perception", "OCR engine not available - text detection will be basic");
            }

            // ✅ Try to load YOLO (optional - don't fail if not available)        loadYOLO();  // ? Re-enabled with YOLOv8 ONNX

            initialized = true;
            LOG_PHASE("Perception system initialized", true);
        }

        void shutdown() {
            if (!initialized) {
                return;
            }

            LOG_PHASE("Shutting down Perception system", true);

            // ✅ Cleanup Tesseract
            if (g_tessEngine != nullptr) {
                g_tessEngine->End();
                delete g_tessEngine;
                g_tessEngine = nullptr;
            }

            // ✅ Cleanup YOLO
            if (g_yoloLoaded) {
                g_yoloNet = cv::dnn::Net(); // Release network
                g_yoloClasses.clear();
                g_yoloLoaded = false;
            }

#ifdef _WIN32
            if (gdiplusToken != 0) {
                Gdiplus::GdiplusShutdown(gdiplusToken);
                gdiplusToken = 0;
            }
#endif

            initialized = false;
            LOG_PHASE("Perception system shut down", true);
        }

        bool isAvailable() {
            return initialized;
        }

        std::string analyzeScreen() {
            if (!initialized) {
                return "[Error] Perception system not initialized";
            }

            LOG_DEBUG("Perception", "Analyzing screen...");

#ifdef _WIN32
            std::vector<BYTE> buffer;
            int width, height;

            if (!captureScreenRegion(-1, -1, -1, -1, buffer, width, height)) {
                return "[Error] Failed to capture screen";
            }

            return simpleTextDetection(buffer, width, height);
#else
            return "[Perception] Screen capture only supported on Windows";
#endif
        }

        // ✅ NEW: Perform OCR with Tesseract
        std::string performOCR(const std::vector<unsigned char>& imageData, int width, int height) {
            if (!g_tessEngine) {
                return "[Error] OCR engine not initialized";
            }

#ifdef _WIN32
            try {
                // Convert to OpenCV Mat
                cv::Mat image = bufferToMat(imageData, width, height);

                if (image.empty()) {
                    return "[Error] Failed to convert image data";
                }

                // Convert to grayscale for better OCR
                cv::Mat gray;
                cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

                // Apply simple thresholding to improve text contrast
                cv::Mat thresh;
                cv::threshold(gray, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

                // Set image data for Tesseract
                g_tessEngine->SetImage(thresh.data, thresh.cols, thresh.rows,
                    thresh.channels(), thresh.step);

                // Get OCR result
                char* outText = g_tessEngine->GetUTF8Text();
                std::string result(outText);
                delete[] outText;

                // Get confidence
                int confidence = g_tessEngine->MeanTextConf();

                std::ostringstream output;
                output << "[Perception] OCR Result (confidence: " << confidence << "%):\n";
                output << "---\n";
                output << result;
                output << "\n---";

                return output.str();

            }
            catch (const cv::Exception& e) {
                return std::string("[Error] OCR failed: ") + e.what();
            }
#else
            return "[Error] OCR only supported on Windows";
#endif
        }

        std::string readText(int x, int y, int width, int height) {
            if (!initialized) {
                return "[Error] Perception system not initialized";
            }

            LOG_DEBUG("Perception", "Reading text from screen region...");

#ifdef _WIN32
            std::vector<BYTE> buffer;
            int outWidth, outHeight;

            if (!captureScreenRegion(x, y, width, height, buffer, outWidth, outHeight)) {
                return "[Error] Failed to capture screen region";
            }

            // ✅ Use Tesseract OCR if available
            if (g_tessEngine) {
                return performOCR(buffer, outWidth, outHeight);
            }
            else {
                // Fallback to basic text detection
                return detectTextRegions(buffer, outWidth, outHeight);
            }
#else
            return "[Perception] OCR only supported on Windows";
#endif
        }

        // ✅ NEW: Advanced object detection with YOLO
        std::string detectObjectsAdvanced(const std::vector<unsigned char>& imageData, int width, int height) {
            if (!g_yoloLoaded) {
                return "[Error] YOLO model not loaded";
            }

#ifdef _WIN32
            try {
                cv::Mat image = bufferToMat(imageData, width, height);

                if (image.empty()) {
                    return "[Error] Failed to convert image data";
                }

                // Prepare blob from image
                cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(416, 416),
                    cv::Scalar(0, 0, 0), true, false);

                g_yoloNet.setInput(blob);

                // Get output layer names
                std::vector<std::string> outNames = g_yoloNet.getUnconnectedOutLayersNames();

                // Forward pass
                std::vector<cv::Mat> outs;
                g_yoloNet.forward(outs, outNames);

                // Process detections
                std::vector<int> classIds;
                std::vector<float> confidences;
                std::vector<cv::Rect> boxes;

                float confThreshold = 0.5f;
                float nmsThreshold = 0.4f;

                for (size_t i = 0; i < outs.size(); ++i) {
                    float* data = (float*)outs[i].data;
                    for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
                        cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                        cv::Point classIdPoint;
                        double confidence;
                        cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

                        if (confidence > confThreshold) {
                            int centerX = (int)(data[0] * image.cols);
                            int centerY = (int)(data[1] * image.rows);
                            int w = (int)(data[2] * image.cols);
                            int h = (int)(data[3] * image.rows);
                            int left = centerX - w / 2;
                            int top = centerY - h / 2;

                            classIds.push_back(classIdPoint.x);
                            confidences.push_back((float)confidence);
                            boxes.push_back(cv::Rect(left, top, w, h));
                        }
                    }
                }

                // Apply non-maximum suppression
                std::vector<int> indices;
                cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

                // Format output
                std::ostringstream output;
                output << "[Perception] YOLO Object Detection:\n";
                output << "Screen size: " << width << "x" << height << "\n";
                output << "Detected " << indices.size() << " objects:\n";

                for (size_t i = 0; i < indices.size(); ++i) {
                    int idx = indices[i];
                    cv::Rect box = boxes[idx];
                    std::string className = g_yoloClasses[classIds[idx]];
                    float conf = confidences[idx];

                    output << "  - " << className
                        << " (confidence: " << (int)(conf * 100) << "%) "
                        << "at [" << box.x << "," << box.y
                        << " " << box.width << "x" << box.height << "]\n";
                }

                return output.str();

            }
            catch (const cv::Exception& e) {
                return std::string("[Error] YOLO detection failed: ") + e.what();
            }
#else
            return "[Error] Object detection only supported on Windows";
#endif
        }

        std::string detectObjects() {
            if (!initialized) {
                return "[Error] Perception system not initialized";
            }

            LOG_DEBUG("Perception", "Detecting objects...");

#ifdef _WIN32
            std::vector<BYTE> buffer;
            int width, height;

            if (!captureScreenRegion(-1, -1, -1, -1, buffer, width, height)) {
                return "[Error] Failed to capture screen";
            }

            // ✅ Use YOLO if available
            if (g_yoloLoaded) {
                return detectObjectsAdvanced(buffer, width, height);
            }

            // ✅ Fallback to simple color detection
            int redRegions = 0, greenRegions = 0, blueRegions = 0;

            for (size_t i = 0; i < buffer.size(); i += 4) {
                BYTE b = buffer[i];
                BYTE g = buffer[i + 1];
                BYTE r = buffer[i + 2];

                if (r > g + 50 && r > b + 50) redRegions++;
                else if (g > r + 50 && g > b + 50) greenRegions++;
                else if (b > r + 50 && b > g + 50) blueRegions++;
            }

            int totalPixels = width * height;

            std::ostringstream result;
            result << "[Perception] Object detection (color-based):\n";
            result << "Screen size: " << width << "x" << height << "\n";

            if (redRegions > totalPixels / 20) {
                result << "- Red regions detected (" << (redRegions * 100 / totalPixels) << "%)\n";
            }
            if (greenRegions > totalPixels / 20) {
                result << "- Green regions detected (" << (greenRegions * 100 / totalPixels) << "%)\n";
            }
            if (blueRegions > totalPixels / 20) {
                result << "- Blue regions detected (" << (blueRegions * 100 / totalPixels) << "%)\n";
            }

            result << "Note: Install YOLO model for advanced object detection\n";
            result << "Place yolov3.cfg, yolov3.weights, coco.names in resources/models/yolo/";

            return result.str();
#else
            return "[Perception] Object detection only supported on Windows";
#endif
        }

        // ✅ NEW: Save screenshot to file
        bool saveScreenshot(const std::string& path, int x, int y, int width, int height) {
            if (!initialized) {
                LOG_ERROR("Perception", "Cannot save screenshot - system not initialized");
                return false;
            }

#ifdef _WIN32
            std::vector<BYTE> buffer;
            int outWidth, outHeight;

            if (!captureScreenRegion(x, y, width, height, buffer, outWidth, outHeight)) {
                LOG_ERROR("Perception", "Failed to capture screen for screenshot");
                return false;
            }

            try {
                cv::Mat image = bufferToMat(buffer, outWidth, outHeight);

                if (image.empty()) {
                    LOG_ERROR("Perception", "Failed to convert captured data to image");
                    return false;
                }

                // Save with OpenCV
                if (cv::imwrite(path, image)) {
                    LOG_DEBUG("Perception", "Screenshot saved: " + path);
                    return true;
                }
                else {
                    LOG_ERROR("Perception", "Failed to write screenshot file: " + path);
                    return false;
                }

            }
            catch (const cv::Exception& e) {
                LOG_ERROR("Perception", std::string("Screenshot save failed: ") + e.what());
                return false;
            }
#else
            LOG_ERROR("Perception", "Screenshot only supported on Windows");
            return false;
#endif
        }

    } // namespace Perception
} // namespace GRIM




