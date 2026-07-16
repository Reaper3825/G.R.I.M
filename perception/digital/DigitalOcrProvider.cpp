#include "DigitalOcrProvider.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <memory>
#include <string>

#include <opencv2/imgproc.hpp>

#ifdef GRIM_HAS_TESSERACT
#include <tesseract/baseapi.h>
#include <tesseract/publictypes.h>
#endif

namespace GRIM { namespace Perception { namespace Digital {

namespace {

constexpr std::size_t kMaxTextBytes = 64 * 1024;
constexpr std::size_t kMaxRegions = 512;

double ElapsedMs(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

std::string Trim(std::string value) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

#ifdef GRIM_HAS_TESSERACT
class TesseractDigitalOcrProvider final : public DigitalOcrProvider {
public:
    ~TesseractDigitalOcrProvider() override {
        if (initialized_) engine_.End();
    }

    const char* ProviderName() const noexcept override { return "tesseract"; }

    DigitalOcrResult Recognize(const cv::Mat& bgr_frame,
                               unsigned int dpi_x,
                               unsigned int dpi_y) override {
        const auto start = std::chrono::steady_clock::now();
        DigitalOcrResult result;
        result.provider = ProviderName();

        if (bgr_frame.empty() || bgr_frame.depth() != CV_8U ||
            (bgr_frame.channels() != 1 && bgr_frame.channels() != 3 &&
             bgr_frame.channels() != 4)) {
            result.status = DigitalPrimitiveStatus::Failed;
            result.error = "OCR requires a non-empty 8-bit, 1/3/4-channel frame";
            result.duration_ms = ElapsedMs(start);
            return result;
        }
        if (!EnsureInitialized(result.error)) {
            result.status = DigitalPrimitiveStatus::Unavailable;
            result.duration_ms = ElapsedMs(start);
            return result;
        }

        cv::Mat stable;
        if (bgr_frame.channels() == 3) {
            cv::cvtColor(bgr_frame, stable, cv::COLOR_BGR2GRAY);
        } else if (bgr_frame.channels() == 4) {
            cv::cvtColor(bgr_frame, stable, cv::COLOR_BGRA2GRAY);
        } else {
            stable = bgr_frame.isContinuous() ? bgr_frame : bgr_frame.clone();
        }
        engine_.SetSourceResolution(static_cast<int>(std::max(70u, dpi_y)));
        engine_.SetImage(stable.data, stable.cols, stable.rows,
                         stable.channels(), static_cast<int>(stable.step));
        if (engine_.Recognize(nullptr) != 0) {
            engine_.Clear();
            result.status = DigitalPrimitiveStatus::Failed;
            result.error = "Tesseract failed to recognize the frame";
            result.duration_ms = ElapsedMs(start);
            return result;
        }

        std::unique_ptr<char[]> full(engine_.GetUTF8Text());
        if (full) {
            result.full_text.assign(full.get());
            if (result.full_text.size() > kMaxTextBytes) {
                result.full_text.resize(kMaxTextBytes);
            }
            result.full_text = Trim(std::move(result.full_text));
        }
        result.mean_confidence =
            std::clamp(static_cast<float>(engine_.MeanTextConf()) / 100.0f, 0.0f, 1.0f);

        if (auto* iterator = engine_.GetIterator()) {
            do {
                if (result.regions.size() >= kMaxRegions) break;
                std::unique_ptr<char[]> line(iterator->GetUTF8Text(tesseract::RIL_TEXTLINE));
                if (!line) continue;
                std::string text = Trim(std::string(line.get()));
                if (text.empty()) continue;
                if (text.size() > 4096) text.resize(4096);

                int left = 0, top = 0, right = 0, bottom = 0;
                if (!iterator->BoundingBox(tesseract::RIL_TEXTLINE,
                                           &left, &top, &right, &bottom)) continue;
                DigitalTextRegion region;
                region.frame_rect = {left, top, right - left, bottom - top};
                region.text = std::move(text);
                region.confidence = std::clamp(
                    iterator->Confidence(tesseract::RIL_TEXTLINE) / 100.0f, 0.0f, 1.0f);
                result.regions.push_back(std::move(region));
            } while (iterator->Next(tesseract::RIL_TEXTLINE));
        }

        engine_.Clear();
        result.status = DigitalPrimitiveStatus::Ok;
        result.duration_ms = ElapsedMs(start);
        (void)dpi_x;
        return result;
    }

private:
    bool EnsureInitialized(std::string& error) {
        if (initialized_) return true;
        if (initialization_attempted_) {
            error = initialization_error_;
            return false;
        }
        initialization_attempted_ = true;

#ifdef GRIM_ROOT_DIR
        const auto data_path = (std::filesystem::path(GRIM_ROOT_DIR) / "resources").string();
#else
        const auto data_path = (std::filesystem::current_path() / "resources").string();
#endif
        if (engine_.Init(data_path.c_str(), "eng") != 0) {
            initialization_error_ = "Tesseract initialization failed; expected resources/tessdata/eng.traineddata";
            error = initialization_error_;
            return false;
        }
#ifdef _WIN32
        engine_.SetVariable("debug_file", "nul");
#else
        engine_.SetVariable("debug_file", "/dev/null");
#endif
        engine_.SetPageSegMode(tesseract::PSM_AUTO);
        initialized_ = true;
        return true;
    }

    tesseract::TessBaseAPI engine_;
    bool initialization_attempted_ = false;
    bool initialized_ = false;
    std::string initialization_error_;
};
#else
class TesseractDigitalOcrProvider final : public DigitalOcrProvider {
public:
    const char* ProviderName() const noexcept override { return "tesseract-unavailable"; }
    DigitalOcrResult Recognize(const cv::Mat&, unsigned int, unsigned int) override {
        DigitalOcrResult result;
        result.status = DigitalPrimitiveStatus::Unavailable;
        result.provider = ProviderName();
        result.error = "GRIM was built without Tesseract";
        return result;
    }
};
#endif

} // namespace

std::unique_ptr<DigitalOcrProvider> CreateDefaultDigitalOcrProvider() {
    return std::make_unique<TesseractDigitalOcrProvider>();
}

}}} // namespace GRIM::Perception::Digital
