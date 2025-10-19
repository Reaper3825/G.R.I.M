#pragma once
#include <string>
#include <map>
#include <chrono>
#include "memory/memory_manager.hpp"
#include "memory/memory_router.hpp"

namespace GRIM {

struct StockData {
    std::string name;                       // e.g., "AAPL"
    std::chrono::system_clock::time_point time; // precise timestamp

    // Market data
    float price = 0.0f;
    float changePercent = 0.0f;
    int volume = 0;

    // Technical indicators
    int RSI = 0;
    float movingAvgShort = 0.0f;
    float movingAvgLong = 0.0f;
    float macd = 0.0f;
    float signal = 0.0f;

    // Fundamentals
    float earningsPerShare = 0.0f;
    float peRatio = 0.0f;
    float debtToEquity = 0.0f;
    float marketCap = 0.0f;

    // Sentiment/meta
    int sentimentScore = 0; // -100 to +100

    // Convert to a MemoryObject for GRIM memory system
    MemoryObject toMemoryObject() const;
};

class Stock {
public:
    explicit Stock(const std::string& symbol);

    void addEntry(int id, const StockData& data);
    const std::map<int, StockData>& getEntries() const;

private:
    std::string tickerSymbol;
    std::map<int, StockData> stockLog;
};

} // namespace GRIM
