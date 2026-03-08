#include "stock.hpp"
#include <ctime>
#include "Logger.hpp"

namespace GRIM {

// ============================================
// Convert StockData → MemoryObject
// ============================================
UnifiedMemoryObject StockData::toMemoryObject() const {
    UnifiedMemoryObject obj;
    obj.id = UnifiedMemoryObject::generateID();
    obj.timestamp = static_cast<uint64_t>(std::time(nullptr));
    obj.source = SourceType::SYSTEM_SW;
    obj.type = TypeTag::FACT;
    obj.intent = MemoryIntent::INFORM;
    obj.context = ContextType::MONITOR;
    obj.confidence = 1.0f;

    obj.raw = "Stock update for " + name;
    obj.normalized =
        "Price=" + std::to_string(price) +
        " RSI=" + std::to_string(RSI) +
        " Volume=" + std::to_string(volume) +
        " Change=" + std::to_string(changePercent) + "%" +
        " Sentiment=" + std::to_string(sentimentScore);

    obj.tags = {"stock", name, "market_data"};

    return obj;
}

// ============================================
// Stock Class
// ============================================
Stock::Stock(const std::string& symbol)
    : tickerSymbol(symbol) {}

// Add new entry and store in GRIM memory system
void Stock::addEntry(int id, const StockData& data) {
    stockLog[id] = data;

    // Convert to GRIM memory object
    UnifiedMemoryObject m = data.toMemoryObject();

    // Dispatch to memory router
    MemoryRouter router;
    router.dispatch(m);

    LOG_DEBUG("Stock", "Recorded stock data for " + data.name);
}

// Accessor
const std::map<int, StockData>& Stock::getEntries() const {
    return stockLog;
}

} // namespace GRIM
