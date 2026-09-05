// Standalone CPU tests for the actual GRMT source-ID codec and version gate.
#include "../resources/models/GRIM-text/Shared/GRMT/GrmtSourceIdentity.hpp"
#include "../resources/models/GRIM-text/Shared/GRMT/GrmtFormat.hpp"
#include <cassert>
#include <iostream>
#include <sstream>
#include <vector>

using namespace GRIM::GRMT;
template<class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
}

int main() {
    const std::vector<std::string> ids{"cb_first", "cb_same_tokens_different_source", u8"cb_\u03bb", std::string(kMaxConceptBlockIdBytes, 'x')};
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    // A stand-in token payload demonstrates exact boundaries: metadata is
    // outside the payload and cannot consume/change the subsequent token bytes.
    const uint32_t token_payload = 12345;
    for (const auto& id : ids) {
        writeConceptBlockId(stream, id, "test");
        stream.write(reinterpret_cast<const char*>(&token_payload), sizeof(token_payload));
    }
    for (const auto& id : ids) {
        assert(readConceptBlockId(stream, "test") == id);
        uint32_t token = 0;
        stream.read(reinterpret_cast<char*>(&token), sizeof(token));
        assert(token == token_payload);
    }
    assert(stream.peek() == std::char_traits<char>::eof());
    rejects([&] { writeConceptBlockId(stream, "", "empty"); });
    rejects([&] { writeConceptBlockId(stream, std::string(kMaxConceptBlockIdBytes + 1, 'x'), "oversized"); });
    for (uint32_t length : {0u, kMaxConceptBlockIdBytes + 1, 0xffffffffu}) {
        std::stringstream invalid;
        invalid.write(reinterpret_cast<const char*>(&length), sizeof(length));
        rejects([&] { readConceptBlockId(invalid, "bad length"); });
    }
    for (int bytes = 0; bytes < 4; ++bytes) {
        std::stringstream truncated(std::string(bytes, '\1'));
        rejects([&] { readConceptBlockId(truncated, "truncated prefix"); });
    }
    std::stringstream truncated;
    uint32_t length = 8;
    truncated.write(reinterpret_cast<const char*>(&length), sizeof(length));
    truncated << "short";
    rejects([&] { readConceptBlockId(truncated, "truncated ID"); });
    std::ostringstream broken;
    broken.setstate(std::ios::badbit);
    rejects([&] { writeConceptBlockId(broken, "cb_valid", "failed output"); });
    Header header{kMagic, 24, 1, 100};
    assert(!headerValidationError(header, "legacy").empty());
    header.version = GRIM::GRMT_FORMAT_VERSION;
    assert(headerValidationError(header, "current").empty());
    std::cout << "GRMT source identity codec and legacy-version rejection tests passed\n";
}
