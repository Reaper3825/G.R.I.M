#include "../Shared/UnigramByte/UniByte.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_text_vocab> <output_binary_vocab>" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

    // Read text vocabulary
    std::ifstream infile(input_path);
    if (!infile) {
        std::cerr << "ERROR: Cannot open input file: " << input_path << std::endl;
        return 1;
    }

    std::vector<std::string> tokens;
    std::string line;
    while (std::getline(infile, line)) {
        // Trim whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        size_t end = line.find_last_not_of(" \t\r\n");
        if (start != std::string::npos && end != std::string::npos) {
            tokens.push_back(line.substr(start, end - start + 1));
        } else if (!line.empty()) {
            tokens.push_back(line);
        }
    }
    infile.close();

    if (tokens.empty()) {
        std::cerr << "ERROR: No tokens read from input file" << std::endl;
        return 1;
    }

    std::cout << "Read " << tokens.size() << " tokens from " << input_path << std::endl;

    // Create tokenizer and train from corpus (adds tokens to vocab)
    GRIM::Tokenizer::UniByteConfig config;
    config.target_vocab_size = tokens.size();
    config.enable_byte_fallback = true;

    GRIM::Tokenizer::UniByte tokenizer(config);
    
    // Train from the token list as a "corpus" (one token per line)
    // This will populate the internal vocabulary
    tokenizer.trainFromCorpus(tokens, 0); // 0 merges = just add to vocab

    // Save in binary format
    if (!tokenizer.save(output_path)) {
        std::cerr << "ERROR: Failed to save binary vocabulary to " << output_path << std::endl;
        return 1;
    }

    std::cout << "Successfully saved binary vocabulary to " << output_path << std::endl;

    // Verify by loading it back
    GRIM::Tokenizer::UniByte verify_tokenizer;
    if (!verify_tokenizer.load(output_path)) {
        std::cerr << "WARNING: Failed to load back the binary vocabulary for verification" << std::endl;
        return 1;
    }

    if (verify_tokenizer.vocabSize() != tokens.size()) {
        std::cerr << "ERROR: Verification failed - vocab size mismatch" << std::endl;
        return 1;
    }

    std::cout << "Verification successful - vocab file is valid" << std::endl;
    return 0;
}
