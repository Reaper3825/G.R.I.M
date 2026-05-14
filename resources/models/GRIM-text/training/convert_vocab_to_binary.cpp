#include "../Shared/UnigramByte/UniByte.hpp"
#include "../Shared/HyperParameters/HyperparameterGroupings.hpp"
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
    GRIM::HyperParameters::TokenizerHP tokenizer_hp;
    tokenizer_hp.target_vocab_size = static_cast<int>(tokens.size());
    tokenizer_hp.character_coverage = 0.9995f;
    tokenizer_hp.min_subword_freq = 3;
    tokenizer_hp.enable_parallel_subword_mining = true;
    tokenizer_hp.enable_scratch_block_reasoning = true;
    tokenizer_hp.detect_numbers = true;
    tokenizer_hp.enable_byte_fallback = true;
    tokenizer_hp.prefer_gpu = true;
    tokenizer_hp.vocab_score_multiplier = 1.0f;

    GRIM::Tokenizer::UniByte tokenizer(tokenizer_hp);
    
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
    GRIM::Tokenizer::UniByte verify_tokenizer(tokenizer_hp);
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
