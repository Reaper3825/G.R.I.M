//======================================================//
//  GRIM HTML Content Extractor
//  Proper HTML parsing using Gumbo parser
//  
//  Extracts meaningful text content from HTML pages
//  Filters out navigation, scripts, styles, etc.
//  
//  Version: 1.0.0
//======================================================//

#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <gumbo.h>

namespace GRIM {
namespace Training {

class HTMLExtractor {
public:
    struct ExtractedContent {
        std::string main_text;
        std::string title;
        size_t text_blocks_found = 0;
        bool has_article_content = false;
    };
    
    static ExtractedContent extract(const std::string& html) {
        ExtractedContent result;
        
        GumboOutput* output = gumbo_parse(html.c_str());
        if (!output) return result;
        
        // Extract title
        result.title = extractTitle(output->root);
        
        // Extract main content
        std::vector<std::string> text_blocks;
        extractTextBlocks(output->root, text_blocks);
        
        result.text_blocks_found = text_blocks.size();
        
        // Join text blocks with double newlines
        std::ostringstream oss;
        for (size_t i = 0; i < text_blocks.size(); ++i) {
            if (i > 0) oss << "\n\n";
            oss << text_blocks[i];
        }
        result.main_text = oss.str();
        
        // Check if we found substantial content - be more lenient
        // Accept if we have at least 1 block OR total text is > 100 chars
        result.has_article_content = (result.text_blocks_found > 0 && 
                                      result.main_text.length() > 100);
        
        gumbo_destroy_output(&kGumboDefaultOptions, output);
        return result;
    }

private:
    // Skip these tags - they don't contain main content
    static bool shouldSkipTag(GumboTag tag) {
        return tag == GUMBO_TAG_SCRIPT || 
               tag == GUMBO_TAG_STYLE ||
               tag == GUMBO_TAG_NOSCRIPT ||
               tag == GUMBO_TAG_IFRAME ||
               tag == GUMBO_TAG_OBJECT ||
               tag == GUMBO_TAG_EMBED;
    }
    
    // Skip these elements by class/id (navigation, ads, etc.)
    static bool shouldSkipElement(GumboNode* node) {
        if (node->type != GUMBO_NODE_ELEMENT) return false;
        
        GumboAttribute* class_attr = gumbo_get_attribute(&node->v.element.attributes, "class");
        GumboAttribute* id_attr = gumbo_get_attribute(&node->v.element.attributes, "id");
        
        std::vector<std::string> skip_patterns = {
            "nav", "menu", "sidebar", "footer", "header",
            "advertisement", "ad-", "social", "share", "comment",
            "cookie", "popup", "modal", "banner"
        };
        
        auto contains_pattern = [&](const std::string& str) {
            std::string lower = str;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            for (const auto& pattern : skip_patterns) {
                if (lower.find(pattern) != std::string::npos) return true;
            }
            return false;
        };
        
        if (class_attr && contains_pattern(class_attr->value)) return true;
        if (id_attr && contains_pattern(id_attr->value)) return true;
        
        return false;
    }
    
    // Check if element likely contains main content
    static bool isContentTag(GumboTag tag) {
        return tag == GUMBO_TAG_ARTICLE ||
               tag == GUMBO_TAG_MAIN ||
               tag == GUMBO_TAG_P ||
               tag == GUMBO_TAG_DIV ||
               tag == GUMBO_TAG_SECTION ||
               tag == GUMBO_TAG_H1 || tag == GUMBO_TAG_H2 || 
               tag == GUMBO_TAG_H3 || tag == GUMBO_TAG_H4 ||
               tag == GUMBO_TAG_H5 || tag == GUMBO_TAG_H6 ||
               tag == GUMBO_TAG_LI ||
               tag == GUMBO_TAG_BLOCKQUOTE ||
               tag == GUMBO_TAG_PRE;
    }
    
    static std::string extractTitle(GumboNode* node) {
        if (node->type != GUMBO_NODE_ELEMENT) return "";
        
        if (node->v.element.tag == GUMBO_TAG_TITLE) {
            return extractText(node);
        }
        
        GumboVector* children = &node->v.element.children;
        for (unsigned int i = 0; i < children->length; ++i) {
            std::string title = extractTitle(static_cast<GumboNode*>(children->data[i]));
            if (!title.empty()) return title;
        }
        
        return "";
    }
    
    static void extractTextBlocks(GumboNode* node, std::vector<std::string>& blocks) {
        if (!node) return;
        
        if (node->type == GUMBO_NODE_ELEMENT) {
            // Skip unwanted tags
            if (shouldSkipTag(node->v.element.tag)) return;
            
            // Skip navigation/ads/etc by class/id
            if (shouldSkipElement(node)) return;
            
            // If this is a content container, extract its text
            if (isContentTag(node->v.element.tag)) {
                std::string text = extractText(node);
                
                // Clean and validate text
                text = cleanText(text);
                
                // Be more lenient - accept blocks >= 30 chars with 40% alpha
                if (text.length() >= 30 && hasEnoughAlphaChars(text, 0.4f)) {
                    blocks.push_back(text);
                    return; // Don't recurse into children if we extracted from parent
                }
            }
            
            // Recurse into children
            GumboVector* children = &node->v.element.children;
            for (unsigned int i = 0; i < children->length; ++i) {
                extractTextBlocks(static_cast<GumboNode*>(children->data[i]), blocks);
            }
        }
    }
    
    static std::string extractText(GumboNode* node) {
        if (!node) return "";
        
        if (node->type == GUMBO_NODE_TEXT) {
            return node->v.text.text;
        }
        
        if (node->type == GUMBO_NODE_ELEMENT) {
            if (shouldSkipTag(node->v.element.tag)) return "";
            
            std::ostringstream oss;
            GumboVector* children = &node->v.element.children;
            
            for (unsigned int i = 0; i < children->length; ++i) {
                std::string child_text = extractText(static_cast<GumboNode*>(children->data[i]));
                if (!child_text.empty()) {
                    oss << child_text << " ";
                }
            }
            
            return oss.str();
        }
        
        return "";
    }
    
    static std::string cleanText(const std::string& text) {
        std::string result = text;
        
        // Replace multiple spaces with single space
        size_t pos = 0;
        while ((pos = result.find("  ", pos)) != std::string::npos) {
            result.replace(pos, 2, " ");
        }
        
        // Trim leading/trailing whitespace
        size_t start = result.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) return "";
        
        size_t end = result.find_last_not_of(" \t\n\r");
        result = result.substr(start, end - start + 1);
        
        return result;
    }
    
    static bool hasEnoughAlphaChars(const std::string& text, float min_ratio) {
        if (text.empty()) return false;
        
        size_t alpha_count = 0;
        for (char c : text) {
            if (std::isalpha(static_cast<unsigned char>(c))) {
                alpha_count++;
            }
        }
        
        float ratio = static_cast<float>(alpha_count) / text.length();
        return ratio >= min_ratio;
    }
};

} // namespace Training
} // namespace GRIM
