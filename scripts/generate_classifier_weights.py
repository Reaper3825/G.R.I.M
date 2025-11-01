#!/usr/bin/env python3
"""
Generate default classifier weights in FlatBuffers format.
This script creates baseline weights that can be modified without recompiling.
"""

import sys
import os

# Add FlatBuffers Python module path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vcpkg_installed', 'x64-windows', 'tools', 'flatbuffers', 'python'))

try:
    import flatbuffers
    # Import generated schema
    ai_path = os.path.join(os.path.dirname(__file__), '..', 'ai')
    sys.path.insert(0, ai_path)
    from ClassifierWeights import WeightEntry, CategoryWeights, ClassifierConfig
except ImportError as e:
    print(f"ERROR: Import failed: {e}")
    print("Run: pip install flatbuffers")
    print("Or run flatc with --python to generate Python bindings first")
    sys.exit(1)

def create_weight_entry(builder, token, weight):
    """Create a WeightEntry FlatBuffer object"""
    token_offset = builder.CreateString(token)
    WeightEntry.Start(builder)
    WeightEntry.AddToken(builder, token_offset)
    WeightEntry.AddWeight(builder, weight)
    return WeightEntry.End(builder)

def create_category_weights(builder, category, weight_dict, default_weight=1.0):
    """Create a CategoryWeights FlatBuffer object"""
    # Create weight entries
    weight_entries = []
    for token, weight in sorted(weight_dict.items()):
        weight_entries.append(create_weight_entry(builder, token, weight))
    
    # Create weights vector
    CategoryWeights.StartWeightsVector(builder, len(weight_entries))
    for entry in reversed(weight_entries):  # Reverse for FlatBuffers
        builder.PrependUOffsetTRelative(entry)
    weights_vector = builder.EndVector()
    
    # Create category name
    category_offset = builder.CreateString(category)
    
    # Build CategoryWeights
    CategoryWeights.Start(builder)
    CategoryWeights.AddCategory(builder, category_offset)
    CategoryWeights.AddWeights(builder, weights_vector)
    CategoryWeights.AddDefaultWeight(builder, default_weight)
    return CategoryWeights.End(builder)

def main():
    # Define baseline weights (these can be tuned without recompiling C++)
    command_weights = {
        # Core actions
        "open": 1.5, "run": 1.5, "launch": 1.5, "start": 1.5,
        "close": 1.5, "stop": 1.5, "kill": 1.5, "end": 1.5,
        "show": 1.2, "display": 1.2, "list": 1.2, "get": 1.2,
        "set": 1.5, "change": 1.5, "update": 1.5, "modify": 1.5,
        "create": 1.5, "make": 1.5, "delete": 1.5, "remove": 1.5,
        "install": 1.5, "uninstall": 1.5, "download": 1.5,
        "restart": 1.5, "reboot": 1.5, "shutdown": 1.5,
        "search": 1.3, "find": 1.3, "locate": 1.3,
        "play": 1.3, "pause": 1.3, "resume": 1.3,
        "save": 1.3, "load": 1.3, "backup": 1.3,
        
        # Modal verbs
        "can": 0.5, "could": 0.5, "would": 0.3, "should": 0.5,
    }
    
    question_weights = {
        # Question words (strong indicators)
        "what": 2.0, "who": 2.0, "where": 2.0, "when": 2.0,
        "why": 2.0, "how": 2.0, "which": 1.8,
        "whats": 2.0, "whos": 2.0, "wheres": 2.0, "whens": 2.0,
        "whys": 2.0, "hows": 2.0,
        
        # Question auxiliaries (weak - only boost when combined)
        "is": 0.3, "are": 0.3, "was": 0.2, "were": 0.2,
        "do": 0.4, "does": 0.4, "did": 0.3,
        "will": 0.4, "shall": 0.4,
        
        # Information seeking
        "tell": 1.5, "explain": 1.8, "describe": 1.8,
        "define": 1.8, "mean": 1.5, "meaning": 1.5,
        "know": 1.2, "understand": 1.2,
        "wonder": 1.5, "curious": 1.5, "question": 1.5,
        
        # Location/context keywords
        "weather": 1.8, "temperature": 1.8, "forecast": 1.8,
        "near": 1.5, "nearby": 1.5, "local": 1.5, "around": 1.3,
        "location": 1.5, "place": 1.3, "store": 1.3, "restaurant": 1.3,
        "closest": 1.5, "nearest": 1.5,
        "best": 1.3, "recommendation": 1.5,
        "area": 1.3, "city": 1.3, "address": 1.5,
    }
    
    banter_weights = {
        # Greetings
        "hello": 2.0, "hi": 2.0, "hey": 2.0, "yo": 2.0,
        "morning": 1.8, "afternoon": 1.8, "evening": 1.8,
        "sup": 2.0, "wassup": 2.0, "howdy": 2.0,
        
        # Gratitude
        "thanks": 2.0, "thank": 2.0, "thx": 2.0, "ty": 2.0,
        "appreciate": 1.5, "grateful": 1.5,
        
        # Affirmations
        "yes": 1.0, "yeah": 1.0, "yep": 1.0, "yup": 1.0,
        "no": 1.0, "nope": 1.0, "nah": 1.0,
        "ok": 1.2, "okay": 1.2, "sure": 1.2, "alright": 1.2,
        
        # Reactions & Compliments
        "lol": 2.0, "haha": 2.0, "lmao": 2.0, "rofl": 2.0,
        "nice": 1.8, "cool": 1.8, "awesome": 1.8, "great": 1.8,
        "good": 1.5, "excellent": 1.8, "perfect": 1.8, "amazing": 1.8,
        "wonderful": 1.8, "fantastic": 1.8, "brilliant": 1.8,
        "love": 1.5, "loved": 1.5, "like": 1.3, "liked": 1.3,
        "wow": 1.5, "omg": 1.5, "damn": 1.5,
        
        # Feedback words
        "that": 1.2, "this": 1.2, "very": 1.3, "really": 1.3,
        "so": 1.2,
        
        # Farewells
        "bye": 2.0, "goodbye": 2.0, "later": 1.8, "cya": 2.0,
        "peace": 1.8, "night": 1.5,
        
        # Small talk
        "doing": 1.2, "feeling": 1.2, "bad": 0.8,
        
        # Internet slang
        "brb": 2.0, "afk": 2.0, "btw": 1.5, "imo": 1.5,
        "tbh": 1.5, "nvm": 1.5, "idk": 1.5,
    }
    
    # Build FlatBuffer
    builder = flatbuffers.Builder(1024)
    
    # Create category weights
    categories = []
    categories.append(create_category_weights(builder, "command", command_weights))
    categories.append(create_category_weights(builder, "question", question_weights))
    categories.append(create_category_weights(builder, "banter", banter_weights))
    
    # Create categories vector
    ClassifierConfig.StartCategoriesVector(builder, len(categories))
    for cat in reversed(categories):
        builder.PrependUOffsetTRelative(cat)
    categories_vector = builder.EndVector()
    
    # Create merge strategy string
    merge_strategy_offset = builder.CreateString("override")
    
    # Build ClassifierConfig
    ClassifierConfig.Start(builder)
    ClassifierConfig.AddVersion(builder, 1)
    ClassifierConfig.AddCategories(builder, categories_vector)
    ClassifierConfig.AddMergeStrategy(builder, merge_strategy_offset)
    ClassifierConfig.AddPriority(builder, 50)
    config = ClassifierConfig.End(builder)
    
    # Finish buffer
    builder.Finish(config)
    
    # Write to file
    output_path = os.path.join(os.path.dirname(__file__), '..', 'resources', 'classifier_weights.fb')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(builder.Output())
    
    print(f"Generated {output_path}")
    print(f"Size: {len(builder.Output())} bytes")
    print(f"Command weights: {len(command_weights)}")
    print(f"Question weights: {len(question_weights)}")
    print(f"Banter weights: {len(banter_weights)}")

if __name__ == '__main__':
    main()
