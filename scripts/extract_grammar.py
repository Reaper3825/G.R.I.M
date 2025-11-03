#!/usr/bin/env python3
"""
Treebank Grammar Extractor for G.R.I.M
Extracts grammar rules from Universal Dependencies treebank and generates weighted rule sets.

Usage:
    python extract_grammar.py --treebank path/to/treebank.conllu --output nlp_grammar.json
    
Requirements:
    pip install conllu pyconll
"""

import argparse
import json
import logging
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple
import re

try:
    import conllu
except ImportError:
    print("Please install conllu: pip install conllu")
    exit(1)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class TreebankGrammarExtractor:
    """Extract grammar rules from UD treebank for NLP pipeline."""
    
    def __init__(self):
        # Rule collections
        self.verb_counts = Counter()
        self.object_counts = Counter()
        self.pattern_counts = Counter()
        self.dependency_patterns = defaultdict(list)
        
        # Component tracking
        self.components = {}
        self.verbs = {}
        self.objects = {}
        self.templates = []
        
        # Pipeline category mappings
        self.pipeline_rules = {
            'command': ['open', 'close', 'start', 'stop', 'launch', 'run', 'execute', 
                       'kill', 'restart', 'install', 'uninstall', 'delete', 'create'],
            'query': ['what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 
                     'can', 'could', 'would', 'should', 'find', 'search', 'lookup'],
            'banter': ['hello', 'hi', 'hey', 'thanks', 'thank', 'bye', 'goodbye', 
                      'please', 'sorry', 'excuse'],
            'conversation': ['tell', 'say', 'speak', 'talk', 'discuss', 'chat']
        }
        
    def categorize_verb(self, verb: str) -> str:
        """Map verb to pipeline category."""
        verb_lower = verb.lower()
        for category, keywords in self.pipeline_rules.items():
            if verb_lower in keywords or any(kw in verb_lower for kw in keywords):
                return category
        return 'command'  # Default to command
    
    def extract_from_sentence(self, sentence):
        """Extract grammar components from a single sentence."""
        try:
            # Extract verbs and their frequencies
            for token in sentence:
                if token['upos'] == 'VERB':
                    lemma = token['lemma'].lower()
                    self.verb_counts[lemma] += 1
                    
                    # Track verb dependencies
                    if token['head'] != 0:
                        dep_rel = token['deprel']
                        self.dependency_patterns[lemma].append(dep_rel)
                
                # Extract nouns as potential objects
                elif token['upos'] in ['NOUN', 'PROPN']:
                    lemma = token['lemma'].lower()
                    self.object_counts[lemma] += 1
            
            # Extract structural patterns (simplified CFG rules)
            pattern = self._extract_pattern(sentence)
            if pattern:
                self.pattern_counts[pattern] += 1
                
        except Exception as e:
            logger.warning(f"Error processing sentence: {e}")
    
    def _extract_pattern(self, sentence) -> str:
        """Extract simplified constituent pattern from sentence."""
        # Simplified pattern based on POS tags
        pattern_parts = []
        
        for token in sentence:
            pos = token['upos']
            
            # Group into major constituents
            if pos in ['DET', 'ADJ', 'NOUN', 'PROPN']:
                if not pattern_parts or not pattern_parts[-1].startswith('NP'):
                    pattern_parts.append('NP')
            elif pos == 'VERB':
                pattern_parts.append('VP')
            elif pos in ['ADP']:
                pattern_parts.append('PP')
            elif pos in ['ADV']:
                if pattern_parts and pattern_parts[-1] == 'VP':
                    continue  # Adverbs modify verbs
                pattern_parts.append('ADVP')
        
        return ' '.join(pattern_parts) if pattern_parts else None
    
    def extract_from_file(self, treebank_path: Path):
        """Extract grammar from CoNLL-U format treebank file."""
        logger.info(f"Processing treebank: {treebank_path}")
        
        with open(treebank_path, 'r', encoding='utf-8') as f:
            data = f.read()
        
        sentences = conllu.parse(data)
        logger.info(f"Found {len(sentences)} sentences")
        
        for sentence in sentences:
            self.extract_from_sentence(sentence)
        
        logger.info(f"Extracted {len(self.verb_counts)} unique verbs")
        logger.info(f"Extracted {len(self.object_counts)} unique objects")
        logger.info(f"Extracted {len(self.pattern_counts)} unique patterns")
    
    def generate_grammar_components(self):
        """Generate grammar components with weights."""
        # Basic components
        self.components = {
            "greeting": {
                "description": "Greeting phrases",
                "patterns": ["^(hi|hello|hey|greetings?)\\b"],
                "optional": True,
                "capture": False,
                "requires_context": False
            },
            "polite": {
                "description": "Polite modifiers",
                "patterns": ["\\b(please|kindly|could you|would you|can you)\\b"],
                "optional": True,
                "capture": False,
                "requires_context": False
            },
            "article": {
                "description": "Articles",
                "patterns": ["\\b(a|an|the)\\b"],
                "optional": True,
                "capture": False,
                "requires_context": False
            },
            "determiner": {
                "description": "Determiners",
                "patterns": ["\\b(this|that|these|those|my|your)\\b"],
                "optional": True,
                "capture": False,
                "requires_context": False
            }
        }
    
    def generate_command_verbs(self, min_frequency: int = 2):
        """Generate command verbs with pipeline categories."""
        total_verbs = sum(self.verb_counts.values())
        
        for verb, count in self.verb_counts.most_common():
            if count < min_frequency:
                break
            
            weight = count / total_verbs
            pipeline_cat = self.categorize_verb(verb)
            
            self.verbs[verb] = {
                "intent": f"{pipeline_cat}_{verb}",
                "synonyms": self._find_synonyms(verb),
                "requires_object": self._requires_object(verb),
                "pipeline_category": pipeline_cat,
                "frequency": count,
                "weight": round(weight, 4)
            }
    
    def _find_synonyms(self, verb: str) -> List[str]:
        """Find common synonyms for verb (placeholder - enhance with WordNet)."""
        # Basic synonym mapping
        synonym_map = {
            'open': ['launch', 'start', 'run'],
            'close': ['quit', 'exit', 'terminate'],
            'search': ['find', 'lookup', 'query', 'google'],
            'create': ['make', 'generate', 'build'],
            'delete': ['remove', 'erase', 'destroy'],
            'show': ['display', 'present', 'reveal'],
            'tell': ['say', 'speak', 'inform']
        }
        return synonym_map.get(verb, [])
    
    def _requires_object(self, verb: str) -> bool:
        """Determine if verb typically requires an object."""
        # Verbs that usually take objects
        transitive_verbs = {
            'open', 'close', 'start', 'stop', 'find', 'search', 'create', 
            'delete', 'show', 'tell', 'give', 'make', 'take', 'get'
        }
        return verb in transitive_verbs
    
    def generate_command_objects(self, min_frequency: int = 2):
        """Generate command objects from noun frequencies."""
        total_objects = sum(self.object_counts.values())
        
        for obj, count in self.object_counts.most_common():
            if count < min_frequency:
                break
            
            weight = count / total_objects
            category = self._categorize_object(obj)
            
            self.objects[obj] = {
                "category": category,
                "synonyms": [],
                "frequency": count,
                "weight": round(weight, 4)
            }
    
    def _categorize_object(self, obj: str) -> str:
        """Categorize object into semantic category."""
        # Simple categorization (enhance with ontology)
        app_keywords = ['browser', 'editor', 'player', 'app', 'program', 'software']
        file_keywords = ['file', 'document', 'folder', 'directory', 'image', 'video']
        
        obj_lower = obj.lower()
        if any(kw in obj_lower for kw in app_keywords):
            return 'application'
        elif any(kw in obj_lower for kw in file_keywords):
            return 'file_system'
        return 'general'
    
    def generate_sentence_templates(self, min_frequency: int = 5):
        """Generate sentence templates from patterns."""
        total_patterns = sum(self.pattern_counts.values())
        
        for pattern, count in self.pattern_counts.most_common():
            if count < min_frequency:
                break
            
            weight = count / total_patterns
            
            # Convert pattern to template structure
            structure = self._pattern_to_template(pattern)
            pipeline_cat = self._categorize_pattern(pattern)
            
            template = {
                "structure": structure,
                "examples": [],
                "requires_context": False,
                "category": pipeline_cat,
                "frequency": count,
                "weight": round(weight, 4),
                "pipeline_category": pipeline_cat
            }
            
            self.templates.append((f"template_{len(self.templates)}", template))
    
    def _pattern_to_template(self, pattern: str) -> str:
        """Convert POS pattern to template structure."""
        # Map patterns to template syntax
        mapping = {
            'NP': '<object>',
            'VP': '<verb>',
            'PP': '[prep_phrase]',
            'ADVP': '[adverb]'
        }
        
        parts = pattern.split()
        template_parts = [mapping.get(p, f'[{p.lower()}]') for p in parts]
        return ' '.join(template_parts)
    
    def _categorize_pattern(self, pattern: str) -> str:
        """Determine pipeline category from pattern."""
        if 'VP' in pattern and 'NP' in pattern:
            return 'command'
        elif pattern.startswith('NP VP'):
            return 'query'
        return 'conversation'
    
    def generate_json_output(self, output_path: Path):
        """Generate final JSON grammar file."""
        self.generate_grammar_components()
        self.generate_command_verbs()
        self.generate_command_objects()
        self.generate_sentence_templates()
        
        grammar = {
            "version": 1,
            "language": "en",
            "treebank_source": "Universal Dependencies",
            "grammar_components": self.components,
            "command_verbs": self.verbs,
            "command_objects": self.objects,
            "sentence_templates": dict(self.templates),
            "context_rules": {},
            "learning_config": {
                "enable_online_learning": True,
                "min_confidence_threshold": 0.6,
                "feedback_weight": 0.3,
                "decay_rate": 0.95
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(grammar, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Grammar written to: {output_path}")
        logger.info(f"  Components: {len(self.components)}")
        logger.info(f"  Verbs: {len(self.verbs)}")
        logger.info(f"  Objects: {len(self.objects)}")
        logger.info(f"  Templates: {len(self.templates)}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract grammar rules from UD treebank for G.R.I.M'
    )
    parser.add_argument(
        '--treebank', '-t',
        type=Path,
        required=True,
        help='Path to CoNLL-U format treebank file'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('resources/nlp_grammar.json'),
        help='Output JSON file path'
    )
    parser.add_argument(
        '--min-verb-freq', '-v',
        type=int,
        default=2,
        help='Minimum frequency for including verbs'
    )
    parser.add_argument(
        '--min-pattern-freq', '-p',
        type=int,
        default=5,
        help='Minimum frequency for including patterns'
    )
    
    args = parser.parse_args()
    
    if not args.treebank.exists():
        logger.error(f"Treebank file not found: {args.treebank}")
        return 1
    
    # Extract grammar
    extractor = TreebankGrammarExtractor()
    extractor.extract_from_file(args.treebank)
    extractor.generate_json_output(args.output)
    
    logger.info("✓ Grammar extraction complete!")
    return 0


if __name__ == '__main__':
    exit(main())
