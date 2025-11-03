#!/usr/bin/env python3
"""
Direct Treebank to FlatBuffer Grammar Compiler
Extracts grammar from UD treebank and compiles directly to FlatBuffer binary.

Usage:
    python treebank_to_flatbuffer.py --treebank data/treebanks/en_ewt-ud-train.conllu --output resources/grammar_rules.fb
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List

# Add the generated FlatBuffer Python modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'nlp'))

try:
    import conllu
    import flatbuffers
    from GrammarRules import (
        GrammarConfig, GrammarComponent, CommandVerb, CommandObject,
        SentenceTemplate, ContextRule, LearningConfig
    )
except ImportError as e:
    print(f"Error: {e}")
    print("\nPlease ensure:")
    print("1. conllu is installed: pip install conllu")
    print("2. flatbuffers is installed: pip install flatbuffers")
    print("3. FlatBuffer bindings are generated: flatc --python nlp/grammar_rules.fbs")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class DirectTreebankCompiler:
    """Extract and compile grammar from treebank directly to FlatBuffer."""
    
    def __init__(self):
        self.verb_counts = Counter()
        self.object_counts = Counter()
        self.pattern_counts = Counter()
        self.total_sentences = 0
        
        # Pipeline category mappings
        self.pipeline_rules = {
            'command': ['open', 'close', 'start', 'stop', 'launch', 'run', 'execute',
                       'kill', 'restart', 'install', 'uninstall', 'delete', 'create',
                       'set', 'configure', 'enable', 'disable', 'turn', 'shut'],
            'query': ['what', 'when', 'where', 'who', 'why', 'how', 'is', 'are',
                     'can', 'could', 'would', 'should', 'find', 'search', 'lookup',
                     'show', 'display', 'get', 'fetch', 'retrieve', 'tell'],
            'banter': ['hello', 'hi', 'hey', 'thanks', 'thank', 'bye', 'goodbye',
                      'please', 'sorry', 'excuse', 'welcome', 'greetings'],
            'conversation': ['tell', 'say', 'speak', 'talk', 'discuss', 'chat',
                           'explain', 'describe', 'inform']
        }
        
    def categorize_verb(self, verb: str) -> str:
        """Map verb to pipeline category."""
        verb_lower = verb.lower()
        for category, keywords in self.pipeline_rules.items():
            if verb_lower in keywords:
                return category
        return 'command'  # Default
    
    def requires_object(self, verb: str) -> bool:
        """Determine if verb typically requires an object."""
        transitive_verbs = {
            'open', 'close', 'start', 'stop', 'find', 'search', 'create',
            'delete', 'show', 'tell', 'give', 'make', 'take', 'get',
            'set', 'configure', 'enable', 'disable', 'install', 'remove'
        }
        return verb.lower() in transitive_verbs
    
    def categorize_object(self, obj: str) -> str:
        """Categorize object semantically."""
        obj_lower = obj.lower()
        
        if any(kw in obj_lower for kw in ['browser', 'editor', 'player', 'app', 'program']):
            return 'application'
        elif any(kw in obj_lower for kw in ['file', 'document', 'folder', 'directory']):
            return 'file_system'
        elif any(kw in obj_lower for kw in ['music', 'video', 'song', 'movie']):
            return 'media'
        elif any(kw in obj_lower for kw in ['window', 'tab', 'pane']):
            return 'ui'
        return 'general'
    
    def extract_from_treebank(self, treebank_path: Path):
        """Extract patterns from treebank."""
        logger.info(f"Processing treebank: {treebank_path}")
        
        with open(treebank_path, 'r', encoding='utf-8') as f:
            data = f.read()
        
        sentences = conllu.parse(data)
        self.total_sentences = len(sentences)
        logger.info(f"Analyzing {self.total_sentences} sentences...")
        
        for sentence in sentences:
            try:
                for token in sentence:
                    # Extract verbs
                    if token['upos'] == 'VERB':
                        lemma = token['lemma'].lower()
                        self.verb_counts[lemma] += 1
                    
                    # Extract nouns as potential objects
                    elif token['upos'] in ['NOUN', 'PROPN']:
                        lemma = token['lemma'].lower()
                        self.object_counts[lemma] += 1
                
                # Extract structural patterns
                pattern = self._extract_pattern(sentence)
                if pattern:
                    self.pattern_counts[pattern] += 1
                    
            except Exception as e:
                logger.warning(f"Error processing sentence: {e}")
                continue
        
        logger.info(f"Extracted {len(self.verb_counts)} unique verbs")
        logger.info(f"Extracted {len(self.object_counts)} unique objects")
        logger.info(f"Extracted {len(self.pattern_counts)} unique patterns")
    
    def _extract_pattern(self, sentence) -> str:
        """Extract simplified constituent pattern."""
        pattern_parts = []
        
        for token in sentence:
            pos = token['upos']
            
            if pos in ['DET', 'ADJ', 'NOUN', 'PROPN']:
                if not pattern_parts or not pattern_parts[-1].startswith('NP'):
                    pattern_parts.append('NP')
            elif pos == 'VERB':
                pattern_parts.append('VP')
            elif pos in ['ADP']:
                pattern_parts.append('PP')
            elif pos in ['ADV']:
                if pattern_parts and pattern_parts[-1] == 'VP':
                    continue
                pattern_parts.append('ADVP')
        
        return ' '.join(pattern_parts) if pattern_parts else ''
    
    def compile_to_flatbuffer(self, output_path: Path, min_verb_freq=10, min_obj_freq=5, min_pattern_freq=20):
        """Compile extracted data directly to FlatBuffer binary."""
        logger.info("Compiling to FlatBuffer...")
        
        builder = flatbuffers.Builder(2048)
        
        # Build grammar components
        component_data = self._build_basic_components()
        component_offsets = []
        
        for name, comp in component_data.items():
            name_off = builder.CreateString(name)
            desc_off = builder.CreateString(comp['description'])
            
            pattern_offs = [builder.CreateString(p) for p in comp['patterns']]
            GrammarComponent.StartPatternsVector(builder, len(pattern_offs))
            for p in reversed(pattern_offs):
                builder.PrependUOffsetTRelative(p)
            patterns_vec = builder.EndVector()
            
            GrammarComponent.Start(builder)
            GrammarComponent.AddName(builder, name_off)
            GrammarComponent.AddDescription(builder, desc_off)
            GrammarComponent.AddPatterns(builder, patterns_vec)
            GrammarComponent.AddOptional(builder, comp['optional'])
            GrammarComponent.AddCapture(builder, comp['capture'])
            GrammarComponent.AddRequiresContext(builder, comp['requires_context'])
            GrammarComponent.AddFrequency(builder, 0)
            GrammarComponent.AddWeight(builder, 1.0)
            component_offsets.append(GrammarComponent.End(builder))
        
        GrammarConfig.StartComponentsVector(builder, len(component_offsets))
        for c in reversed(component_offsets):
            builder.PrependUOffsetTRelative(c)
        components_vec = builder.EndVector()
        
        # Build verbs from treebank data
        verb_offsets = []
        total_verbs = sum(self.verb_counts.values())
        
        for verb, count in self.verb_counts.most_common():
            if count < min_verb_freq:
                break
            
            weight = count / total_verbs if total_verbs > 0 else 0.5
            pipeline_cat = self.categorize_verb(verb)
            
            verb_off = builder.CreateString(verb)
            intent_off = builder.CreateString(f"{pipeline_cat}_{verb}")
            pipeline_off = builder.CreateString(pipeline_cat)
            
            # No synonyms vector for now (empty)
            CommandVerb.StartSynonymsVector(builder, 0)
            syn_vec = builder.EndVector()
            
            CommandVerb.Start(builder)
            CommandVerb.AddCanonical(builder, verb_off)
            CommandVerb.AddSynonyms(builder, syn_vec)
            CommandVerb.AddIntent(builder, intent_off)
            CommandVerb.AddRequiresObject(builder, self.requires_object(verb))
            CommandVerb.AddFrequency(builder, count)
            CommandVerb.AddWeight(builder, weight)
            CommandVerb.AddPipelineCategory(builder, pipeline_off)
            verb_offsets.append(CommandVerb.End(builder))
        
        GrammarConfig.StartVerbsVector(builder, len(verb_offsets))
        for v in reversed(verb_offsets):
            builder.PrependUOffsetTRelative(v)
        verbs_vec = builder.EndVector()
        
        # Build objects
        object_offsets = []
        total_objects = sum(self.object_counts.values())
        
        for obj, count in self.object_counts.most_common():
            if count < min_obj_freq:
                break
            
            weight = count / total_objects if total_objects > 0 else 0.5
            category = self.categorize_object(obj)
            
            obj_off = builder.CreateString(obj)
            cat_off = builder.CreateString(category)
            
            CommandObject.StartSynonymsVector(builder, 0)
            syn_vec = builder.EndVector()
            
            CommandObject.Start(builder)
            CommandObject.AddCanonical(builder, obj_off)
            CommandObject.AddSynonyms(builder, syn_vec)
            CommandObject.AddCategory(builder, cat_off)
            CommandObject.AddFrequency(builder, count)
            CommandObject.AddWeight(builder, weight)
            object_offsets.append(CommandObject.End(builder))
        
        GrammarConfig.StartObjectsVector(builder, len(object_offsets))
        for o in reversed(object_offsets):
            builder.PrependUOffsetTRelative(o)
        objects_vec = builder.EndVector()
        
        # Build templates from patterns
        template_offsets = []
        total_patterns = sum(self.pattern_counts.values())
        
        for pattern, count in self.pattern_counts.most_common():
            if count < min_pattern_freq:
                break
            
            weight = count / total_patterns if total_patterns > 0 else 0.5
            structure = self._pattern_to_template(pattern)
            pipeline_cat = self._categorize_pattern(pattern)
            
            name_off = builder.CreateString(f"template_{len(template_offsets)}")
            struct_off = builder.CreateString(structure)
            cat_off = builder.CreateString(pipeline_cat)
            pipeline_off = builder.CreateString(pipeline_cat)
            
            SentenceTemplate.StartExamplesVector(builder, 0)
            ex_vec = builder.EndVector()
            
            SentenceTemplate.StartCommonModifiersVector(builder, 0)
            mod_vec = builder.EndVector()
            
            SentenceTemplate.Start(builder)
            SentenceTemplate.AddName(builder, name_off)
            SentenceTemplate.AddStructure(builder, struct_off)
            SentenceTemplate.AddExamples(builder, ex_vec)
            SentenceTemplate.AddRequiresContext(builder, False)
            SentenceTemplate.AddCategory(builder, cat_off)
            SentenceTemplate.AddFrequency(builder, count)
            SentenceTemplate.AddWeight(builder, weight)
            SentenceTemplate.AddPipelineCategory(builder, pipeline_off)
            SentenceTemplate.AddAvgDepth(builder, 0.0)
            SentenceTemplate.AddCommonModifiers(builder, mod_vec)
            template_offsets.append(SentenceTemplate.End(builder))
        
        GrammarConfig.StartTemplatesVector(builder, len(template_offsets))
        for t in reversed(template_offsets):
            builder.PrependUOffsetTRelative(t)
        templates_vec = builder.EndVector()
        
        # Build learning config
        LearningConfig.Start(builder)
        LearningConfig.AddEnableOnlineLearning(builder, True)
        LearningConfig.AddMinConfidenceThreshold(builder, 0.6)
        LearningConfig.AddFeedbackWeight(builder, 0.3)
        LearningConfig.AddDecayRate(builder, 0.95)
        learning_off = LearningConfig.End(builder)
        
        # Empty context rules vector (must be created before GrammarConfig.Start)
        GrammarConfig.StartContextRulesVector(builder, 0)
        context_vec = builder.EndVector()
        
        # Build root config strings
        lang_off = builder.CreateString("en")
        source_off = builder.CreateString("UD_English-EWT")
        merge_off = builder.CreateString("weighted")
        
        # Build root config
        GrammarConfig.Start(builder)
        GrammarConfig.AddVersion(builder, 1)
        GrammarConfig.AddLanguage(builder, lang_off)
        GrammarConfig.AddTreebankSource(builder, source_off)
        GrammarConfig.AddComponents(builder, components_vec)
        GrammarConfig.AddVerbs(builder, verbs_vec)
        GrammarConfig.AddObjects(builder, objects_vec)
        GrammarConfig.AddTemplates(builder, templates_vec)
        GrammarConfig.AddContextRules(builder, context_vec)
        GrammarConfig.AddLearningConfig(builder, learning_off)
        GrammarConfig.AddGeneratedTimestamp(builder, int(time.time()))
        GrammarConfig.AddTotalSentencesAnalyzed(builder, self.total_sentences)
        GrammarConfig.AddMergeStrategy(builder, merge_off)
        
        config = GrammarConfig.End(builder)
        builder.Finish(config)
        
        # Write to file
        buf = builder.Output()
        with open(output_path, 'wb') as f:
            f.write(buf)
        
        logger.info(f"✓ Compiled grammar to: {output_path}")
        logger.info(f"  Size: {len(buf):,} bytes")
        logger.info(f"  Components: {len(component_data)}")
        logger.info(f"  Verbs: {len(verb_offsets)} (from {len(self.verb_counts)} unique)")
        logger.info(f"  Objects: {len(object_offsets)} (from {len(self.object_counts)} unique)")
        logger.info(f"  Templates: {len(template_offsets)} (from {len(self.pattern_counts)} patterns)")
        logger.info(f"  Sentences analyzed: {self.total_sentences:,}")
    
    def _build_basic_components(self) -> Dict:
        """Build basic grammar components."""
        return {
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
    
    def _pattern_to_template(self, pattern: str) -> str:
        """Convert POS pattern to template structure."""
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


def main():
    parser = argparse.ArgumentParser(
        description='Compile UD treebank directly to FlatBuffer grammar'
    )
    parser.add_argument('--treebank', '-t', type=Path, required=True,
                       help='Path to CoNLL-U treebank file')
    parser.add_argument('--output', '-o', type=Path,
                       default=Path('resources/grammar_rules.fb'),
                       help='Output FlatBuffer file')
    parser.add_argument('--min-verb-freq', type=int, default=10,
                       help='Minimum verb frequency threshold')
    parser.add_argument('--min-obj-freq', type=int, default=5,
                       help='Minimum object frequency threshold')
    parser.add_argument('--min-pattern-freq', type=int, default=20,
                       help='Minimum pattern frequency threshold')
    
    args = parser.parse_args()
    
    if not args.treebank.exists():
        logger.error(f"Treebank not found: {args.treebank}")
        return 1
    
    try:
        compiler = DirectTreebankCompiler()
        compiler.extract_from_treebank(args.treebank)
        compiler.compile_to_flatbuffer(
            args.output,
            args.min_verb_freq,
            args.min_obj_freq,
            args.min_pattern_freq
        )
        logger.info("✓ Compilation complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
