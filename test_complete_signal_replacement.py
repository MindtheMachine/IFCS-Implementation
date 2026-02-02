#!/usr/bin/env python3
"""
Test complete replacement of text-matching heuristics with signal estimation
across ALL components (IFCS, ECR, Control Probe)
"""

import sys
import os
import re
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def scan_for_text_matching_patterns(filepath: str) -> list:
    """Scan a file for text-matching heuristic patterns"""
    patterns_found = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
        # Look for regex word boundary patterns (text-matching heuristics)
        regex_patterns = [
            r'r\'\\b.*\\b\'',  # r'\bword\b' patterns
            r'r"\\b.*\\b"',    # r"\bword\b" patterns
            r'\\b\w+\\b',      # \bword\b in strings
            r'findall.*\\b',   # re.findall with word boundaries
            r'search.*\\b',    # re.search with word boundaries
            r'match.*\\b',     # re.match with word boundaries
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern in regex_patterns:
                if re.search(pattern, line):
                    patterns_found.append({
                        'file': filepath,
                        'line': i,
                        'content': line.strip(),
                        'pattern_type': 'regex_word_boundary'
                    })
        
        # Look for hardcoded word lists (text-matching heuristics)
        word_list_patterns = [
            r'\[.*[\'"][a-z]+[\'"].*\]',  # ['word1', 'word2'] lists
            r'in \[.*[\'"][a-z]+[\'"].*\]',  # word in ['list'] checks
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern in word_list_patterns:
                if re.search(pattern, line) and 'word' in line.lower():
                    patterns_found.append({
                        'file': filepath,
                        'line': i,
                        'content': line.strip(),
                        'pattern_type': 'hardcoded_word_list'
                    })
                    
    except Exception as e:
        print(f"Error scanning {filepath}: {e}")
    
    return patterns_found

def test_complete_signal_replacement():
    """Test that text-matching heuristics are replaced with signal estimation across all components"""
    print("COMPREHENSIVE TEXT-MATCHING HEURISTIC SCAN")
    print("=" * 60)
    print("Industry requirement: Replace text-matching with signal estimation")
    print("Scanning all components: IFCS, ECR, Control Probe, Benchmarks")
    print()
    
    # Files to scan for text-matching heuristics
    files_to_scan = [
        'ifcs_engine.py',
        'semantic_analyzer.py', 
        'ecr_engine.py',
        'control_probe.py',
        'benchmark_metrics.py',
        'benchmark_adapters.py'
    ]
    
    all_patterns = []
    
    for filepath in files_to_scan:
        if os.path.exists(filepath):
            patterns = scan_for_text_matching_patterns(filepath)
            all_patterns.extend(patterns)
            
            component_name = os.path.basename(filepath).replace('.py', '').upper()
            print(f"[{component_name}] Found {len(patterns)} text-matching patterns")
            
            for pattern in patterns:
                print(f"  Line {pattern['line']}: {pattern['content'][:80]}...")
        else:
            print(f"[WARNING] File not found: {filepath}")
    
    print()
    print("=" * 60)
    print("SIGNAL REPLACEMENT ANALYSIS")
    print("=" * 60)
    
    if all_patterns:
        print(f"❌ INCOMPLETE: Found {len(all_patterns)} text-matching heuristics")
        print()
        print("Components still using text-matching:")
        
        by_component = {}
        for pattern in all_patterns:
            component = os.path.basename(pattern['file']).replace('.py', '')
            if component not in by_component:
                by_component[component] = 0
            by_component[component] += 1
        
        for component, count in by_component.items():
            print(f"  • {component}: {count} patterns")
        
        print()
        print("REQUIRED ACTIONS:")
        print("1. Replace regex word boundary patterns with signal estimation")
        print("2. Replace hardcoded word lists with semantic signal computation")
        print("3. Use industry-standard approach: estimate latent epistemic signals")
        print("4. Apply fuzzy logic over signals (not fuzzy text matching)")
        
        return False
    else:
        print("✅ COMPLETE: No text-matching heuristics found")
        print("✅ All components use signal estimation approach")
        print("✅ Industry-standard implementation achieved")
        
        return True

if __name__ == "__main__":
    success = test_complete_signal_replacement()
    
    if success:
        print("\n🎉 SIGNAL-BASED ARCHITECTURE COMPLETE")
        print("All components successfully replaced text-matching with signal estimation")
    else:
        print("\n⚠️  SIGNAL REPLACEMENT INCOMPLETE")
        print("Additional work needed to complete industry-standard approach")