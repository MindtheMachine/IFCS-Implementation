# Text Output Capture Summary

## Overview
The Trilogy system captures complete text outputs from both baseline and regulated responses in multiple formats for comprehensive analysis.

## Output Files and Content

### 1. CSV Results (`*_results.csv`)
**Default behavior:**
- Contains metrics, mechanism firing data, and processing statistics
- Does NOT include full text responses (keeps file size manageable)
- Includes `prompt_preview` (first 100 characters of prompt)

**With `--include-full-text` flag:**
- Adds `baseline_response` column with full baseline LLM response
- Adds `regulated_response` column with full regulated (trilogy) response
- Text is cleaned (newlines removed) and truncated to 1000 characters
- Makes CSV files significantly larger but enables spreadsheet analysis

**Usage:**
```bash
# Standard CSV (metrics only)
python trilogy_app.py --benchmark truthfulqa --batch-size 5

# CSV with full text responses
python trilogy_app.py --benchmark truthfulqa --batch-size 5 --include-full-text
```

### 2. Comparison Report (`*_comparison.txt`)
**Always includes full text responses:**
- Complete baseline response (unregulated LLM output)
- Complete regulated response (trilogy system output)
- Side-by-side comparison for first 20 examples
- Mechanism firing details
- Metric improvements per example
- Human-readable format for detailed analysis

**Example format:**
```
EXAMPLE 1: truthfulqa_0
Prompt: What is the smallest country...

BASELINE (Unregulated)
A) Nauru is the smallest country...

REGULATED (Trilogy)
I cannot provide a confident response...
Mechanisms Fired: ECR: True, CP Type-1: True...
```

### 3. JSON Summary (`*_summary.json`)
**Contains:**
- Aggregated statistics and metrics
- Mechanism firing rates
- Processing statistics
- Does NOT include individual text responses (focuses on aggregate data)

### 4. HTML Report (`*_report.html`)
**Contains:**
- Visual charts and tables
- Aggregated statistics
- Does NOT include individual text responses
- Optimized for executive summaries

## Data Structure in Memory

The `BenchmarkEvaluationResult` object stores:
```python
@dataclass
class BenchmarkEvaluationResult:
    baseline_response: str              # Full baseline text
    regulated_result: TrilogyResult     # Contains final_response with full regulated text
    # ... other fields
```

Where `regulated_result.final_response` contains the complete trilogy system output.

## Storage Locations

Results are organized by model name:
```
Results/
├── mistral-7b-instruct-q4_K_M/
│   ├── truthfulqa_results.csv          # Metrics (+ optional full text)
│   ├── truthfulqa_comparison.txt       # Full text responses (always)
│   ├── truthfulqa_summary.json         # Aggregated stats
│   └── truthfulqa_report.html          # Visual report
└── claude-sonnet-4-20250514/
    └── ... (same structure)
```

## Recommendations

### For Analysis Workflows:

1. **Quick metrics analysis:** Use CSV without full text
2. **Detailed response analysis:** Use comparison.txt file
3. **Spreadsheet analysis with text:** Use CSV with `--include-full-text`
4. **Executive reporting:** Use HTML report
5. **Statistical analysis:** Use JSON summary

### For Different Use Cases:

- **Research papers:** Comparison.txt for examples, JSON for statistics
- **Quality assurance:** CSV with full text for systematic review
- **Performance monitoring:** JSON summary for dashboards
- **Stakeholder reports:** HTML report for presentations

## File Size Considerations

- **CSV without full text:** ~50KB for 100 examples
- **CSV with full text:** ~500KB-2MB for 100 examples (depends on response length)
- **Comparison.txt:** ~200KB-1MB for 20 examples
- **JSON summary:** ~10KB (constant size)
- **HTML report:** ~50KB (constant size)

## Command Line Examples

```bash
# Standard benchmark (all outputs except CSV full text)
python trilogy_app.py --benchmark truthfulqa --batch-size 50

# Include full text in CSV for detailed analysis
python trilogy_app.py --benchmark truthfulqa --batch-size 50 --include-full-text

# Custom batch with full text
python trilogy_app.py --benchmark truthfulqa --batch-size 100 --batch-start 50 --include-full-text
```

## Integration with Analysis Tools

The multiple output formats support different analysis workflows:
- **Python/Pandas:** Load CSV for quantitative analysis
- **R/Excel:** Use CSV with full text for mixed analysis
- **Text analysis tools:** Use comparison.txt for qualitative analysis
- **Dashboards:** Use JSON summary for real-time monitoring
- **Reports:** Use HTML for stakeholder communication