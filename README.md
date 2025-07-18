# PDF Outline Extractor

A robust solution for extracting structured document outlines from PDF files, including title and hierarchical headings (H1, H2, H3) with page numbers.

## Approach

### Core Strategy
Our solution uses a multi-layered approach to accurately identify document structure:

1. **Font-Based Analysis**: Analyzes font size, style, and formatting to identify potential headings
2. **Pattern Recognition**: Uses regex patterns to identify common heading formats (numbered, lettered, roman numerals)
3. **Heuristic Filtering**: Applies multiple heuristics to distinguish headings from regular text
4. **Hierarchical Classification**: Determines heading levels based on font size ratios and formatting

### Key Features

- **Robust Heading Detection**: Doesn't rely solely on font size; uses multiple indicators
- **Pattern-Based Recognition**: Identifies various heading formats (Chapter 1, 1.1, A., etc.)
- **Multilingual Support**: Works with Unicode text and various character sets
- **Duplicate Filtering**: Prevents duplicate headings in the output
- **Error Handling**: Gracefully handles corrupted or unusual PDF formats

### Algorithm Details

1. **Text Extraction**: Uses PyMuPDF to extract text with detailed font information
2. **Font Analysis**: Calculates average and maximum font sizes for relative comparison
3. **Title Detection**: Identifies the document title from the first page using font size and positioning
4. **Heading Classification**: 
   - Analyzes font size ratios (>1.2x average = potential heading)
   - Checks for bold/header fonts
   - Applies regex patterns for structured headings
   - Validates text length and content
5. **Level Assignment**: Assigns H1/H2/H3 levels based on font size hierarchy
6. **Text Cleaning**: Removes numbering prefixes while preserving meaningful content

## Libraries Used

- **PyMuPDF (fitz)**: Primary PDF processing library
  - Version: 1.23.14
  - Size: ~40MB
  - Features: Text extraction with font information, efficient processing
- **Python Standard Library**: `re`, `json`, `pathlib`, `logging`

## Architecture

```
pdf_extractor.py
├── PDFOutlineExtractor (main class)
│   ├── extract_text_with_formatting()    # PDF text extraction
│   ├── is_bold_or_header_font()          # Font analysis
│   ├── calculate_heading_level()         # Level determination
│   ├── extract_title_from_text_blocks()  # Title extraction
│   ├── is_likely_heading()               # Heading detection
│   ├── clean_heading_text()              # Text cleaning
│   ├── extract_outline()                 # Main extraction logic
│   └── process_directory()               # Batch processing
```

## Performance Optimizations

- **Efficient Text Processing**: Single-pass extraction with font information
- **Memory Management**: Processes one PDF at a time to minimize memory usage
- **Pattern Compilation**: Pre-compiled regex patterns for faster matching
- **Early Filtering**: Filters out obvious non-headings before expensive analysis

## Build and Run Instructions

### Build the Docker Image
```bash
docker build --platform linux/amd64 -t pdf-extractor:latest .
```

### Run the Container
```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  pdf-extractor:latest
```

### Input/Output Structure
- **Input**: Place PDF files in `./input/` directory
- **Output**: JSON files will be created in `./output/` directory
- **Naming**: `filename.pdf` → `filename.json`

## Sample Output Format

```json
{
  "title": "Understanding AI",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "What is AI?", "page": 2 },
    { "level": "H3", "text": "History of AI", "page": 3 },
    { "level": "H3", "text": "Types of AI", "page": 4 },
    { "level": "H2", "text": "Applications", "page": 5 }
  ]
}
```

## Technical Specifications

- **Platform**: Linux AMD64 (x86_64)
- **Runtime**: CPU-only, no GPU dependencies
- **Memory**: Optimized for 16GB RAM systems
- **Performance**: <10 seconds for 50-page PDFs
- **Model Size**: PyMuPDF library ~40MB (well under 200MB limit)
- **Network**: Fully offline, no internet required

## Handling Edge Cases

- **No Clear Title**: Returns `null` for title field
- **Inconsistent Formatting**: Uses multiple heuristics for robustness
- **Mixed Languages**: Supports Unicode text extraction
- **Complex Layouts**: Handles multi-column and complex page layouts
- **Corrupted PDFs**: Graceful error handling with empty output

## Testing Recommendations

Test with various PDF types:
- Academic papers with numbered sections
- Technical documentation with hierarchical structure
- Reports with chapter-based organization
- Multi-language documents
- PDFs with unusual formatting

## File Structure

```
project/
├── Dockerfile
├── requirements.txt
├── pdf_extractor.py
├── README.md
└── input/           # Place PDFs here
    └── output/      # JSON outputs appear here
```

## Compliance Notes

- ✅ AMD64 architecture compatibility
- ✅ No internet access required
- ✅ CPU-only execution
- ✅ Model size <200MB
- ✅ Performance optimized for <10s execution
- ✅ Handles up to 50-page PDFs
- ✅ Automatic batch processing
- ✅ Valid JSON output format