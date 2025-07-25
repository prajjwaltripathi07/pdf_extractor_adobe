# Multilingual PDF Outline Extractor

A machine learning-based solution for extracting structured outlines from PDF documents with multilingual support.

## Features

- **Multilingual Support**: Handles English, Japanese, Chinese, Spanish, French, and German documents
- **Smart Heading Detection**: Uses ML model trained on document-specific heuristics
- **Lightweight**: Optimized for speed and size constraints (model < 200MB)
- **Offline Operation**: No internet connectivity required
- **Fast Processing**: Processes 50-page PDFs in under 10 seconds

## Architecture

### Core Components

1. **PDFProcessor**: Extracts text elements with formatting information using PyMuPDF
2. **MultilingualFeatureExtractor**: Creates features for ML classification with multilingual patterns
3. **HeadingClassifier**: Random Forest classifier for heading level prediction
4. **PDFOutlineExtractor**: Main orchestrator that combines all components

### Feature Engineering

The system extracts multiple types of features:

- **Text Features**: Language-agnostic patterns, multilingual keywords, character analysis
- **Layout Features**: Font size, positioning, styling (bold/italic)
- **Context Features**: Relationship with surrounding elements, document structure

### Multilingual Patterns

Supports heading detection for:
- **English**: "Chapter", "Section", "Introduction", etc.
- **Japanese**: "章", "節", "序論", "結論", etc.
- **Chinese**: Similar character patterns
- **Spanish**: "Capítulo", "Sección", "Introducción", etc.
- **French**: "Chapitre", "Section", "Introduction", etc.
- **German**: "Kapitel", "Abschnitt", "Einleitung", etc.

## Model Training

The system uses a hybrid approach:
1. **Heuristic-based Training**: Generates training data using multilingual patterns
2. **Random Forest Classification**: Lightweight model optimized for speed
3. **Feature Scaling**: Standardized features for better performance

## Technical Specifications

- **Model Size**: < 200MB
- **Processing Time**: ≤ 10 seconds for 50-page PDF
- **Memory Usage**: Optimized for 16GB RAM systems
- **CPU Architecture**: AMD64 (x86_64)
- **No GPU Required**: Runs entirely on CPU

## Usage

### Docker Build and Run

```bash
# Build the Docker image
docker build --platform linux/amd64 -t pdf-outline-extractor .

# Run the container
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none pdf-outline-extractor
```

### Input/Output

- **Input**: PDF files in `/app/input` directory
- **Output**: JSON files in `/app/output` directory

### Output Format

```json
{
  "title": "Document Title",
  "outline": [
    {"level": "H1", "text": "Chapter 1: Introduction", "page": 1},
    {"level": "H2", "text": "Background", "page": 2},
    {"level": "H3", "text": "Related Work", "page": 3}
  ]
}
```

## Dependencies

- **PyMuPDF**: PDF text extraction with formatting
- **scikit-learn**: Machine learning framework
- **NumPy**: Numerical computations
- **Standard Library**: JSON, logging, regex, unicodedata

## Optimization Strategies

1. **Reduced Model Complexity**: Smaller Random Forest with optimized parameters
2. **Feature Selection**: Focused on most discriminative features
3. **Memory Management**: Efficient data structures and processing
4. **Character-based Analysis**: Better multilingual support with char-level patterns
