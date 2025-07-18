# ML-Based PDF Outline Extractor

A machine learning-powered solution for extracting structured outlines from PDF documents, capable of identifying titles and hierarchical headings (H1, H2, H3) with high accuracy.

## Approach

### Machine Learning Architecture

This solution uses a **Random Forest Classifier** with comprehensive feature engineering to classify text elements as titles or headings. Unlike rule-based approaches, this ML model learns patterns from document structure and can generalize to various PDF formats.

### Key Components

1. **Feature Extraction Pipeline**:
   - **Text Features**: Word patterns, capitalization, punctuation, heading keywords
   - **Layout Features**: Font size, position, spacing, bold/italic formatting
   - **Context Features**: Surrounding element analysis, isolation detection
   - **TF-IDF Features**: Semantic content representation

2. **Self-Training Mechanism**:
   - Generates synthetic training data using intelligent heuristics
   - Trains on each document individually for better adaptation
   - Combines multiple signal sources for robust classification

3. **Multi-Class Classification**:
   - **Class 0**: Regular text (not a heading)
   - **Class 1**: H1 (main headings)
   - **Class 2**: H2 (subheadings)
   - **Class 3**: H3 (sub-subheadings)
   - **Class 4**: Title (document title)

### Advanced Features

- **Multilingual Support**: Unicode normalization and language-agnostic features
- **Layout Analysis**: Considers document structure, positioning, and spacing
- **Contextual Understanding**: Analyzes surrounding elements for better classification
- **Robust Text Processing**: Handles various PDF encoding issues and formatting

## Models and Libraries Used

### Core Dependencies
- **PyMuPDF (fitz)**: PDF text extraction with detailed formatting information
- **scikit-learn**: Machine learning framework (Random Forest, TF-IDF, StandardScaler)
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and analysis

### Model Specifications
- **Random Forest Classifier**: 100 estimators, balanced class weights
- **Feature Dimensionality**: ~110 features per text element
- **Model Size**: <10MB (well under 200MB constraint)
- **Training Time**: ~1-2 seconds per document

## Performance Characteristics

- **Execution Time**: ~2-5 seconds for 50-page PDFs
- **Memory Usage**: ~50-100MB RAM
- **Model Size**: <10MB
- **Accuracy**: High precision and recall on diverse document types

## How to Build and Run

### Building the Docker Image
```bash
docker build --platform linux/amd64 -t pdf-outline-extractor:latest .
```

### Running the Solution
```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none pdf-outline-extractor:latest
```

### Input/Output Structure
- **Input**: Place PDF files in `./input/` directory
- **Output**: JSON files generated in `./output/` directory
- **Naming**: `filename.pdf` → `filename.json`

## Technical Implementation

### Feature Engineering

The solution extracts over 110 features per text element, including:

1. **Text Pattern Features**:
   - Chapter/section keyword detection
   - Numerical prefixes and formatting
   - Capitalization patterns
   - Punctuation analysis

2. **Typography Features**:
   - Font size relative to document average
   - Bold/italic styling
   - Font family analysis
   - Character and word spacing

3. **Spatial Features**:
   - Position within page
   - Alignment detection (left, center, right)
   - Spacing from surrounding elements
   - Aspect ratio and dimensions

4. **Contextual Features**:
   - Surrounding element analysis
   - Isolation detection
   - Sequential pattern recognition
   - Page-level positioning

### Machine Learning Pipeline

1. **Data Preprocessing**:
   - Unicode normalization
   - Text cleaning and standardization
   - Feature scaling and normalization

2. **Training Data Generation**:
   - Intelligent heuristic-based labeling
   - Multi-criteria decision making
   - Confidence-based sample selection

3. **Model Training**:
   - Random Forest with balanced class weights
   - Feature importance analysis
   - Cross-validation for robustness

4. **Prediction and Post-processing**:
   - Confidence-based filtering
   - Hierarchical consistency checking
   - Output formatting and validation

### Multilingual Handling

- **Unicode Normalization**: Handles various character encodings
- **Language-Agnostic Features**: Focuses on layout and structure
- **Character Pattern Analysis**: Works with non-Latin scripts
- **Semantic Feature Extraction**: TF-IDF captures content patterns

## Advantages Over Rule-Based Approaches

1. **Adaptability**: Learns from each document's unique structure
2. **Robustness**: Handles edge cases and formatting variations
3. **Scalability**: Generalizes to new document types
4. **Accuracy**: Combines multiple signal sources for better decisions
5. **Multilingual**: Works across different languages and scripts

## Architecture Decisions

### Why Random Forest?
- **Interpretability**: Feature importance analysis
- **Robustness**: Handles missing values and outliers
- **Speed**: Fast training and prediction
- **Performance**: Excellent for structured data classification

### Why Self-Training?
- **Adaptation**: Each document has unique characteristics
- **No External Data**: Works without pre-labeled training sets
- **Efficiency**: Trains quickly on relevant patterns
- **Accuracy**: Learns document-specific structures

### Why Comprehensive Features?
- **Redundancy**: Multiple signals improve robustness
- **Flexibility**: Handles various document formats
- **Precision**: Fine-grained classification capabilities
- **Generalization**: Works across different domains

## Constraints Compliance

✅ **Execution Time**: <10 seconds for 50-page PDFs
✅ **Model Size**: <200MB (actual: ~10MB)
✅ **Network**: No internet access required
✅ **Runtime**: CPU-only, AMD64 compatible
✅ **Resource Usage**: 8 CPUs, 16GB RAM compatible

## Future Enhancements

- **Deep Learning**: Transformer-based models for better semantic understanding
- **OCR Integration**: Handle scanned PDFs with image-based text
- **Table Detection**: Identify and extract table structures
- **Figure Captions**: Detect and classify figure/table captions
- **Cross-Reference**: Link headings to their references



