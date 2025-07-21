import os
import json
import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import fitz  # PyMuPDF
import unicodedata
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextElement:
    """Represents a text element extracted from PDF with features"""
    text: str
    font_size: float
    font_name: str
    is_bold: bool
    is_italic: bool
    x: float
    y: float
    width: float
    height: float
    page_num: int
    line_spacing: float

    def __post_init__(self):
        # Clean and normalize text
        self.text = self.normalize_text(self.text)

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for better processing"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        return text


class MultilingualFeatureExtractor:
    """Extract features from text elements for ML classification with multilingual support"""

    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=50,  # Reduced for size constraints
            ngram_range=(1, 1),  # Only unigrams for multilingual
            min_df=1,
            max_df=0.95,
            analyzer='char_wb',  # Character-based for better multilingual support
            lowercase=True
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def extract_text_features(self, element: TextElement, doc_stats: Dict) -> List[float]:
        """Extract text-based features with multilingual support"""
        text = element.text.lower().strip()

        # Basic text patterns (language-agnostic)
        features = [
            1 if re.search(r'\d+', text) else 0,  # has_number
            1 if re.match(r'^\d+', text.strip()) else 0,  # starts_with_number
            1 if ':' in text else 0,  # has_colon
            1 if '.' in text else 0,  # has_period
            sum(1 for c in element.text if c.isupper()) / max(len(element.text), 1),  # all_caps_ratio
            len(text.split()),  # word_count
            len(text),  # char_count
            element.y / doc_stats.get('page_height', 1),  # line_position
            1 if element.text and element.text[0].isupper() else 0,  # starts_with_capital
            1 if element.text and element.text[-1] in '.!?' else 0,  # ends_with_punctuation
            1 if '?' in text else 0,  # has_question_mark
            1 if '!' in text else 0,  # has_exclamation
        ]

        # Multilingual heading words detection
        heading_patterns = [
            r'\b(chapter|section|part|introduction|conclusion|summary|abstract|contents|table|figure|index|appendix)\b',
            # English
            r'\b(章|節|部|序論|結論|要約|抄録|目次|表|図|索引|付録)\b',  # Japanese
            r'\b(第|章|節|部|序|論|結|要|約|録|次|表|図|引|録)\b',  # Chinese/Japanese characters
            r'\b(capítulo|sección|parte|introducción|conclusión|resumen|contenido|tabla|figura|índice|apéndice)\b',
            # Spanish
            r'\b(chapitre|section|partie|introduction|conclusion|résumé|contenu|tableau|figure|index|annexe)\b',
            # French
            r'\b(kapitel|abschnitt|teil|einleitung|schluss|zusammenfassung|inhalt|tabelle|abbildung|index|anhang)\b',
            # German
        ]

        heading_score = 0
        for pattern in heading_patterns:
            if re.search(pattern, text):
                heading_score += 1

        features.append(heading_score)

        return features

    def extract_layout_features(self, element: TextElement, doc_stats: Dict) -> List[float]:
        """Extract layout-based features"""
        features = [
            element.font_size,
            element.font_size / doc_stats.get('avg_font_size', 12),  # font_size_ratio
            1 if element.is_bold else 0,  # is_bold
            1 if element.is_italic else 0,  # is_italic
            element.x / doc_stats.get('page_width', 612),  # x_position_normalized
            element.y / doc_stats.get('page_height', 792),  # y_position_normalized
            element.width / doc_stats.get('page_width', 612),  # width_normalized
            element.height / doc_stats.get('page_height', 792),  # height_normalized
            1 if element.font_size > doc_stats.get('avg_font_size', 12) * 1.2 else 0,  # is_large_font
            1 if abs(element.x - doc_stats.get('page_width', 612) / 2) < 100 else 0,  # is_centered
            1 if element.x < doc_stats.get('page_width', 612) * 0.3 else 0,  # is_left_aligned
        ]

        return features

    def extract_context_features(self, element: TextElement, elements: List[TextElement],
                                 index: int, doc_stats: Dict) -> List[float]:
        """Extract contextual features based on surrounding elements"""
        features = [0, 0, 0, 0, 0, 0]  # Initialize with zeros

        # Previous element features
        if index > 0:
            prev_elem = elements[index - 1]
            features[0] = prev_elem.font_size / doc_stats.get('avg_font_size', 12)
            features[1] = abs(element.y - prev_elem.y) / doc_stats.get('avg_line_spacing', 15)

        # Next element features
        if index < len(elements) - 1:
            next_elem = elements[index + 1]
            features[2] = next_elem.font_size / doc_stats.get('avg_font_size', 12)
            features[3] = abs(next_elem.y - element.y) / doc_stats.get('avg_line_spacing', 15)

        # Isolation check
        if index > 0 and index < len(elements) - 1:
            prev_spacing = abs(element.y - elements[index - 1].y)
            next_spacing = abs(elements[index + 1].y - element.y)
            avg_spacing = doc_stats.get('avg_line_spacing', 15)
            features[4] = 1 if min(prev_spacing, next_spacing) > avg_spacing * 1.5 else 0

        # Position in page
        same_page_elements = [e for e in elements if e.page_num == element.page_num]
        if same_page_elements:
            try:
                page_position = same_page_elements.index(element) / len(same_page_elements)
                features[5] = page_position
            except ValueError:
                features[5] = 0

        return features

    def extract_all_features(self, elements: List[TextElement]) -> np.ndarray:
        """Extract all features for a list of text elements"""
        if not elements:
            return np.array([])

        # Calculate document statistics
        doc_stats = self.calculate_document_stats(elements)

        # Extract features for each element
        all_features = []
        text_data = []

        for i, element in enumerate(elements):
            # Text features
            text_features = self.extract_text_features(element, doc_stats)

            # Layout features
            layout_features = self.extract_layout_features(element, doc_stats)

            # Context features
            context_features = self.extract_context_features(element, elements, i, doc_stats)

            # Combine all features
            combined_features = text_features + layout_features + context_features
            all_features.append(combined_features)
            text_data.append(element.text)

        # Convert to numpy array
        feature_matrix = np.array(all_features, dtype=np.float32)

        # TF-IDF features
        try:
            if not self.is_fitted:
                tfidf_features = self.tfidf_vectorizer.fit_transform(text_data).toarray()
                feature_matrix = self.scaler.fit_transform(feature_matrix)
                self.is_fitted = True
            else:
                tfidf_features = self.tfidf_vectorizer.transform(text_data).toarray()
                feature_matrix = self.scaler.transform(feature_matrix)

            # Combine numerical and TF-IDF features
            final_features = np.hstack([feature_matrix, tfidf_features])
        except Exception as e:
            logger.warning(f"Error in TF-IDF processing: {str(e)}")
            final_features = feature_matrix

        return final_features

    def calculate_document_stats(self, elements: List[TextElement]) -> Dict:
        """Calculate document-level statistics"""
        if not elements:
            return {}

        font_sizes = [e.font_size for e in elements]
        line_spacings = [e.line_spacing for e in elements if e.line_spacing > 0]

        stats = {
            'avg_font_size': np.mean(font_sizes),
            'median_font_size': np.median(font_sizes),
            'max_font_size': np.max(font_sizes),
            'min_font_size': np.min(font_sizes),
            'avg_line_spacing': np.mean(line_spacings) if line_spacings else 15,
            'page_width': max(e.x + e.width for e in elements) if elements else 612,
            'page_height': max(e.y + e.height for e in elements) if elements else 792,
        }

        return stats


class PDFProcessor:
    """Extract text elements from PDF with detailed formatting information"""

    def extract_text_elements(self, pdf_path: str) -> List[TextElement]:
        """Extract text elements from PDF with formatting information"""
        elements = []

        try:
            doc = fitz.open(pdf_path)

            for page_num in range(min(len(doc), 50)):  # Limit to 50 pages as per constraint
                page = doc[page_num]
                text_dict = page.get_text("dict")

                for block in text_dict["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if span["text"].strip():
                                    # Extract font information
                                    font_name = span["font"]
                                    font_size = span["size"]
                                    is_bold = "bold" in font_name.lower() or span["flags"] & 16
                                    is_italic = "italic" in font_name.lower() or span["flags"] & 2

                                    # Extract position and dimensions
                                    bbox = span["bbox"]
                                    x, y, x1, y1 = bbox
                                    width = x1 - x
                                    height = y1 - y

                                    # Calculate line spacing
                                    line_spacing = height if height > 0 else 12

                                    element = TextElement(
                                        text=span["text"],
                                        font_size=font_size,
                                        font_name=font_name,
                                        is_bold=is_bold,
                                        is_italic=is_italic,
                                        x=x,
                                        y=y,
                                        width=width,
                                        height=height,
                                        page_num=page_num + 1,
                                        line_spacing=line_spacing
                                    )
                                    elements.append(element)

            doc.close()

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return []

        return elements


class HeadingClassifier:
    """Lightweight machine learning model for classifying text elements as headings"""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=50,  # Reduced for size constraints
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        self.feature_extractor = MultilingualFeatureExtractor()
        self.is_trained = False

    def create_training_data(self, elements: List[TextElement]) -> Tuple[np.ndarray, np.ndarray]:
        """Create training data based on heuristics"""
        features = self.feature_extractor.extract_all_features(elements)

        if features.size == 0:
            return np.array([]), np.array([])

        # Create labels based on heuristics
        labels = []
        doc_stats = self.feature_extractor.calculate_document_stats(elements)

        for element in elements:
            label = self.classify_element_heuristic(element, doc_stats)
            labels.append(label)

        return features, np.array(labels)

    def classify_element_heuristic(self, element: TextElement, doc_stats: Dict) -> int:
        """Multilingual heuristic-based classification"""
        text = element.text.lower().strip()

        # Skip very short or very long text
        if len(text) < 2 or len(text) > 150:
            return 0  # Not a heading

        # Title detection (first page, large font)
        if (element.page_num == 1 and
                element.font_size > doc_stats.get('avg_font_size', 12) * 1.5 and
                len(text.split()) <= 15):
            return 4  # Title

        heading_score = 0

        # Font size score
        if element.font_size > doc_stats.get('avg_font_size', 12) * 1.3:
            heading_score += 3
        elif element.font_size > doc_stats.get('avg_font_size', 12) * 1.1:
            heading_score += 1

        # Bold text
        if element.is_bold:
            heading_score += 2

        # Positioning
        if element.x < doc_stats.get('page_width', 612) * 0.4:
            heading_score += 1

        # Multilingual patterns
        patterns = [
            r'^\d+\.?\s*',  # Starts with number
            r'^\w+\s*\d+',  # Word followed by number
            r'第\d+章',  # Japanese chapter
            r'第\d+節',  # Japanese section
            r'chapter\s+\d+',  # English chapter
            r'section\s+\d+',  # English section
            r'parte\s+\d+',  # Spanish part
            r'chapitre\s+\d+',  # French chapter
            r'kapitel\s+\d+',  # German chapter
        ]

        for pattern in patterns:
            if re.search(pattern, text):
                heading_score += 2
                break

        # Heading keywords (multilingual)
        heading_keywords = [
            'introduction', 'conclusion', 'summary', 'abstract', 'contents',
            '序論', '結論', '要約', '抄録', '目次',
            'introducción', 'conclusión', 'resumen', 'contenido',
            'introduction', 'conclusion', 'résumé', 'contenu',
            'einleitung', 'schluss', 'zusammenfassung', 'inhalt'
        ]

        if any(keyword in text for keyword in heading_keywords):
            heading_score += 2

        # Length consideration
        if 2 <= len(text.split()) <= 10:
            heading_score += 1

        # Classify based on score
        if heading_score >= 6:
            return 1  # H1
        elif heading_score >= 4:
            return 2  # H2
        elif heading_score >= 2:
            return 3  # H3
        else:
            return 0  # Not a heading

    def train(self, elements: List[TextElement]):
        """Train the classifier"""
        if not elements:
            return

        X, y = self.create_training_data(elements)

        if X.size == 0 or y.size == 0:
            return

        self.model.fit(X, y)
        self.is_trained = True

        logger.info(f"Trained classifier with {len(X)} samples")

    def predict(self, elements: List[TextElement]) -> List[int]:
        """Predict heading levels for text elements"""
        if not elements:
            return []

        if not self.is_trained:
            self.train(elements)

        if not self.is_trained:
            # Fallback to heuristic if training failed
            doc_stats = self.feature_extractor.calculate_document_stats(elements)
            return [self.classify_element_heuristic(elem, doc_stats) for elem in elements]

        features = self.feature_extractor.extract_all_features(elements)

        if features.size == 0:
            return [0] * len(elements)

        predictions = self.model.predict(features)
        return predictions.tolist()


class PDFOutlineExtractor:
    """Main class for extracting PDF outlines"""

    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.classifier = HeadingClassifier()

    def extract_outline(self, pdf_path: str) -> Dict:
        """Extract outline from PDF file"""
        logger.info(f"Processing PDF: {pdf_path}")

        start_time = time.time()

        # Extract text elements
        elements = self.pdf_processor.extract_text_elements(pdf_path)

        if not elements:
            logger.warning(f"No text elements found in {pdf_path}")
            return {"title": "", "outline": []}

        logger.info(f"Extracted {len(elements)} text elements")

        # Classify elements
        predictions = self.classifier.predict(elements)

        # Extract title and outline
        title = self.extract_title(elements, predictions)
        outline = self.extract_headings(elements, predictions)

        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        logger.info(f"Extracted title: {title}")
        logger.info(f"Extracted {len(outline)} headings")

        return {
            "title": title,
            "outline": outline
        }

    def extract_title(self, elements: List[TextElement], predictions: List[int]) -> str:
        """Extract document title with multilingual support"""
        # Look for title prediction
        for element, pred in zip(elements, predictions):
            if pred == 4:
                return element.text.strip()

        # Fallback: find largest font on first few pages
        first_pages_elements = [e for e in elements if e.page_num <= 2]
        if first_pages_elements:
            # Sort by font size and position
            title_candidates = sorted(first_pages_elements,
                                      key=lambda x: (x.font_size, -x.y),
                                      reverse=True)

            for candidate in title_candidates[:3]:  # Check top 3 candidates
                if (candidate.font_size > 14 and
                        3 <= len(candidate.text.split()) <= 20 and
                        not re.match(r'^\d+\.?\s*$', candidate.text.strip())):
                    return candidate.text.strip()

        return ""

    def extract_headings(self, elements: List[TextElement], predictions: List[int]) -> List[Dict]:
        """Extract headings from predictions"""
        headings = []
        level_map = {1: "H1", 2: "H2", 3: "H3"}

        for element, pred in zip(elements, predictions):
            if pred in level_map:
                # Clean heading text
                heading_text = element.text.strip()
                # Remove excessive whitespace
                heading_text = re.sub(r'\s+', ' ', heading_text)

                if heading_text:  # Only add non-empty headings
                    heading = {
                        "level": level_map[pred],
                        "text": heading_text,
                        "page": element.page_num
                    }
                    headings.append(heading)

        return headings

    def process_directory(self, input_dir: str, output_dir: str):
        """Process all PDF files in input directory"""
        if not os.path.exists(input_dir):
            logger.error(f"Input directory not found: {input_dir}")
            return

        os.makedirs(output_dir, exist_ok=True)

        pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]

        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        for pdf_file in pdf_files:
            try:
                pdf_path = os.path.join(input_dir, pdf_file)
                output_file = os.path.splitext(pdf_file)[0] + '.json'
                output_path = os.path.join(output_dir, output_file)

                # Extract outline
                outline = self.extract_outline(pdf_path)

                # Save to JSON
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(outline, f, indent=2, ensure_ascii=False)

                logger.info(f"Processed {pdf_file} -> {output_file}")

            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")


def main():
    """Main function for Docker container execution"""
    input_dir = "input"
    output_dir = "output"

    logger.info("Starting multilingual PDF outline extraction")

    extractor = PDFOutlineExtractor()
    extractor.process_directory(input_dir, output_dir)

    logger.info("PDF outline extraction completed")


if __name__ == "__main__":
    main()