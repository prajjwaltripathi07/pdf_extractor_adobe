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
import csv
from collections import defaultdict
import pickle
import glob
import argparse
import ast

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    page_num: int  # 0-indexed page number
    line_spacing: float
    bbox: Tuple[float, float, float, float]

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
            max_features=200,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            analyzer='char_wb',
            lowercase=True
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def get_feature_names(self) -> List[str]:
        """Returns the names of all features in the correct order."""
        text_features = [
            'has_number', 'starts_with_number', 'has_colon', 'has_period', 'all_caps_ratio',
            'word_count', 'char_count', 'starts_with_capital', 'ends_with_punctuation',
            'has_question_mark', 'has_exclamation', 'is_all_caps_short',
            'starts_with_capital_then_lowercase', 'starts_with_bullet',
            'starts_with_list_item', 'contains_fig_table_keyword', 'heading_keyword_score'
        ]
        layout_features = [
            'font_size', 'font_size_ratio_avg', 'font_size_ratio_median', 'is_bold',
            'is_italic', 'x_position_normalized', 'y_position_normalized',
            'width_normalized', 'height_normalized', 'is_significantly_larger_font_median',
            'is_significantly_larger_font_avg', 'is_roughly_centered',
            'is_far_left_aligned', 'is_far_right_aligned', 'relative_x_to_main_text',
            'is_out_left_of_main_text_block', 'is_indented_from_main_text',
            'normalized_line_spacing', 'absolute_width', 'absolute_height',
            'absolute_x', 'absolute_y', 'font_size_ratio_max'
        ]
        context_features = [
            'prev_font_size_ratio_median', 'spacing_before_normalized', 'is_prev_bold',
            'is_same_font_size_as_prev', 'prev_ends_with_period', 'is_indented_from_prev',
            'large_vertical_gap_before', 'prev_ends_with_colon',
            'next_font_size_ratio_median', 'spacing_after_normalized', 'is_next_bold',
            'next_indented_from_current', 'large_vertical_gap_after',
            'next_starts_with_list_char', 'is_isolated_by_spacing',
            'page_position_in_elements', 'spans_wide', 'char_count_ratio_avg'
        ]
        
        tfidf_feature_names = []
        if self.is_fitted:
            try:
                # Use get_feature_names_out() which is standard, fallback for safety
                tfidf_features = self.tfidf_vectorizer.get_feature_names_out()
                tfidf_feature_names = [f'tfidf_{name}' for name in tfidf_features]
            except AttributeError:
                 # Fallback for older sklearn versions
                num_tfidf_features = len(self.tfidf_vectorizer.vocabulary_)
                tfidf_feature_names = [f'tfidf_{i}' for i in range(num_tfidf_features)]

        return text_features + layout_features + context_features + tfidf_feature_names

    def extract_text_features(self, element: TextElement, doc_stats: Dict) -> List[float]:
        """Extract text-based features with multilingual support"""
        text = element.text.lower().strip()

        features = [
            1 if re.search(r'\d+', text) else 0,  # has_number
            1 if re.match(r'^\d[\d\.\s]*', text.strip()) else 0,  # starts_with_number
            1 if ':' in text else 0,  # has_colon
            1 if '.' in text else 0,  # has_period (note: might be part of abbreviations)
            sum(1 for c in element.text if c.isupper()) / max(len(element.text), 1),  # all_caps_ratio
            len(text.split()),  # word_count
            len(text),  # char_count
            1 if element.text and element.text[0].isupper() else 0,  # starts_with_capital
            1 if element.text and element.text[-1] in '.!?' else 0,  # ends_with_punctuation
            1 if '?' in text else 0,  # has_question_mark
            1 if '!' in text else 0,  # has_exclamation
            1 if text.isupper() and len(text.split()) < 10 else 0,  # is_all_caps_short (more likely a heading)
            1 if re.match(r'^[A-Z][a-z]', element.text) else 0,  # Starts with Capital then lowercase
            1 if re.search(r'^[\-\*•·]\s*', text) else 0,  # Starts with a common bullet point
            1 if re.search(r'^\(\w+\)|\w+\.', text) else 0,  # Starts with (a) or a. (list item)
            1 if re.search(r'\b(fig\.|figure|table|graph|chart)\b', text) else 0  # Contains figure/table keywords (demotion)
        ]

        heading_patterns = [
            r'\b(chapter|section|part|introduction|conclusion|summary|abstract|contents|table of contents|index|appendix|acknowledgements|preface|bibliography|references)\b',
            r'\b(章|節|部|序論|結論|要約|抄録|目次|表|図|索引|付録|謝辞|まえがき|参考文献)\b',
            r'\b(第|章|節|部|序|論|結|要|約|録|次|表|図|引|録)\b',
            r'\b(capítulo|sección|parte|introducción|conclusión|resumen|contenido|índice|apéndice|agradecimientos|prólogo|bibliografía|referencias)\b',
            r'\b(chapitre|section|partie|introduction|conclusion|résumé|contenu|index|annexe|remerciements|préface|bibliographie|références)\b',
            r'\b(kapitel|abschnitt|teil|einleitung|schluss|zusammenfassung|inhalt|index|anhang|danksagung|vorwort|literaturverzeichnis)\b',
            r'\b(introduzione|conclusione|sommario|indice|appendice|ringraziamenti|prefazione|bibliografia)\b',
            r'\b(introdução|conclusão|resumo|índice|apêndice|agradecimentos|prefácio|bibliografia|referências)\b',
            r'\b(methodology|results|discussion|analysis|experiment|design|implementation|future work)\b'  # Technical terms
        ]

        heading_keyword_score = 0
        for pattern in heading_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                heading_keyword_score += 1
        features.append(heading_keyword_score)

        return features

    def extract_layout_features(self, element: TextElement, doc_stats: Dict) -> List[float]:
        """Extract layout-based features, enhanced for robustness."""
        page_width = doc_stats.get('page_width', 612)
        page_height = doc_stats.get('page_height', 792)
        avg_font_size = doc_stats.get('avg_font_size', 12)
        median_font_size = doc_stats.get('median_font_size', 12)
        max_font_size = doc_stats.get('max_font_size', 12)
        main_text_x_start = doc_stats.get('main_text_x_start', page_width * 0.1)
        main_text_x_end = doc_stats.get('main_text_x_end', page_width * 0.9)

        features = [
            element.font_size,
            element.font_size / avg_font_size,  # font_size_ratio_avg
            element.font_size / median_font_size,  # font_size_ratio_median
            1 if element.is_bold else 0,  # is_bold
            1 if element.is_italic else 0,  # is_italic
            element.x / page_width,  # x_position_normalized
            element.y / page_height,  # y_position_normalized
            element.width / page_width,  # width_normalized
            element.height / page_height,  # height_normalized
            # Font size relative to document's largest and median font
            1 if element.font_size > median_font_size * 1.4 else 0,  # is_significantly_larger_font_median
            1 if element.font_size > avg_font_size * 1.4 else 0,  # is_significantly_larger_font_avg
            
            # More nuanced centering / alignment
            1 if (element.x > page_width * 0.2 and element.x + element.width < page_width * 0.8 and
                  abs((element.x + element.width / 2) - page_width / 2) < page_width * 0.1) else 0, # is_roughly_centered
            1 if element.x < page_width * 0.15 else 0,  # is_far_left_aligned (adjusted margin)
            1 if element.x + element.width > page_width * 0.85 else 0, # is_far_right_aligned

            # X-position relative to the main text block
            (element.x - main_text_x_start) / max(1, (main_text_x_end - main_text_x_start)), # relative_x_to_main_text
            1 if element.x < main_text_x_start - 5 else 0, # is_out_left_of_main_text_block
            1 if element.x > main_text_x_start + 5 and element.x < main_text_x_end * 0.5 else 0, # is_indented_from_main_text
            
            element.line_spacing / doc_stats.get('avg_line_spacing', 15) if doc_stats.get('avg_line_spacing', 15) > 0 else 0, # normalized line spacing
            element.width,  # Absolute width
            element.height, # Absolute height
            element.x, # Absolute X position
            element.y, # Absolute Y position
            element.font_size / max_font_size # Font size relative to document's largest font
        ]
        return features

    def extract_context_features(self, element: TextElement, elements: List[TextElement],
                                 index: int, doc_stats: Dict) -> List[float]:
        """Extract contextual features based on surrounding elements"""
        features = [0] * 18  # Increased size for more context features

        avg_spacing = doc_stats.get('avg_line_spacing', 15)
        median_font_size = doc_stats.get('median_font_size', 12)

        # Features related to previous element (only if on the same page)
        if index > 0 and elements[index - 1].page_num == element.page_num:
            prev_elem = elements[index - 1]
            features[0] = prev_elem.font_size / median_font_size  # prev_font_size_ratio_median
            features[1] = abs(element.y - prev_elem.y) / avg_spacing  # spacing_before_normalized
            features[2] = 1 if prev_elem.is_bold else 0  # is_prev_bold
            features[3] = 1 if prev_elem.font_size == element.font_size else 0  # is_same_font_size_as_prev
            features[4] = 1 if prev_elem.text.lower().strip().endswith('.') else 0  # prev_ends_with_period (indicates body text)
            features[5] = 1 if prev_elem.x < element.x - (element.width * 0.1) else 0  # is_indented_from_prev (more robust threshold)
            features[6] = 1 if (element.y - prev_elem.y) > (median_font_size * 2) else 0 # large_vertical_gap_before
            features[7] = 1 if prev_elem.text.lower().strip().endswith(':') else 0 # prev_ends_with_colon

        # Features related to next element (only if on the same page)
        if index < len(elements) - 1 and elements[index + 1].page_num == element.page_num:
            next_elem = elements[index + 1]
            features[8] = next_elem.font_size / median_font_size  # next_font_size_ratio_median
            features[9] = abs(next_elem.y - element.y) / avg_spacing  # spacing_after_normalized
            features[10] = 1 if next_elem.is_bold else 0  # is_next_bold
            features[11] = 1 if next_elem.x > element.x + (element.width * 0.1) else 0  # next_indented_from_current
            features[12] = 1 if (next_elem.y - element.y) > (median_font_size * 2) else 0 # large_vertical_gap_after
            features[13] = 1 if next_elem.text.lower().strip().startswith((':', '-', '.', '*', '(', '[')) else 0 # next_starts_with_list_char

        # Isolation check: significant space before AND after (on same page)
        if index > 0 and index < len(elements) - 1 and \
           elements[index - 1].page_num == element.page_num and \
           elements[index + 1].page_num == element.page_num:
            prev_spacing = abs(element.y - elements[index - 1].y)
            next_spacing = abs(elements[index + 1].y - element.y)
            # Isolated by large spacing (relative to average line spacing and font size)
            features[14] = 1 if (prev_spacing > avg_spacing * 2.5 and next_spacing > avg_spacing * 2.5 and
                                  prev_spacing > median_font_size * 1.5 and next_spacing > median_font_size * 1.5) else 0

        # Vertical position on page (normalized)
        same_page_elements = [e for e in elements if e.page_num == element.page_num]
        if same_page_elements:
            try:
                # Use total number of elements, or more robustly, index within its page group
                page_position_in_elements = same_page_elements.index(element) / max(len(same_page_elements) - 1, 1)
                features[15] = page_position_in_elements
            except ValueError:
                features[15] = 0

        # Does it span a significant portion of the page width?
        features[16] = 1 if element.width > doc_stats.get('page_width', 612) * 0.7 else 0 # spans_wide
        features[17] = len(element.text) / doc_stats.get('avg_line_char_count', 50) # char_count_ratio_avg

        return features

    def extract_all_features(self, elements: List[TextElement]) -> np.ndarray:
        """Extract all features for a list of text elements"""
        if not elements:
            return np.array([])

        doc_stats = self.calculate_document_stats(elements)
        all_features = []
        text_data = []

        for i, element in enumerate(elements):
            combined_features = (
                self.extract_text_features(element, doc_stats) +
                self.extract_layout_features(element, doc_stats) +
                self.extract_context_features(element, elements, i, doc_stats)
            )
            all_features.append(combined_features)
            text_data.append(element.text)

        feature_matrix = np.array(all_features, dtype=np.float32)

        try:
            # Ensure TF-IDF vectorizer and scaler are fitted only once on the training set
            if not self.is_fitted:
                tfidf_features = self.tfidf_vectorizer.fit_transform(text_data).toarray()
                feature_matrix = self.scaler.fit_transform(feature_matrix)
                self.is_fitted = True
            else:
                tfidf_features = self.tfidf_vectorizer.transform(text_data).toarray()
                # Ensure feature matrix has correct dimensions before transforming
                if feature_matrix.shape[0] > 0 and feature_matrix.shape[1] == self.scaler.n_features_in_:
                    feature_matrix = self.scaler.transform(feature_matrix)
                elif feature_matrix.shape[0] > 0:
                    logger.warning(f"Feature matrix column count ({feature_matrix.shape[1]}) mismatch with fitted scaler ({self.scaler.n_features_in_}). Skipping scaling for this prediction.")
                else:
                    logger.warning("Feature matrix is empty during scaling. Skipping scaling.")

            final_features = np.hstack([feature_matrix, tfidf_features])
        except Exception as e:
            logger.warning(f"Error in TF-IDF or scaling processing, falling back to numerical features only: {e}")
            final_features = feature_matrix

        return final_features

    def calculate_document_stats(self, elements: List[TextElement]) -> Dict:
        """Calculate document-level statistics, including main text block approximations."""
        if not elements:
            return {}

        font_sizes = [e.font_size for e in elements]
        line_spacings = [e.line_spacing for e in elements if e.line_spacing > 0]
        char_counts = [len(e.text) for e in elements]

        # Use robust statistics for page dimensions
        page_widths = [e.x + e.width for e in elements]
        page_heights = [e.y + e.height for e in elements]

        # Approximate main text block X coordinates
        # Filter out very short lines or lines that are too far left/right
        relevant_x_coords = [e.x for e in elements if len(e.text.strip()) > 10 and e.width > 0.1 * max(page_widths, default=612)]
        if relevant_x_coords:
            # Using percentiles to get typical left and right bounds of text
            x_start = np.percentile(relevant_x_coords, 5)
            x_end_points = [e.x + e.width for e in elements if len(e.text.strip()) > 10 and e.width > 0.1 * max(page_widths, default=612)]
            x_end = np.percentile(x_end_points, 95)
        else:
            x_start = max(page_widths, default=612) * 0.1
            x_end = max(page_widths, default=612) * 0.9

        stats = {
            'avg_font_size': np.mean(font_sizes) if font_sizes else 12,
            'median_font_size': np.median(font_sizes) if font_sizes else 12,
            'max_font_size': np.max(font_sizes) if font_sizes else 12,
            'min_font_size': np.min(font_sizes) if font_sizes else 12,
            'avg_line_spacing': np.mean(line_spacings) if line_spacings else 15,
            'avg_line_char_count': np.mean(char_counts) if char_counts else 50,
            'page_width': max(page_widths) if page_widths else 612,
            'page_height': max(page_heights) if page_heights else 792,
            'main_text_x_start': x_start,
            'main_text_x_end': x_end,
        }
        return stats


class PDFProcessor:
    """Extract text elements from PDF with detailed formatting information"""

    def extract_text_elements(self, pdf_path: str) -> List[TextElement]:
        """Extract text elements from PDF with formatting information"""
        elements = []
        try:
            doc = fitz.open(pdf_path)

            for page_num in range(min(len(doc), 50)): # Limit to first 50 pages for performance
                page = doc[page_num]
                text_dict = page.get_text("dict")

                spans_on_page = []
                for block in text_dict["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if span["text"].strip():
                                    spans_on_page.append({
                                        "text": span["text"],
                                        "font": span["font"],
                                        "size": span["size"],
                                        "flags": span["flags"],
                                        "bbox": span["bbox"],
                                    })

                # Sort spans by y-coordinate (top to bottom) then x-coordinate (left to right)
                spans_on_page.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))

                previous_span_y_bottom = None
                previous_span_page = -1 # Track page for spacing reset

                for span_data in spans_on_page:
                    font_name = span_data["font"]
                    font_size = span_data["size"]
                    # PyMuPDF flags: 16 is bold, 2 is italic
                    is_bold = "bold" in font_name.lower() or bool(span_data["flags"] & 16)
                    is_italic = "italic" in font_name.lower() or bool(span_data["flags"] & 2)

                    bbox = span_data["bbox"]
                    x, y, x1, y1 = bbox
                    width = x1 - x
                    height = y1 - y

                    current_line_spacing = 0.0
                    # Calculate spacing from bottom of previous span to top of current span
                    # Only if on the same page and it's not the very first span on a page
                    if previous_span_y_bottom is not None and page_num == previous_span_page:
                        current_line_spacing = y - previous_span_y_bottom

                    element = TextElement(
                        text=span_data["text"],
                        font_size=font_size,
                        font_name=font_name,
                        is_bold=is_bold,
                        is_italic=is_italic,
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                        page_num=page_num,
                        line_spacing=max(0.0, current_line_spacing), # Ensure non-negative spacing
                        bbox=bbox
                    )
                    elements.append(element)
                    previous_span_y_bottom = y1 # Update for the next span
                    previous_span_page = page_num

            doc.close()
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}", exc_info=True)
            return []
        return elements


class HeadingClassifier:
    """Machine learning model for classifying text elements as headings"""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        self.feature_extractor = MultilingualFeatureExtractor()
        self.is_trained = False

    def save(self, model_dir: str):
        """Saves the trained model and its components."""
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "heading_classifier.pkl")
        try:
            with open(model_path, "wb") as f:
                pickle.dump({
                    'model': self.model,
                    'vectorizer': self.feature_extractor.tfidf_vectorizer,
                    'scaler': self.feature_extractor.scaler
                }, f)
            logger.info(f"Model successfully saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)

    def load(self, model_dir: str) -> bool:
        """Loads a pre-trained model and its components."""
        model_path = os.path.join(model_dir, "heading_classifier.pkl")
        if not os.path.exists(model_path):
            logger.warning(f"No pre-trained model found at {model_path}.")
            return False
        
        try:
            with open(model_path, "rb") as f:
                components = pickle.load(f)
            self.model = components['model']
            self.feature_extractor.tfidf_vectorizer = components['vectorizer']
            self.feature_extractor.scaler = components['scaler']
            self.is_trained = True
            self.feature_extractor.is_fitted = True
            logger.info(f"Successfully loaded pre-trained model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model, will retrain if necessary: {e}", exc_info=True)
            return False

    def _load_elements_from_csv(self, file_path: str) -> Tuple[List[TextElement], List[int]]:
        """Loads TextElements and labels from a single CSV file."""
        elements = []
        labels = []
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        element = TextElement(
                            text=row['text'],
                            font_size=float(row['font_size']),
                            font_name=row['font_name'],
                            is_bold=row['is_bold'] == 'True',
                            is_italic=row['is_italic'] == 'True',
                            x=float(row['x']),
                            y=float(row['y']),
                            width=float(row['width']),
                            height=float(row['height']),
                            page_num=int(row['page_num']),
                            line_spacing=float(row['line_spacing']),
                            bbox=ast.literal_eval(row['bbox']) # Safely evaluate string to tuple
                        )
                        elements.append(element)
                        labels.append(int(row['predicted_label'])) # Or 'true_label'
                    except (ValueError, KeyError, SyntaxError) as e:
                        logger.warning(f"Skipping malformed row in {file_path}: {row}. Error: {e}")
        except Exception as e:
            logger.error(f"Failed to read or process CSV file {file_path}: {e}", exc_info=True)
        return elements, labels

    def train_from_csv_folder(self, folder_path: str):
        """Train the classifier using all CSV files in a given folder."""
        logger.info(f"Starting training from CSV files in '{folder_path}'...")
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        if not csv_files:
            logger.error(f"No CSV files found in '{folder_path}'. Training aborted.")
            return

        all_elements = []
        all_labels = []
        for csv_file in csv_files:
            logger.info(f"Loading data from {csv_file}")
            elements, labels = self._load_elements_from_csv(csv_file)
            all_elements.extend(elements)
            all_labels.extend(labels)

        if not all_elements:
            logger.error("No valid data loaded from CSV files. Training aborted.")
            return

        # Use the combined elements to generate features and train
        self.train(all_elements, pre_labeled_data=(np.array(all_labels),))
    
    def create_training_data(self, elements: List[TextElement]) -> Tuple[np.ndarray, np.ndarray]:
        """Create training data based on heuristics"""
        features = self.feature_extractor.extract_all_features(elements)

        if features.size == 0:
            logger.warning("No features extracted for training. Returning empty arrays.")
            return np.array([]), np.array([])

        labels = []
        doc_stats = self.feature_extractor.calculate_document_stats(elements)

        for element in elements:
            label = self.classify_element_heuristic(element, doc_stats)
            labels.append(label)

        return features, np.array(labels)

    def classify_element_heuristic(self, element: TextElement, doc_stats: Dict) -> int:
        """
        Multilingual heuristic-based classification for training data generation and fallback.
        Enhanced for more accurate title detection and table handling, less reliance on strict centering.
        """
        text = element.text.lower().strip()
        
        page_width = doc_stats.get('page_width', 612)
        page_height = doc_stats.get('page_height', 792)
        median_font_size = doc_stats.get('median_font_size', 12)
        max_font_size = doc_stats.get('max_font_size', 12)
        avg_line_spacing = doc_stats.get('avg_line_spacing', 15)
        main_text_x_start = doc_stats.get('main_text_x_start', page_width * 0.1)

        # Rule 1: Filter out obviously non-heading text
        if len(text) < 3 or len(text) > 300: # Increased max length for potential long titles
            return 0
        if len(re.findall(r'[a-zA-Z0-9]', text)) / max(1, len(text)) < 0.3: # Filter mostly symbols/spaces
            return 0
        if re.match(r'^\s*\d+\s*$', text) or re.match(r'^-?\s*\d+\s*-$', text): # Just a number (likely page number) or range
            return 0
        if re.search(r'\b(www\.|http:|https:|ftp\.)', text): # URLs
            return 0
        if re.match(r'^[A-Za-z]{1,2}$', text): # Single or double letter (often just initials or labels)
            return 0

        # Rule 2: Title detection (highest priority, label 4)
        is_title_candidate_heuristic = (
            element.page_num == 0 and
            element.font_size >= max_font_size * 0.9 and # Must be very close to max font
            element.y < page_height*0.5  and # ANYWHERE ON THE PAGE
            element.width < page_width * 0.8 # Title should span a reasonable width
        )
        # More flexible centering check for titles
        is_title_centered_or_wide = (
            abs((element.x + element.width / 2) - page_width / 2) < page_width * 0.1 or # Roughly centered
            element.width > page_width * 0.7 # Or spans most of the width
        )

        if is_title_candidate_heuristic and is_title_centered_or_wide:
            # Add conditions to avoid abstracts/introductions for very long texts
            if len(text.split()) > 60: # If it's very long, check for abstract/intro keywords more strictly
                if re.search(r'\b(abstract|introduction|summary|preface|foreword|序論|抄録|resumen|table of contents|目次)\b', text):
                    return 0 # Likely not the main title if it's that long and contains these
                if text.count('.') > 2: # Very long text with multiple sentences is probably body/abstract
                    return 0
            if text.endswith('.'): # Titles rarely end with periods
                return 0
            return 4 # Title

        heading_score = 0

        # Feature Group 1: Font and Style (relative to median font size)
        if element.font_size >= median_font_size * 1.8: # Significantly larger (H1 candidate)
            heading_score += 4
        elif element.font_size >= median_font_size * 1.4: # Larger (H2 candidate)
            heading_score += 3
        elif element.font_size >= median_font_size * 1.15: # Slightly larger (H3 candidate)
            heading_score += 1.5

        if element.is_bold:
            heading_score += 3.5 # Strong indicator

        if element.is_italic and not element.is_bold: # Italic alone is weaker
            heading_score += 0.5

        if text.isupper() and len(text.split()) < 20: # ALL CAPS for shorter headings
            heading_score += 1.5

        # Feature Group 2: Positioning and Spacing (relative to document stats)
        # Check alignment relative to main text block start
        if element.x < main_text_x_start + (page_width * 0.02): # Very close to typical left margin
            heading_score += 2
        elif element.x < main_text_x_start + (page_width * 0.08): # Slightly indented
            heading_score += 1

        # Significant space before (relative to average line spacing and median font size)
        # Using element.line_spacing from TextElement (calculated as distance from previous element)
        if element.line_spacing > avg_line_spacing * 2.5 and element.line_spacing > median_font_size * 1.5:
            heading_score += 2.5
        elif element.line_spacing > avg_line_spacing * 1.5 and element.line_spacing > median_font_size * 0.8:
            heading_score += 1.5

        # Check if centered horizontally relative to page or main text block
        if abs((element.x + element.width / 2) - page_width / 2) < page_width * 0.2: # Roughly centered
            heading_score += 1
        
        # Heading usually don't end with a period unless it's a short numbered entry
        if text.endswith('.') and not re.match(r'^\d+(\.\d+)*(\s|$)', text):
             heading_score -= 1 # Penalty for ending with a period if not a number sequence

        # Feature Group 3: Text Content Patterns (Multilingual & Table Exclusion)
        # Numbering patterns (e.g., 1, 1.1, A, A.1, Roman Numerals)
        if re.match(r'^\s*(\d+(\.\d+)*|[IVXLCDM]+\.?|[A-Z]\.?)\s+', text):
            heading_score += 3 # Very strong indicator if numbered

        # Common heading keywords (re-emphasize here for heuristic)
        heading_keywords_strong = [
            'chapter', 'section', 'part', 'introduction', 'conclusion', 'summary', 'contents',
            '目次', '序論', '結論', '要約', '参考文献',
            'methodology', 'results', 'discussion', 'analysis'
        ]
        if any(keyword in text for keyword in heading_keywords_strong):
            heading_score += 2.5

        # Demotion for Table/Figure Captions or Footer-like text
        if re.search(r'\b(fig\.|figure|table|graph|chart|appendix)\s+\d+(\.\d+)*', text, re.IGNORECASE) or \
           re.search(r'\b(source|copyright|page)\b', text, re.IGNORECASE) or \
           re.search(r'^\s*(p\.\s*\d+|page\s*\d+)\s*$', text, re.IGNORECASE) or \
           re.search(r'\(cid:\d+\)', text) or \
           (len(text.split()) < 5 and text.isdigit()) : # Short numeric likely page/figure number
            heading_score -= 5 # Heavy penalty for common table/figure/footer indicators

        # Length consideration: headings are usually concise, but allowing for slightly longer titles
        word_count = len(text.split())
        if word_count > 30 and not (element.font_size >= max_font_size * 0.8 and element.is_bold):
            heading_score -= 2 # Penalize very long text unless it's very prominent visually

        # Classification based on refined score thresholds
        if heading_score >= 6:
            return 1  # H1
        elif heading_score >= 5 and heading_score < 6:
            return 2  # H2
        elif heading_score >= 3 and heading_score < 5:
            return 3  # H3
        else:
            return 0  # Not a heading

    def train(self, elements: List[TextElement], pre_labeled_data: Optional[Tuple] = None):
        """Train the classifier using either heuristics or pre-labeled data."""
        if not elements:
            logger.warning("No elements provided for training. Skipping training.")
            return

        if pre_labeled_data:
            logger.info("Training with pre-labeled data...")
            y = pre_labeled_data[0]
            # When training from CSV, feature extractor is fitted on the entire dataset
            X = self.feature_extractor.extract_all_features(elements)
        else:
            logger.info("Generating labels with heuristics for self-supervised training...")
            X, y = self.create_training_data(elements)

        if X.size == 0 or y.size == 0 or len(np.unique(y)) < 2:
            logger.warning("Not enough valid training data generated or only one class present. Skipping classifier training.")
            self.is_trained = False
            return

        try:
            self.model.fit(X, y)
            self.is_trained = True
            logger.info(f"Trained classifier with {len(X)} samples. Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        except Exception as e:
            logger.error(f"Error during classifier training: {e}", exc_info=True)
            self.is_trained = False


    def predict(self, elements: List[TextElement], features: Optional[np.ndarray] = None) -> List[int]:
        """Predict heading levels for text elements. Can accept pre-computed features."""
        if not elements:
            return []

        if not self.is_trained:
            logger.info("Classifier not trained. Attempting self-supervised training with provided elements.")
            self.train(elements)

        if not self.is_trained:
            logger.warning("Classifier training failed or not possible. Falling back to heuristic prediction.")
            doc_stats = self.feature_extractor.calculate_document_stats(elements)
            return [self.classify_element_heuristic(elem, doc_stats) for elem in elements]

        # If features are not provided, extract them.
        if features is None:
            features = self.feature_extractor.extract_all_features(elements)

        if features.size == 0:
            logger.warning("No features extracted for prediction. Returning all zeros.")
            return [0] * len(elements)

        try:
            predictions = self.model.predict(features)
            return predictions.tolist()
        except Exception as e:
            logger.error(f"Error during prediction, falling back to heuristic: {e}", exc_info=True)
            doc_stats = self.feature_extractor.calculate_document_stats(elements)
            return [self.classify_element_heuristic(elem, doc_stats) for elem in elements]


class PDFOutlineExtractor:
    """Main class for extracting PDF outlines"""

    def __init__(self, model_dir: str = 'model'):
        self.pdf_processor = PDFProcessor()
        self.classifier = HeadingClassifier()
        self.model_dir = model_dir
        # Attempt to load a pre-trained model on initialization
        self.classifier.load(self.model_dir)

    def _save_dataset_as_csv(self, file_path: str, elements: List[TextElement], predictions: List[int]):
        """Saves the extracted element data and predictions to a CSV file."""
        try:
            header = [
                'text', 'predicted_label', 'page_num', 'font_size', 'font_name', 
                'is_bold', 'is_italic', 'x', 'y', 'width', 'height', 'line_spacing', 'bbox'
            ]

            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                
                for i, element in enumerate(elements):
                    row = [
                        element.text,
                        predictions[i],
                        element.page_num,
                        element.font_size,
                        element.font_name,
                        element.is_bold,
                        element.is_italic,
                        element.x,
                        element.y,
                        element.width,
                        element.height,
                        element.line_spacing,
                        str(element.bbox) # Convert tuple to string for CSV compatibility
                    ]
                    writer.writerow(row)
            
            logger.info(f"Successfully generated dataset at: {file_path}")

        except Exception as e:
            logger.error(f"Failed to generate CSV dataset for {file_path}: {e}", exc_info=True)

    def extract_outline(self, pdf_path: str, csv_output_path: Optional[str] = None) -> Dict:
        """Extract outline from PDF file and optionally save the feature dataset."""
        logger.info(f"Processing PDF: {pdf_path}")
        start_time = time.time()

        elements = self.pdf_processor.extract_text_elements(pdf_path)
        if not elements:
            logger.warning(f"No text elements found in {pdf_path}. Returning empty outline.")
            return {"title": "", "outline": []}

        logger.info(f"Extracted {len(elements)} text elements from {pdf_path}.")

        predictions = self.classifier.predict(elements)

        if csv_output_path:
            self._save_dataset_as_csv(csv_output_path, elements, predictions)

        title = self.extract_title(elements, predictions)
        outline = self.post_process_headings(elements, predictions)

        processing_time = time.time() - start_time
        logger.info(f"Finished processing {pdf_path} in {processing_time:.2f} seconds.")
        logger.info(f"Extracted title: '{title}'.")
        logger.info(f"Extracted {len(outline)} hierarchical headings.")

        return {
            "title": title,
            "outline": outline
        }

    def train_and_save_model(self, assets_dir: str):
        """High-level function to train from CSVs and save the model."""
        self.classifier.train_from_csv_folder(assets_dir)
        if self.classifier.is_trained:
            self.classifier.save(self.model_dir)
        else:
            logger.error("Training failed. Model was not saved.")


    def extract_title(self, elements: List[TextElement], predictions: List[int]) -> str:
        """Extract document title with multilingual support. Enhanced for higher accuracy."""
        doc_stats = self.classifier.feature_extractor.calculate_document_stats(elements)
        page_width = doc_stats.get('page_width', 612)
        page_height = doc_stats.get('page_height', 792)
        max_font_size = doc_stats.get('max_font_size', 12)

        title_candidates = []

        # Combine model predictions (label 4) and strong heuristic candidates
        for i, (element, pred) in enumerate(zip(elements, predictions)):
            text = element.text.lower().strip()

            # Criteria for a strong title candidate
            is_strong_title_candidate = (
                element.page_num == 0 and
                element.font_size >= max_font_size * 0.85 and # Very large font (closest to max)
                element.y < page_height * 0.45 and # In upper 45% of the first page
                3 <= len(element.text.split()) <= 60 and # Reasonable length for a title
                element.width > page_width * 0.3 # Title should span a reasonable width
            )

            is_centered_or_wide = (
                abs((element.x + element.width / 2) - page_width / 2) < page_width * 0.25 or # Roughly centered
                element.width > page_width * 0.7 # Or spans most of the width
            )

            # Avoid common non-title patterns in candidates
            is_not_abstract_or_toc = not (
                re.search(r'\b(abstract|introduction|summary|table of contents|目次|序論|抄録|resumen)\b', text) or
                text.count('.') > 1 and len(text.split()) > 10 or # Avoid long text with multiple sentences
                text.endswith('.') # Titles rarely end with periods
            )
            
            # Filter out obvious page numbers or short numeric strings
            is_not_page_number = not re.match(r'^\s*(\d+(\.\d+)*|[IVXLCDM]+)\s*$', text) and len(text.strip()) > 5


            if (pred == 4 and is_not_abstract_or_toc) or \
               (is_strong_title_candidate and is_centered_or_wide and is_not_abstract_or_toc and is_not_page_number):
                title_candidates.append(element)
        
        if title_candidates:
            # Sort by font size (desc), then y-position (asc - higher on page is better), then x-position (closer to center)
            sorted_candidates = sorted(title_candidates,
                                       key=lambda x: (x.font_size, x.y, abs(x.x - page_width / 2)),
                                       reverse=True)
            
            # Select the best candidate
            best_title_element = sorted_candidates[0]
            
            # Aggressive post-check for the best candidate
            final_text = best_title_element.text.strip()
            if (len(final_text) > 2 and
                not re.match(r'^\s*(\d+(\.\d+)*|[IVXLCDM]+)\s*$', final_text) and # Not just a number
                not final_text.endswith('.') and # Titles usually don't end with a period
                not re.search(r'\b(abstract|introduction|summary|table of contents|目次|序論|抄録|resumen)\b', final_text.lower())):
                return final_text

        return "" # No title found

    def post_process_headings(self, elements: List[TextElement], predictions: List[int]) -> List[Dict]:
        """
        Post-processes raw predictions to form a structured, hierarchical outline.
        This step cleans up false positives and enforces logical hierarchy.
        """
        raw_headings = []
        doc_stats = self.classifier.feature_extractor.calculate_document_stats(elements)
        median_font_size = doc_stats.get('median_font_size', 12)

        for element, pred in zip(elements, predictions):
            # Only consider elements predicted as H1, H2, or H3
            if pred in [1, 2, 3]:
                heading_text = element.text.strip()
                heading_text = re.sub(r'\s+', ' ', heading_text)

                # Aggressive filtering of non-heading patterns
                if not heading_text or \
                   re.match(r'^\s*(\d+(\.\d+)*)\s*$', heading_text) and len(heading_text.split()) < 3 or \
                   heading_text.count('.') > 2 and len(heading_text.split()) > 10 or \
                   len(re.findall(r'[a-zA-Z0-9]', heading_text)) / max(1, len(heading_text)) < 0.3:
                    continue

                # Exclude if it explicitly looks like a figure/table caption
                if re.search(r'\b(fig\.|figure|table|graph|chart|appendix)\s+\d+(\.\d+)*', heading_text, re.IGNORECASE):
                    # Only allow if it's explicitly a "List of Figures/Tables" type heading
                    if not re.search(r'\b(list of figures|list of tables|目次|表目次|図目次|contents)\b', heading_text, re.IGNORECASE):
                        logger.debug(f"Filtered out likely table/figure caption: '{heading_text}'")
                        continue
                
                # Exclude very short and non-descriptive text if it's not very prominent
                # Adjust font size threshold based on median font size
                if len(heading_text) < 8 and element.font_size < median_font_size * 1.2:
                    continue
                
                # Filter out footers/headers appearing as headings
                if element.y > doc_stats['page_height'] * 0.9 or element.y < doc_stats['page_height'] * 0.05:
                    if len(heading_text.split()) < 10: # Short text at top/bottom, likely not a real heading
                        continue


                raw_headings.append({
                    "level": pred, # Use numeric level for easier comparison
                    "text": heading_text,
                    "page": element.page_num, # Changed to 0-indexed page number for output
                    "y_pos": element.y, # Keep y-pos for sorting within a page
                    "x_pos": element.x,
                    "font_size": element.font_size,
                    "is_bold": element.is_bold
                })

        # Sort all identified headings by page number, then by y-position
        raw_headings.sort(key=lambda h: (h["page"], h["y_pos"]))

        # Refine hierarchy based on font size and numeric levels
        final_outline = []
        # Store the last seen heading for each level to enforce hierarchy
        last_headings = {1: None, 2: None, 3: None}

        for i, heading in enumerate(raw_headings):
            current_level = heading["level"]
            
            # Simple deduplication based on text and very close proximity
            if final_outline and final_outline[-1]["text"] == heading["text"] and \
               final_outline[-1]["page"] == heading["page"] and \
               abs(final_outline[-1]["y_pos"] - heading["y_pos"]) < 5: # Small vertical tolerance
                continue

            # Heuristic to re-evaluate level if hierarchy seems broken (e.g., H3 before H2 on same page)
            # Prioritize larger font sizes and higher positions for promotion
            
            # Promote if current heading is significantly larger or higher than expected for its level
            if current_level > 1 and last_headings[current_level - 1] is not None and \
               heading["page"] == last_headings[current_level - 1]["page"] and \
               heading["y_pos"] < last_headings[current_level - 1]["y_pos"] and \
               heading["font_size"] > last_headings[current_level - 1]["font_size"] * 0.95:
                # If current item is above and similar/larger font than previous higher level, promote it
                current_level = max(1, current_level - 1) # Promote by one level

            # Ensure logical flow: a level X heading cannot appear before a level X-1 heading
            # on the same page unless it's clearly a new section (much larger font, significant vertical space).
            if current_level > 1 and last_headings[current_level - 1] is None:
                # If there's no parent of the immediate higher level, check if it should be promoted
                # e.g., an H3 appears, but no H2 yet. It might be an H2.
                # Check for larger font than typical body and significant vertical gap.
                if heading["font_size"] > median_font_size * 1.2 and \
                   (i == 0 or abs(heading["y_pos"] - raw_headings[i-1]["y_pos"]) > median_font_size * 2):
                    current_level = max(1, current_level - 1) # Promote

            # Update last_headings for current and lower levels
            last_headings[current_level] = heading
            for l in range(current_level + 1, 4): # Reset lower levels
                last_headings[l] = None
            
            heading["level"] = current_level # Update heading with adjusted level
            final_outline.append(heading)

        # Final formatting and cleanup of temporary keys
        for h in final_outline:
            h["level"] = f"H{h['level']}" # Convert numeric level back to H1, H2, H3
            if "y_pos" in h: del h["y_pos"]
            if "x_pos" in h: del h["x_pos"]
            if "font_size" in h: del h["font_size"]
            if "is_bold" in h: del h["is_bold"]

        return final_outline

    def process_directory(self, input_dir: str, output_dir: str, save_csv: bool = False):
        """Process all PDF files in input directory"""
        if not os.path.exists(input_dir):
            logger.error(f"Input directory not found: {input_dir}")
            return

        os.makedirs(output_dir, exist_ok=True)
        pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]

        if not pdf_files:
            logger.warning(f"No PDF files found in '{input_dir}'. Nothing to process.")
            return

        logger.info(f"Found {len(pdf_files)} PDF files in '{input_dir}' to process.")

        for i, pdf_file in enumerate(pdf_files):
            logger.info(f"Processing file {i+1}/{len(pdf_files)}: '{pdf_file}'")
            try:
                pdf_path = os.path.join(input_dir, pdf_file)
                base_name = os.path.splitext(pdf_file)[0]
                
                # Define paths for outputs
                json_output_path = os.path.join(output_dir, base_name + '.json')
                csv_output_path = os.path.join(output_dir, base_name + '_dataset.csv') if save_csv else None

                outline = self.extract_outline(pdf_path, csv_output_path)

                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(outline, f, indent=2, ensure_ascii=False)

                logger.info(f"Successfully processed '{pdf_file}' and saved outline to '{base_name}.json'.")

            except Exception as e:
                logger.error(f"An error occurred while processing '{pdf_file}': {e}", exc_info=True)


def main():
    """
    MAIN FUNCTION FOR DEVELOPMENT:
    Use this to train your model or run predictions with specific folders.
    """
    parser = argparse.ArgumentParser(description="Multilingual PDF Outline Extraction Tool")
    subparsers = parser.add_subparsers(dest='action', required=True, help='Action to perform')

    # --- Training Parser ---
    parser_train = subparsers.add_parser('train', help='Train the heading classifier model from CSV data.')
    parser_train.add_argument('--assets-dir', type=str, default='assets', help='Directory containing the labeled CSV files for training.')
    parser_train.add_argument('--model-dir', type=str, default='model', help='Directory to save the trained model.')

    # --- Prediction Parser ---
    parser_predict = subparsers.add_parser('predict', help='Predict outlines for PDFs in a directory.')
    parser_predict.add_argument('--input-dir', type=str, default='input', help='Input directory containing PDF files.')
    parser_predict.add_argument('--output-dir', type=str, default='output', help='Output directory to save JSON outlines.')
    parser_predict.add_argument('--model-dir', type=str, default='model', help='Directory to load the trained model from.')
    parser_predict.add_argument('--save-csv', action='store_true', help='If set, save a CSV dataset for each PDF.')

    args = parser.parse_args()

    if args.action == 'train':
        logger.info("--- Running in Training Mode ---")
        extractor = PDFOutlineExtractor(model_dir=args.model_dir)
        extractor.train_and_save_model(assets_dir=args.assets_dir)
        logger.info("--- Training complete. ---")

    elif args.action == 'predict':
        logger.info("--- Running in Prediction Mode ---")
        extractor = PDFOutlineExtractor(model_dir=args.model_dir)
        extractor.process_directory(args.input_dir, args.output_dir, save_csv=args.save_csv)
        logger.info("--- Prediction complete for all files. ---")

# def main():
#     """
#     MAIN FUNCTION FOR HACKATHON SUBMISSION:
#     Processes all PDFs in /app/input and saves JSONs to /app/output.
#     """
#     input_dir = './input'
#     output_dir = './output'
#     model_dir = 'model'
#     logger.info("--- Running in Hackathon Prediction Mode ---")
#     extractor = PDFOutlineExtractor(model_dir=model_dir)
#     extractor.process_directory(input_dir, output_dir, save_csv=False)
#     logger.info("--- All files processed. Exiting. ---")



if __name__ == "__main__":
    main()