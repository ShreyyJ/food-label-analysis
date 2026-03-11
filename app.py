# app.py — Nutrition Label Insights (Local Processing)

import streamlit as st
from PIL import Image
import easyocr
try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None
import numpy as np
import re
import datetime
from typing import Dict, Any, List, Tuple, Optional, Sequence
import pandas as pd

# Initialize OCR engine (only needs to be done once)
@st.cache_resource
def load_ocr():
    if PaddleOCR is not None:
        try:
            return {
                'backend': 'PaddleOCR',
                'engine': PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            }
        except Exception:
            pass

    return {
        'backend': 'EasyOCR',
        'engine': easyocr.Reader(['en'])
    }


def run_ocr_text(ocr_bundle: Dict[str, Any], img_array: np.ndarray) -> List[str]:
    """Run OCR and normalize output to list[str]."""
    backend = ocr_bundle['backend']
    engine = ocr_bundle['engine']

    if backend == 'PaddleOCR':
        raw_result = engine.ocr(img_array, cls=True)
        lines = raw_result[0] if raw_result and isinstance(raw_result[0], list) else []
        extracted: List[str] = []
        for line in lines:
            if (
                isinstance(line, list)
                and len(line) >= 2
                and isinstance(line[1], (list, tuple))
                and len(line[1]) >= 1
            ):
                text = str(line[1][0]).strip()
                if text:
                    extracted.append(text)
        return extracted

    return [str(item) for item in engine.readtext(img_array, detail=0)]

def extract_number(text: str) -> float:
    """Extract numeric value from text, handling various formats."""
    try:
        # Remove everything except digits, decimal points, and negative signs
        clean = re.sub(r'[^0-9.-]', '', text)
        return float(clean)
    except:
        return 0.0

def extract_value_with_unit(text: str, patterns: Dict[str, str]) -> Tuple[float, str]:
    """Extract value and unit from text using patterns."""
    for unit, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = extract_number(match.group(1))
            return value, unit
    return 0.0, ''

def parse_nutrition_text(text_list: Sequence[Any]) -> Dict[str, Any]:
    """Parse OCR text into structured nutrition data."""
    result = {
        'product_name': None,
        'serving_size': None,
        'total_weight_g': None,
        'calories': None,
        'protein': None,
        'total_fat': None,
        'saturated_fat': None,
        'carbohydrates': None,
        'sugar': None,
        'fiber': None,
        'sodium': None,
        'unit_type': {},
        'daily_values': {}
    }
    
    # Join OCR text into one searchable string
    text_items = [str(item) for item in text_list if item is not None]
    full_text = ' '.join(text_items).lower()
    
    # Try to find total package weight
    weight_match = re.search(r'net\s*wt\.?\s*(\d+[\d.]*)\s*g', full_text)
    if weight_match:
        result['total_weight_g'] = extract_number(weight_match.group(1))
    
    # Pattern dictionary with unit patterns - made more flexible for OCR quirks
    nutrient_patterns = {
        'serving_size': {
            'g': r'serving\s*(?:size)?\s*[:\s]*(\d+[\d.]*)\s*g',
            'ml': r'serving\s*(?:size)?\s*[:\s]*(\d+[\d.]*)\s*ml'
        },
        'calories': {
            'cal': r'(?:energy|calories?)\s*[:\s]*(\d+)'
        },
        'protein': {
            'g': r'protein\s*[:\s]*(\d+[\d.]*)\s*g?',
            '%': r'protein\s*[:\s]*(\d+[\d.]*)\s*%'
        },
        'total_fat': {
            'g': r'total\s+fat\s*[:\s]*(\d+[\d.]*)\s*g?',
            '%': r'total\s+fat\s*[:\s]*(\d+[\d.]*)\s*%'
        },
        'saturated_fat': {
            'g': r'saturated\s+fat\s*[:\s]*(\d+[\d.]*)\s*g?',
            '%': r'saturated\s+fat\s*[:\s]*(\d+[\d.]*)\s*%'
        },
        'carbohydrates': {
            'g': r'carbohydrate[s]?\s*[:\s]*(\d+[\d.]*)\s*g?',
            '%': r'carbohydrate[s]?\s*[:\s]*(\d+[\d.]*)\s*%'
        },
        'sugar': {
            'g': r'[~t]?(?:total\s+)?sugars?\s*[:\s]*(\d+[\d.]*)\s*g?',
            '%': r'[~t]?(?:total\s+)?sugars?\s*[:\s]*(\d+[\d.]*)\s*%',
            'plain': r'[~t]?(?:total\s+)?sugars?\s*[:\s]*(\d+[\d.]*)'
        },
        'fiber': {
            'g': r'(?:dietary\s+)?fiber\s*[:\s]*(\d+[\d.]*)\s*g?',
            '%': r'(?:dietary\s+)?fiber\s*[:\s]*(\d+[\d.]*)\s*%'
        },
        'sodium': {
            'mg': r'sodium\s*[:\s]*(\d+[\d.]*)\s*mg?',
            '%': r'sodium\s*[:\s]*(\d+[\d.]*)\s*%'
        }
    }
    
    # Process each nutrient
    for nutrient, unit_patterns in nutrient_patterns.items():
        for unit, pattern in unit_patterns.items():
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                # Extract value (some patterns have multiple groups, take last number group)
                groups = match.groups()
                value = next((g for g in reversed(groups) if g and any(c.isdigit() for c in g)), None)
                if value:
                    result[nutrient] = extract_number(value)
                    result['unit_type'][nutrient] = unit
                break  # Stop after first match for this nutrient

    # Fallback parsing for OCR that splits labels and values across lines
    normalized_lines = [re.sub(r'\s+', ' ', str(item).strip().lower()) for item in text_items]

    # Common OCR fixes for nutrition lines
    normalized_lines = [
        re.sub(r'\b(total\s+fat|protein|saturated\s+fat)\s+og\b', r'\1 0g', line)
        for line in normalized_lines
    ]

    def extract_from_line(
        line_text: str,
        allowed_units: Optional[Tuple[str, ...]] = None,
        allow_unitless: bool = False
    ) -> Optional[Tuple[float, str]]:
        match = re.search(r'(\d+[\d.]*)\s*(mg|g|kcal|cal|%)?', line_text)
        if not match:
            return None
        value = extract_number(match.group(1))
        unit = (match.group(2) or '').lower()
        if allowed_units:
            if unit in allowed_units:
                return value, unit
            if unit == '' and allow_unitless:
                return value, unit
            return None
        return value, unit

    def assign_if_missing(
        nutrient: str,
        label_pattern: str,
        default_unit: str,
        lookahead: int = 3,
        allowed_units: Optional[Tuple[str, ...]] = None,
        allow_unitless_same_line: bool = True,
        allow_unitless_nearby: bool = False
    ):
        if result.get(nutrient) is not None:
            return
        for idx, line in enumerate(normalized_lines):
            if re.search(label_pattern, line):
                same_line = extract_from_line(
                    line,
                    allowed_units=allowed_units,
                    allow_unitless=allow_unitless_same_line
                )
                if same_line:
                    value, unit = same_line
                    result[nutrient] = value
                    result['unit_type'][nutrient] = unit or default_unit
                    return
                for next_idx in range(idx + 1, min(len(normalized_lines), idx + lookahead + 1)):
                    nearby = extract_from_line(
                        normalized_lines[next_idx],
                        allowed_units=allowed_units,
                        allow_unitless=allow_unitless_nearby
                    )
                    if nearby:
                        value, unit = nearby
                        result[nutrient] = value
                        result['unit_type'][nutrient] = unit or default_unit
                        return

    assign_if_missing('sugar', r'sugars?', 'g', lookahead=5, allowed_units=('g', '%'), allow_unitless_nearby=False)
    assign_if_missing('sodium', r'sodium', 'mg', lookahead=20, allowed_units=('mg', '%'), allow_unitless_nearby=False)
    assign_if_missing('protein', r'protein', 'g', lookahead=3, allowed_units=('g', '%'), allow_unitless_nearby=False)
    assign_if_missing('total_fat', r'total\s+fat', 'g', lookahead=3, allowed_units=('g', '%'), allow_unitless_nearby=False)

    # Sodium-specific rescue: if sodium is missing/implausible, prefer an explicit mg value over stray OCR digits
    sodium_val = result.get('sodium')
    sodium_unit = result['unit_type'].get('sodium')
    if sodium_val is None or (sodium_unit == 'mg' and float(sodium_val) < 5):
        sodium_line_indices = [i for i, line in enumerate(normalized_lines) if re.search(r'sodium', line)]
        mg_candidates: List[Tuple[float, int]] = []
        for idx, line in enumerate(normalized_lines):
            if 'caffeine' in line:
                continue
            match = re.search(r'(\d+[\d.]*)\s*mg\b', line)
            if match:
                value = extract_number(match.group(1))
                if 5 <= value <= 3000:
                    mg_candidates.append((value, idx))
        if sodium_line_indices and mg_candidates:
            best_value, _ = min(
                mg_candidates,
                key=lambda candidate: min(abs(candidate[1] - s_idx) for s_idx in sodium_line_indices)
            )
            result['sodium'] = best_value
            result['unit_type']['sodium'] = 'mg'

    # Beverage heuristic: if sugars are missing but carbs are present and ingredients indicate added sugars
    if (
        result.get('sugar') is None
        and result.get('carbohydrates') is not None
        and re.search(r'fructose|corn\s*syrup|cola|soda|added\s+sugars?', full_text, re.IGNORECASE)
    ):
        result['sugar'] = result['carbohydrates']
        result['unit_type']['sugar'] = 'g'
    
    # Try to find product name (usually at the top, capitalized)
    # Take the first few words before "nutrition facts" if present
    name_match = re.search(r'^(.+?)(?:nutrition facts|serving size)', full_text, re.I)
    if name_match:
        result['product_name'] = name_match.group(1).strip().title()
    
    return result

def calculate_health_score(data: Dict[str, Any]) -> Tuple[int, List[str], Dict[str, Any]]:
    """Calculate health score and generate insights."""
    score = 100
    insights = []
    score_breakdown = {
        'base_score': 100,
        'adjustments': []
    }

    def as_float(value: Any) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def add_score_adjustment(points: int, reason: str, insight: Optional[str] = None):
        score_breakdown['adjustments'].append({
            'points': points,
            'reason': reason
        })
        if insight:
            insights.append(insight)

    sugar_val = as_float(data.get('sugar'))
    added_sugar_val = as_float(data.get('sugar'))  # Some labels list added sugars separately
    sodium_val = as_float(data.get('sodium'))
    protein_val = as_float(data.get('protein'))
    fiber_val = as_float(data.get('fiber'))
    total_fat_val = as_float(data.get('total_fat'))
    sat_fat_val = as_float(data.get('saturated_fat'))

    # Sugar analysis (stricter thresholds)
    if sugar_val > 0:
        if data['unit_type'].get('sugar') == '%':
            if sugar_val > 30:
                add_score_adjustment(-25, "Very high sugar (>30% DV)", f"Very high sugar content ({sugar_val}% DV)")
            elif sugar_val > 20:
                add_score_adjustment(-20, "High sugar content (>20% DV)", f"High sugar content ({sugar_val}% DV)")
            elif sugar_val > 10:
                add_score_adjustment(-10, "Moderate sugar content (>10% DV)", f"Moderate sugar content ({sugar_val}% DV)")
        else:  # Assuming grams
            if sugar_val > 20:
                add_score_adjustment(-25, "Very high sugar (>20g)", f"Very high sugar content ({sugar_val}g)")
            elif sugar_val > 12:
                add_score_adjustment(-20, "High sugar content (>12g)", f"High sugar content ({sugar_val}g)")
            elif sugar_val > 5:
                add_score_adjustment(-10, "Moderate sugar content (5-12g)", f"Moderate sugar content ({sugar_val}g)")

    # Sodium analysis
    if sodium_val > 0:
        if data['unit_type'].get('sodium') == '%':
            if sodium_val > 50:
                add_score_adjustment(-20, "High sodium content (>50% DV)", f"High sodium content ({sodium_val}% DV)")
            elif sodium_val > 25:
                add_score_adjustment(-10, "Moderate sodium content (>25% DV)", f"Moderate sodium content ({sodium_val}% DV)")
        else:  # Assuming mg
            if sodium_val > 2000:
                add_score_adjustment(-20, "High sodium content (>2000mg)", f"High sodium content ({sodium_val}mg)")
            elif sodium_val > 1000:
                add_score_adjustment(-10, "Moderate sodium content (>1000mg)", f"Moderate sodium content ({sodium_val}mg)")

    # Fat analysis
    if total_fat_val > 17:
        add_score_adjustment(-10, "High total fat (>17g)", f"High total fat ({total_fat_val}g)")
    elif total_fat_val > 10:
        add_score_adjustment(-5, "Moderate total fat (>10g)", f"Moderate total fat ({total_fat_val}g)")

    if sat_fat_val > 5:
        add_score_adjustment(-10, "High saturated fat (>5g)", f"High saturated fat ({sat_fat_val}g)")
    elif sat_fat_val > 2:
        add_score_adjustment(-5, "Moderate saturated fat (>2g)", f"Moderate saturated fat ({sat_fat_val}g)")

    # Protein analysis
    if protein_val > 0:
        if data['unit_type'].get('protein') == '%':
            if protein_val > 20:
                add_score_adjustment(5, "Good protein content (>20% DV)", f"Good source of protein ({protein_val}% DV)")
            elif protein_val < 5:
                add_score_adjustment(-5, "Low protein content (<5% DV)", "Low in protein")
        else:  # Assuming grams
            if protein_val > 20:
                add_score_adjustment(5, "Good protein content (>20g)", f"Good source of protein ({protein_val}g)")
            elif protein_val < 5:
                add_score_adjustment(-5, "Low protein content (<5g)", "Low in protein")

    # Fiber analysis
    if fiber_val > 0:
        if data['unit_type'].get('fiber') == '%':
            if fiber_val > 20:
                add_score_adjustment(5, "Good fiber content (>20% DV)", f"Good source of fiber ({fiber_val}% DV)")
        else:  # Assuming grams
            if fiber_val > 5:
                add_score_adjustment(5, "Good fiber content (>5g)", f"Good source of fiber ({fiber_val}g)")

    # Apply completeness penalties (instead of hard caps) so score remains sensitive
    tracked_metrics = ['sugar', 'sodium', 'protein', 'fiber', 'total_fat', 'saturated_fat', 'carbohydrates', 'calories']
    extracted_count = sum(1 for metric in tracked_metrics if data.get(metric) is not None)

    if extracted_count <= 2:
        add_score_adjustment(-18, "Very limited nutrition data extracted", "Very limited nutrition data extracted; confidence is lower.")
    elif extracted_count <= 4:
        add_score_adjustment(-12, "Limited nutrition data extracted", "Limited nutrition data extracted; confidence is moderate.")
    elif extracted_count <= 6:
        add_score_adjustment(-6, "Partial nutrition data extracted", "Partial nutrition data extracted; score adjusted for reliability.")

    final_score = score + sum(adj['points'] for adj in score_breakdown['adjustments'])

    final_score = max(0, min(100, final_score))

    return final_score, insights, score_breakdown

# --- Helpers ---
def main():
    # Page config
    st.set_page_config(
        page_title="Nutrition Insights",
        page_icon="🥗",
        layout="wide"
    )

    # Custom CSS for better styling
    st.markdown("""
        <style>
            .main {
                padding: 2rem;
            }
            .stImage {
                max-width: 400px !important;
                margin: 0 auto;
            }
            .nutrition-stats {
                padding: 1.5rem;
                background-color: #f0f2f6;
                border-radius: 10px;
                margin: 1rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .stProgress > div > div > div > div {
                background-color: #2ecc71;
            }
            .health-score-container {
                padding: 1.5rem;
                border-radius: 8px;
                background: linear-gradient(135deg, #2ecc71, #27ae60);
                color: white;
                text-align: center;
                margin: 1rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .health-score-explanation {
                font-size: 0.9rem;
                color: #666;
                padding: 1.5rem;
                background-color: #fff;
                border-radius: 8px;
                border: 1px solid #eee;
                margin-top: 1rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }
            .insights-card {
                background-color: #fff;
                padding: 1.5rem;
                border-radius: 8px;
                border: 1px solid #eee;
                margin: 1rem 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }
            h1 {
                color: #2c3e50;
                margin-bottom: 2rem;
                text-align: center;
            }
            h2, h3, h4 {
                color: #34495e;
                margin: 1.5rem 0 1rem 0;
            }
            .stAlert {
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            }
        </style>
    """, unsafe_allow_html=True)

    # Custom header with better contrast
    st.markdown("""
        <h1 style='color: white; text-align: center; margin-bottom: 1rem;'>
            🥗 Nutrition Label Insights
        </h1>
    """, unsafe_allow_html=True)

    # Add app description with improved styling
    st.markdown("""
    <div class="insights-card">
        <p style='color: #2c3e50; font-size: 1.1rem; line-height: 1.5;'>
            Upload a nutrition label image to get detailed insights about the nutritional content, 
            health score, and macronutrient breakdown. Our advanced analysis will help you make 
            informed decisions about your food choices.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Two-column layout for upload and display
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("📤 Upload Label")
        uploaded_file = st.file_uploader(
            "Choose a nutrition label image",
            type=["png", "jpg", "jpeg"],
            key="nutrition_label_uploader"
        )

        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            
            # Resize image while maintaining aspect ratio
            max_width = 400
            ratio = max_width / float(img.size[0])
            new_size = (max_width, int(float(img.size[1]) * ratio))
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
            
            st.image(
                img_resized, 
                caption="Uploaded nutrition label", 
                use_container_width=True
            )
            analyze_button = st.button(
                "🔍 Analyze Nutrition",
                key="analyze_button",
                use_container_width=True
            )

            if analyze_button:
                try:
                    # Convert PIL Image to numpy array for OCR
                    img_array = np.array(img)
                    
                    with st.spinner("Analyzing nutrition label..."):
                        # Get OCR engine
                        ocr_bundle = load_ocr()
                        
                        # Extract text from image
                        results = run_ocr_text(ocr_bundle, img_array)
                        
                        # Debug: Show what OCR extracted
                        with st.expander("🔍 Debug: OCR Text (click to expand)"):
                            st.text(f"OCR backend: {ocr_bundle['backend']}")
                            st.text(f"OCR extracted {len(results)} text blocks:")
                            for i, text in enumerate(results):
                                st.code(f"{i}: {text}")
                        
                        # Parse the extracted text
                        nutrition_data = parse_nutrition_text(results)
                        
                        # Calculate health score and insights
                        health_score, insights, score_breakdown = calculate_health_score(nutrition_data)
                        
                        with col2:
                            st.success("✅ Analysis Complete")
                            
                            # Display the results using the existing display code...
                            process_and_display_results(
                                nutrition_data,
                                health_score,
                                insights,
                                score_breakdown
                            )
                            
                except Exception as e:
                    st.error(f"Error analyzing image: {str(e)}")
                    
        else:
            analyze_button = False
            with col2:
                st.info("📸 Upload a nutrition label image to begin analysis")

def process_and_display_results(nutrition_data, health_score, insights, score_breakdown):
    """Process and display the analysis results."""
    def as_float(value: Any) -> float:
        """Safely convert nutrition values to float for calculations."""
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    # Display health score with explanation
    st.markdown("## 🏆 Health Score Analysis")
    
    # Display score in custom container
    st.markdown(f"""
    <div class="health-score-container">
        <h2 style="color: white; margin: 0;">Health Score: {health_score}/100</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Show score explanation
    st.markdown("""
    <div class="health-score-explanation">
        <h4>How is this score calculated?</h4>
        <p>The health score starts at 100 and is adjusted based on:</p>
        <ul>
            <li>Sugar content (WHO recommendation: &lt;25g/day)</li>
            <li>Sodium levels (WHO recommendation: &lt;2000mg/day)</li>
            <li>Total and saturated fat penalties</li>
            <li>Protein/fiber bonuses (moderate weighting)</li>
            <li>Data completeness cap when OCR extracts limited fields</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Show score breakdown
    if score_breakdown['adjustments']:
        st.markdown("### Score Breakdown")
        for adj in score_breakdown['adjustments']:
            if adj['points'] > 0:
                st.success(f"+{adj['points']} points: {adj['reason']}")
            else:
                st.warning(f"{adj['points']} points: {adj['reason']}")
    
    # Calculate and display macronutrient distribution
    protein_g = as_float(nutrition_data.get('protein'))
    carbs_g = as_float(nutrition_data.get('carbohydrates'))
    fat_g = as_float(nutrition_data.get('total_fat'))

    total_calories = (protein_g * 4) + (carbs_g * 4) + (fat_g * 9)
    
    if total_calories > 0:
        st.markdown("### 📊 Macronutrient Distribution")
        cols = st.columns(3)
        
        with cols[0]:
            protein_pct = int((protein_g * 4 / total_calories) * 100)
            st.metric("Protein", f"{protein_pct}%")
        with cols[1]:
            carbs_pct = int((carbs_g * 4 / total_calories) * 100)
            st.metric("Carbs", f"{carbs_pct}%")
        with cols[2]:
            fat_pct = int((fat_g * 9 / total_calories) * 100)
            st.metric("Fats", f"{fat_pct}%")
    
    # Display detailed nutrition data in expander
    with st.expander("📋 Detailed Nutrition Data"):
        # Convert the nutrition data to a prettier format
        display_data = []
        for nutrient, value in nutrition_data.items():
            if nutrient not in ['unit_type', 'daily_values', 'vitamins'] and value is not None:
                unit = nutrition_data['unit_type'].get(nutrient, '')
                if unit:
                    display_data.append({
                        'Nutrient': nutrient.replace('_', ' ').title(),
                        'Value': f"{value}{unit}"
                    })
        
        if display_data:
            df = pd.DataFrame(display_data)
            st.dataframe(df, hide_index=True, use_container_width=True)

if __name__ == "__main__":
    main()

st.caption("Upload nutrition labels to see structured data + health insights. Powered by PaddleOCR (offline) with EasyOCR fallback.")
