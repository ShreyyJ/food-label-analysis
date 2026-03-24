# Food Label Analysis

A Streamlit-based application that analyzes nutrition labels from food product images using Optical Character Recognition (OCR). Extract nutrition information like calories, protein, fat, carbohydrates, and more directly from food packaging photos.

## Features

- 📸 **Image Upload**: Load food label images from your device
- 🔍 **OCR Processing**: Dual OCR engine support with automatic fallback
  - **PaddleOCR**: Fast and accurate text extraction
  - **EasyOCR**: Reliable backup engine
- 📊 **Data Extraction**: Automatically extracts nutrition information including:
  - Product name
  - Serving size
  - Total weight
  - Calories
  - Macronutrients (protein, fat, carbohydrates)
  - Micronutrients (sodium, fiber, sugar)
  - Daily value percentages
- 💾 **Local Processing**: All processing happens on your machine - no data sent to external servers

## Tech Stack

- **Frontend**: Streamlit
- **OCR Engines**: PaddleOCR, EasyOCR
- **Image Processing**: Pillow, NumPy
- **Data Analysis**: Pandas

## Installation

### Prerequisites
- Python 3.8 or higher

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ShreyyJ/food-label-analysis.git
cd food-label-analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

### How to Use

1. Upload an image of a food label
2. The application will process the image using OCR
3. Extracted nutrition information will be displayed in a structured format
4. Review and use the extracted data

## Requirements

All dependencies are listed in `requirements.txt`:
- streamlit
- pillow
- easyocr
- paddleocr
- paddlepaddle
- numpy
- pandas

## Project Structure

```
food-label-analysis/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Notes

- The first run may take some time as OCR engines download their models
- Results are cached to improve performance on subsequent runs
- The application automatically selects the best available OCR engine

## License


## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## Contact

Shreya J - shreyareddy5544@gmail.com