from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from core.detector import AdvancedPlagiarismDetector
from utils.pdf_processor import extract_text_from_pdf
from config import DEFAULT_REFERENCE_PATH, MIN_TEXT_LENGTH
from report_generator import generate_detailed_report

app = Flask(__name__)
CORS(app)

# Global Instance
detector = None

def get_detector():
    global detector
    if detector is None:
        detector = AdvancedPlagiarismDetector(DEFAULT_REFERENCE_PATH)
    return detector

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    det = get_detector()
    return jsonify({'status': 'ok', 'books_loaded': len(det.books_df)})

@app.route('/analyze', methods=['POST'])
def analyze():
    text = ""
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.pdf'):
                text = extract_text_from_pdf(file)
            elif file.filename.endswith('.txt'):
                text = file.read().decode('utf-8')
        elif request.is_json:
            text = request.get_json().get('essay_text', '')

        if len(text) < MIN_TEXT_LENGTH:
            return jsonify({'error': f"Text too short. Min {MIN_TEXT_LENGTH} chars required."}), 400

        det = get_detector()
        # Run Analysis
        analysis_result = det.analyze_text(text)
        
        # MUST return the extracted text so the frontend can use it for the detailed report!
        analysis_result['extracted_text'] = text
        
        return jsonify(analysis_result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/report-viewer')
def report_viewer():
    return render_template('report_viewer.html')

@app.route('/report', methods=['POST'])
def report():
    try:
        data = request.get_json()
        if not data or 'text' not in data or 'analysis' not in data:
            return jsonify({'error': 'Missing text or analysis data'}), 400
            
        detailed_report = generate_detailed_report(data['text'], data['analysis'])
        return jsonify(detailed_report)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize on startup
    get_detector()
    app.run(debug=True, host='0.0.0.0', port=5001)