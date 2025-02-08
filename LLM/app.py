from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
import imghdr

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///images.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Model
class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    file_size = db.Column(db.Integer, nullable=False)  # in bytes

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file_path):
    """Validate that the file is actually an image"""
    return imghdr.what(file_path) is not None

@app.route('/upload', methods=['POST'])
def upload_image():
    logger.info('Received upload request')
    logger.debug(f'Files in request: {request.files}')
    logger.debug(f'Headers: {request.headers}')

    # Check if the post request has the file part
    if 'image' not in request.files:
        logger.error('No image file in request')
        return jsonify({'error': 'No image file found in request'}), 400

    file = request.files['image']
    
    # Check if a file was selected
    if file.filename == '':
        logger.error('Empty filename received')
        return jsonify({'error': 'No file selected'}), 400

    # Check file extension
    if not allowed_file(file.filename):
        logger.error('Invalid file type')
        return jsonify({'error': 'Invalid file type. Allowed types: PNG, JPG, JPEG, GIF'}), 400

    try:
        # Save the file with a secure filename
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        unique_filename = timestamp + filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        # Validate that it's actually an image
        if not validate_image(file_path):
            os.remove(file_path)  # Remove the invalid file
            logger.error('Invalid image file')
            return jsonify({'error': 'Invalid image file'}), 400

        # Get file size
        file_size = os.path.getsize(file_path)

        # Save to database
        image_entry = Image(
            filename=unique_filename,
            filepath=file_path,
            file_size=file_size
        )
        db.session.add(image_entry)
        db.session.commit()

        logger.info(f'Successfully saved file to {file_path} and database')
        
        return jsonify({
            'message': 'Image uploaded successfully',
            'file_path': file_path,
            'filename': unique_filename,
            'id': image_entry.id
        }), 200

    except Exception as e:
        logger.error(f'Error saving file: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/images', methods=['GET'])
def get_images():
    """Endpoint to retrieve list of uploaded images"""
    try:
        images = Image.query.order_by(Image.upload_date.desc()).all()
        return jsonify({
            'images': [{
                'id': img.id,
                'filename': img.filename,
                'upload_date': img.upload_date.isoformat(),
                'file_size': img.file_size
            } for img in images]
        }), 200
    except Exception as e:
        logger.error(f'Error retrieving images: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File is too large. Maximum size is 16MB'}), 413

# Create the database tables
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    logger.info('Starting Flask server...')
    app.run(debug=True, host='0.0.0.0', port=5000)