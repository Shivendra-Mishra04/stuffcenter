"""
Professional Flask API for Product Identification System

This Flask application provides a comprehensive REST API for visual product similarity search
using deep learning feature extraction and PostgreSQL with pgvector for efficient vector search.

Features:
- Image upload and processing with comprehensive validation
- Vector similarity search with multiple metrics
- Product database management
- Health monitoring and detailed statistics
- Rate limiting and enhanced security
- Comprehensive error handling and logging
- Background task management
- Production-ready configurations
"""

import os
import uuid
import time
import logging
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from functools import wraps
from contextlib import contextmanager
import weakref
import gc

from flask import Flask, request, jsonify, abort, g
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix
import numpy as np
from PIL import Image

try:
    from search_postgresql_refined import ProductIdentifier, Product
except ImportError as e:
    logging.error(f"Failed to import search module: {e}")
    raise ImportError("search_postgresql_refined module is required")

# Enhanced logging configuration
def setup_logging():
    """Configure enhanced logging with rotation."""
    from logging.handlers import RotatingFileHandler
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(
                'logs/flask_app.log', 
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
        ]
    )
    
    # Set specific log levels for third-party libraries
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

logger = setup_logging()

# Enhanced Flask app configuration
app = Flask(__name__)

# Apply ProxyFix for production deployment behind reverse proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Comprehensive app configuration
app.config.update({
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB max file size
    'UPLOAD_FOLDER': 'uploads',
    'SECRET_KEY': os.environ.get('SECRET_KEY', 'change-this-in-production'),
    'JSON_SORT_KEYS': False,
    'JSONIFY_PRETTYPRINT_REGULAR': True,
    'SEND_FILE_MAX_AGE_DEFAULT': 300,  # 5 minutes cache for static files
    'PERMANENT_SESSION_LIFETIME': timedelta(hours=1),
})

# Enhanced security and configuration constants
ALLOWED_EXTENSIONS = frozenset(['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'])
RATE_LIMIT_PER_MINUTE = int(os.environ.get('RATE_LIMIT_PER_MINUTE', '60'))
MAX_PRODUCTS_PER_BATCH = int(os.environ.get('MAX_PRODUCTS_PER_BATCH', '100'))
CLEANUP_INTERVAL_SECONDS = int(os.environ.get('CLEANUP_INTERVAL_SECONDS', '1800'))  # 30 minutes
FILE_MAX_AGE_SECONDS = int(os.environ.get('FILE_MAX_AGE_SECONDS', '3600'))  # 1 hour

# Create upload directory with proper permissions
upload_dir = app.config['UPLOAD_FOLDER']
os.makedirs(upload_dir, mode=0o755, exist_ok=True)

# Global application state with thread safety
class ApplicationState:
    """Thread-safe application state management."""
    def __init__(self):
        self._lock = threading.RLock()
        self._product_identifier = None
        self._initialization_time = None
        self._request_counts = {}
        self._cleanup_thread = None
        self._stats = {
            'total_requests': 0,
            'successful_searches': 0,
            'failed_searches': 0,
            'total_processing_time': 0.0
        }
    
    @property
    def product_identifier(self):
        with self._lock:
            return self._product_identifier
    
    @product_identifier.setter
    def product_identifier(self, value):
        with self._lock:
            self._product_identifier = value
            if value:
                self._initialization_time = datetime.utcnow()
    
    def get_request_counts(self):
        with self._lock:
            return self._request_counts.copy()
    
    def update_request_count(self, client_ip: str):
        with self._lock:
            current_time = datetime.now()
            minute_ago = current_time - timedelta(minutes=1)
            
            # Clean old entries
            if client_ip in self._request_counts:
                self._request_counts[client_ip] = [
                    req_time for req_time in self._request_counts[client_ip]
                    if req_time > minute_ago
                ]
            else:
                self._request_counts[client_ip] = []
            
            # Add current request
            self._request_counts[client_ip].append(current_time)
            self._stats['total_requests'] += 1
            
            return len(self._request_counts[client_ip])
    
    def update_stats(self, stat_name: str, value: Any = 1):
        with self._lock:
            if stat_name in self._stats:
                if isinstance(self._stats[stat_name], (int, float)):
                    self._stats[stat_name] += value
                else:
                    self._stats[stat_name] = value
    
    def get_stats(self):
        with self._lock:
            stats = self._stats.copy()
            stats['initialization_time'] = self._initialization_time.isoformat() if self._initialization_time else None
            return stats

# Initialize application state
app_state = ApplicationState()

# Enhanced utility functions
def allowed_file(filename: str) -> bool:
    """Check if uploaded file has allowed extension with enhanced validation."""
    if not filename or '.' not in filename:
        return False
    
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS

def get_client_ip() -> str:
    """Get client IP with support for proxy headers."""
    if request.environ.get('HTTP_X_FORWARDED_FOR'):
        # Handle comma-separated list of IPs
        return request.environ['HTTP_X_FORWARDED_FOR'].split(',')[0].strip()
    elif request.environ.get('HTTP_X_REAL_IP'):
        return request.environ['HTTP_X_REAL_IP']
    else:
        return request.remote_addr or 'unknown'

def check_rate_limit(client_ip: str) -> bool:
    """Enhanced rate limiting with sliding window."""
    current_requests = app_state.update_request_count(client_ip)
    return current_requests <= RATE_LIMIT_PER_MINUTE

@contextmanager
def safe_file_operation(filepath: str):
    """Context manager for safe file operations with automatic cleanup."""
    try:
        yield filepath
    finally:
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
                logger.debug(f"Cleaned up temporary file: {filepath}")
        except OSError as e:
            logger.warning(f"Failed to cleanup file {filepath}: {e}")

def validate_search_parameters(form_data: Dict) -> Tuple[Dict, Optional[str]]:
    """Validate and sanitize search parameters."""
    try:
        # Parse and validate top_k
        top_k = min(max(int(form_data.get('top_k', 5)), 1), 50)  # Clamp between 1-50
        
        # Parse and validate similarity_threshold
        similarity_threshold = float(form_data.get('similarity_threshold', 0.1))
        if not (0.0 <= similarity_threshold <= 1.0):
            return {}, "similarity_threshold must be between 0.0 and 1.0"
        
        # Validate similarity_metric
        similarity_metric = form_data.get('similarity_metric', 'cosine').lower()
        if similarity_metric not in ['cosine', 'l2', 'euclidean']:
            return {}, "similarity_metric must be 'cosine' or 'l2'"
        
        # Normalize euclidean to l2
        if similarity_metric == 'euclidean':
            similarity_metric = 'l2'
        
        # Get optional filters
        category_filter = form_data.get('category', '').strip() or None
        brand_filter = form_data.get('brand', '').strip() or None
        
        return {
            'top_k': top_k,
            'similarity_threshold': similarity_threshold,
            'similarity_metric': similarity_metric,
            'category_filter': category_filter,
            'brand_filter': brand_filter
        }, None
        
    except ValueError as e:
        return {}, f"Invalid parameter format: {str(e)}"

def cleanup_old_files():
    """Enhanced background task for file cleanup with error recovery."""
    logger.info(f"Starting file cleanup task (interval: {CLEANUP_INTERVAL_SECONDS}s)")
    
    while True:
        try:
            current_time = time.time()
            upload_folder = app.config['UPLOAD_FOLDER']
            
            if not os.path.exists(upload_folder):
                logger.warning(f"Upload folder {upload_folder} does not exist")
                time.sleep(CLEANUP_INTERVAL_SECONDS)
                continue
            
            cleaned_count = 0
            error_count = 0
            
            for filename in os.listdir(upload_folder):
                filepath = os.path.join(upload_folder, filename)
                
                try:
                    if os.path.isfile(filepath):
                        file_age = current_time - os.path.getctime(filepath)
                        if file_age > FILE_MAX_AGE_SECONDS:
                            os.remove(filepath)
                            cleaned_count += 1
                            logger.debug(f"Cleaned up old file: {filename} (age: {file_age:.1f}s)")
                except OSError as e:
                    error_count += 1
                    logger.warning(f"Could not process file {filename}: {e}")
            
            if cleaned_count > 0 or error_count > 0:
                logger.info(f"Cleanup completed: {cleaned_count} files removed, {error_count} errors")
            
            # Trigger garbage collection periodically
            gc.collect()
                                
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
        
        time.sleep(CLEANUP_INTERVAL_SECONDS)

def initialize_application() -> bool:
    """Enhanced application initialization with comprehensive error handling."""
    try:
        logger.info("Starting application initialization...")
        
        # Get configuration from environment
        DATABASE_URL = os.environ.get(
            'DATABASE_URL',
            'postgresql+psycopg2://username:password@localhost:5432/product_db'
        )
        
        MODEL_NAME = os.environ.get('MODEL_NAME', 'resnet18')
        DEVICE = os.environ.get('TORCH_DEVICE', 'auto')
        
        logger.info(f"Initializing ProductIdentifier with model: {MODEL_NAME}, device: {DEVICE}")
        
        # Initialize ProductIdentifier
        product_identifier = ProductIdentifier(
            db_url=DATABASE_URL,
            model_name=MODEL_NAME,
            device=DEVICE
        )
        
        # Store in application state
        app_state.product_identifier = product_identifier
        
        # Create dummy images for testing
        product_identifier.create_dummy_images()
        
        # Check if database needs initialization
        stats = product_identifier.get_database_stats()
        total_products = stats.get('total_products', 0)
        
        logger.info(f"Database contains {total_products} products")
        
        if total_products == 0:
            logger.info("Initializing database with sample data...")
            sample_products = [
                {
                    'image_path': 'blue_shirt.png',
                    'details': {
                        'name': 'Classic Blue Cotton Shirt',
                        'price': 25.00,
                        'brand': 'Fashion Co.',
                        'category': 'Shirts',
                        'link': 'https://example.com/blue-shirt',
                        'rating': 4.2,
                        'reviews_count': 127,
                        'availability': 'In Stock',
                        'description': 'Classic blue cotton shirt perfect for any occasion'
                    }
                },
                {
                    'image_path': 'red_shirt.png',
                    'details': {
                        'name': 'Vibrant Red Casual Shirt',
                        'price': 30.00,
                        'brand': 'Style Inc.',
                        'category': 'Shirts',
                        'link': 'https://example.com/red-shirt',
                        'rating': 4.5,
                        'reviews_count': 89,
                        'availability': 'In Stock',
                        'description': 'Bold red shirt perfect for casual outings'
                    }
                },
                {
                    'image_path': 'yellow_shirt.png',
                    'details': {
                        'name': 'Sunny Yellow T-Shirt',
                        'price': 22.50,
                        'brand': 'Trendy Wear',
                        'category': 'T-Shirts',
                        'link': 'https://example.com/yellow-shirt',
                        'rating': 4.0,
                        'reviews_count': 156,
                        'availability': 'Limited Stock',
                        'description': 'Bright yellow t-shirt for summer days'
                    }
                }
            ]
            
            product_identifier.populate_database(sample_products)
            logger.info("Sample data populated successfully")
        
        # Test the system
        model_info = product_identifier.get_model_info()
        logger.info(f"System initialized - Model: {model_info.get('model_name')}, "
                   f"Device: {model_info.get('device')}, "
                   f"DB Connected: {model_info.get('database_connected', False)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}", exc_info=True)
        return False

# Enhanced error handlers
@app.errorhandler(400)
def bad_request(e):
    """Handle bad request errors with detailed logging."""
    logger.warning(f"Bad request from {get_client_ip()}: {request.url}")
    return jsonify({
        'error': 'Bad Request',
        'message': 'The request was invalid or malformed',
        'timestamp': datetime.utcnow().isoformat()
    }), 400

@app.errorhandler(413)
def too_large(e):
    """Handle file too large errors."""
    logger.warning(f"File too large from {get_client_ip()}")
    return jsonify({
        'error': 'File too large',
        'message': 'File size exceeds 16MB limit',
        'max_size': '16MB'
    }), 413

@app.errorhandler(429)
def rate_limit_exceeded(e):
    """Handle rate limit exceeded errors."""
    client_ip = get_client_ip()
    logger.warning(f"Rate limit exceeded for {client_ip}")
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': f'Maximum {RATE_LIMIT_PER_MINUTE} requests per minute allowed',
        'retry_after': 60
    }), 429

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors with detailed logging."""
    error_id = str(uuid.uuid4())[:8]
    logger.error(f"Internal server error [{error_id}]: {e}", exc_info=True)
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred',
        'error_id': error_id,
        'timestamp': datetime.utcnow().isoformat()
    }), 500

@app.errorhandler(503)
def service_unavailable(e):
    """Handle service unavailable errors."""
    return jsonify({
        'error': 'Service unavailable',
        'message': 'The product identification service is temporarily unavailable',
        'timestamp': datetime.utcnow().isoformat()
    }), 503

# Enhanced middleware
@app.before_request
def before_request():
    """Enhanced pre-request processing."""
    g.request_start_time = time.time()
    g.client_ip = get_client_ip()
    
    # Skip checks for health endpoint
    if request.endpoint == 'health':
        return None
    
    # Rate limiting
    if not check_rate_limit(g.client_ip):
        abort(429)
    
    # Service availability check
    if app_state.product_identifier is None and request.endpoint not in ['health', 'home']:
        abort(503)

@app.after_request
def after_request(response):
    """Enhanced post-request processing."""
    if hasattr(g, 'request_start_time'):
        processing_time = time.time() - g.request_start_time
        response.headers['X-Processing-Time'] = f'{processing_time:.3f}s'
        
        # Update stats
        if processing_time > 0:
            app_state.update_stats('total_processing_time', processing_time)
    
    # Security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    return response

# Enhanced API Routes
@app.route('/')
def home():
    """Enhanced API information and documentation."""
    uptime = None
    if app_state.product_identifier:
        init_time = app_state._initialization_time
        if init_time:
            uptime = str(datetime.utcnow() - init_time)
    
    return jsonify({
        'name': 'Product Identification API',
        'version': '2.0.0',
        'description': 'Advanced visual product similarity search using deep learning and PostgreSQL',
        'status': 'operational' if app_state.product_identifier else 'limited',
        'uptime': uptime,
        'features': [
            'Deep learning feature extraction (ResNet18/50)',
            'Vector similarity search with pgvector',
            'PostgreSQL persistence with indexing',
            'Real-time image processing',
            'Multi-metric similarity search',
            'Category and brand filtering',
            'Rate limiting and security',
            'Comprehensive monitoring'
        ],
        'endpoints': {
            '/': 'API documentation',
            '/health': 'Comprehensive health check',
            '/stats': 'Detailed system statistics',
            '/categories': 'Available product categories',
            '/brands': 'Available product brands',
            '/search': 'Visual similarity search (POST)',
            '/populate': 'Bulk product upload (POST)',
            '/products': 'Product management (GET)'
        },
        'limits': {
            'max_file_size': '16MB',
            'rate_limit': f'{RATE_LIMIT_PER_MINUTE} requests/minute',
            'max_results': 50,
            'supported_formats': list(ALLOWED_EXTENSIONS),
            'max_products_per_batch': MAX_PRODUCTS_PER_BATCH
        },
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/health')
def health():
    """Comprehensive health check with detailed system information."""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'components': {
            'api': True,
            'product_identifier': app_state.product_identifier is not None,
            'database': False,
            'model': False,
            'upload_directory': os.path.exists(app.config['UPLOAD_FOLDER'])
        },
        'system_info': {},
        'performance': app_state.get_stats()
    }
    
    identifier = app_state.product_identifier
    if identifier:
        try:
            # Test database connection
            health_status['components']['database'] = identifier.db_manager.test_connection()
            
            # Test model (always True if identifier exists)
            health_status['components']['model'] = True
            
            # Get detailed system info
            model_info = identifier.get_model_info()
            db_stats = identifier.get_database_stats()
            
            health_status['system_info'] = {
                'model_name': model_info.get('model_name'),
                'device': model_info.get('device'),
                'feature_dimension': model_info.get('feature_dimension'),
                'database_type': 'PostgreSQL with pgvector',
                'total_products': db_stats.get('total_products', 0),
                'products_with_features': db_stats.get('products_with_features', 0),
                'extraction_rate': db_stats.get('extraction_rate_percent', 0),
                'unique_categories': db_stats.get('unique_categories', 0),
                'unique_brands': db_stats.get('unique_brands', 0)
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status['components']['database'] = False
            health_status['components']['model'] = False
            health_status['error'] = str(e)
    
    # Overall health determination
    critical_components = ['api', 'product_identifier', 'database', 'model']
    all_healthy = all(health_status['components'].get(comp, False) for comp in critical_components)
    health_status['status'] = 'healthy' if all_healthy else 'unhealthy'
    
    return jsonify(health_status), 200 if all_healthy else 503

@app.route('/stats')
def get_stats():
    """Get comprehensive system statistics."""
    try:
        request_counts = app_state.get_request_counts()
        api_stats = {
            'total_requests_last_minute': sum(len(reqs) for reqs in request_counts.values()),
            'active_clients': len([ip for ip, reqs in request_counts.items() if reqs]),
            'rate_limit': f'{RATE_LIMIT_PER_MINUTE} requests/minute',
            'performance': app_state.get_stats()
        }
        
        response_data = {'api_stats': api_stats}
        
        identifier = app_state.product_identifier
        if identifier:
            response_data['database_stats'] = identifier.get_database_stats()
            response_data['model_info'] = identifier.get_model_info()
        
        # Add system resource info
        try:
            import psutil
            response_data['system_resources'] = {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }
        except ImportError:
            response_data['system_resources'] = {'note': 'psutil not available'}
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': 'Failed to retrieve statistics'}), 500

@app.route('/categories')
def get_categories():
    """Get all available product categories with caching."""
    try:
        identifier = app_state.product_identifier
        categories = identifier.get_categories()
        
        return jsonify({
            'categories': sorted(categories),  # Sort for consistency
            'total_count': len(categories),
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        return jsonify({'error': 'Failed to retrieve categories'}), 500

@app.route('/brands')
def get_brands():
    """Get all available product brands with caching."""
    try:
        identifier = app_state.product_identifier
        brands = identifier.get_brands()
        
        return jsonify({
            'brands': sorted(brands),  # Sort for consistency
            'total_count': len(brands),
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting brands: {e}")
        return jsonify({'error': 'Failed to retrieve brands'}), 500

@app.route('/search', methods=['POST'])
def search():
    """Enhanced visual similarity search endpoint with improved error handling."""
    search_start_time = time.time()
    temp_filepath = None
    
    try:
        # Validate image file presence
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image provided',
                'message': 'Please provide an image file with the "image" parameter'
            }), 400

        file = request.files['image']
        if not file or file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Please select a file to upload'
            }), 400

        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'message': f'Supported formats: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400

        # Validate and parse search parameters
        params, error_msg = validate_search_parameters(request.form)
        if error_msg:
            return jsonify({
                'error': 'Invalid parameters',
                'message': error_msg
            }), 400

        # Generate secure temporary filename
        original_filename = secure_filename(file.filename)
        file_extension = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else 'tmp'
        temp_filename = f"{uuid.uuid4().hex}.{file_extension}"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        # Use context manager for safe file handling
        with safe_file_operation(temp_filepath):
            # Save uploaded file
            try:
                file.save(temp_filepath)
                logger.info(f"Processing uploaded image: {temp_filename} from {g.client_ip}")
            except Exception as e:
                logger.error(f"Failed to save uploaded file: {e}")
                return jsonify({
                    'error': 'File save failed',
                    'message': 'Could not save the uploaded file'
                }), 500
            
            # Validate the saved image
            identifier = app_state.product_identifier
            if not identifier.validate_image(temp_filepath):
                return jsonify({
                    'error': 'Invalid image',
                    'message': 'The uploaded file is not a valid image or is corrupted'
                }), 400
            
            # Extract features
            feature_start_time = time.time()
            query_features = identifier.get_image_features(temp_filepath)
            feature_time = time.time() - feature_start_time
            
            if query_features is None:
                app_state.update_stats('failed_searches')
                return jsonify({
                    'error': 'Feature extraction failed',
                    'message': 'Could not extract features from the uploaded image'
                }), 500

            # Perform similarity search
            search_start = time.time()
            search_results = identifier.find_similar_products(**params, query_features=query_features)
            search_time = time.time() - search_start
            
            total_processing_time = time.time() - search_start_time
            
            # Update statistics
            app_state.update_stats('successful_searches')
            
            # Prepare enhanced response
            response = {
                'success': True,
                'results_count': len(search_results),
                'processing_times': {
                    'total_seconds': round(total_processing_time, 3),
                    'feature_extraction_seconds': round(feature_time, 3),
                    'similarity_search_seconds': round(search_time, 3)
                },
                'results': search_results,
                'search_parameters': params,
                'metadata': {
                    'original_filename': original_filename,
                    'query_timestamp': datetime.utcnow().isoformat(),
                    'client_ip': g.client_ip
                }
            }
            
            logger.info(f"Search completed in {total_processing_time:.3f}s, "
                       f"found {len(search_results)} results for {g.client_ip}")
            
            return jsonify(response)

    except RequestEntityTooLarge:
        logger.warning(f"File too large uploaded by {g.client_ip}")
        abort(413)
        
    except ValueError as e:
        logger.warning(f"Invalid parameters from {g.client_ip}: {e}")
        return jsonify({
            'error': 'Invalid parameter',
            'message': str(e)
        }), 400
        
    except Exception as e:
        app_state.update_stats('failed_searches')
        logger.error(f"Search error for {g.client_ip}: {e}", exc_info=True)
        return jsonify({
            'error': 'Search failed',
            'message': 'An error occurred during the search process',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/populate', methods=['POST'])
def populate():
    """Enhanced bulk product upload endpoint."""
    try:
        # Validate content type
        if not request.is_json:
            return jsonify({
                'error': 'Invalid content type',
                'message': 'Request must be JSON'
            }), 400
        
        data = request.get_json()
        if not data or 'products' not in data:
            return jsonify({
                'error': 'Missing products data',
                'message': 'Request must contain a "products" array'
            }), 400
        
        products_data = data['products']
        
        # Validate data structure
        if not isinstance(products_data, list):
            return jsonify({
                'error': 'Invalid products format',
                'message': '"products" must be an array'
            }), 400
        
        if len(products_data) == 0:
            return jsonify({
                'error': 'Empty products list',
                'message': 'At least one product is required'
            }), 400
        
        if len(products_data) > MAX_PRODUCTS_PER_BATCH:
            return jsonify({
                'error': 'Too many products',
                'message': f'Maximum {MAX_PRODUCTS_PER_BATCH} products per batch'
            }), 400
        
        # Validate product structure
        required_fields = ['image_path', 'details']
        for i, product in enumerate(products_data):
            if not isinstance(product, dict):
                return jsonify({
                    'error': f'Invalid product format at index {i}',
                    'message': 'Each product must be an object'
                }), 400
            
            for field in required_fields:
                if field not in product:
                    return jsonify({
                        'error': f'Missing field in product {i}',
                        'message': f'Product at index {i} is missing required field: {field}'
                    }), 400
            
            # Validate details structure
            if not isinstance(product['details'], dict):
                return jsonify({
                    'error': f'Invalid details format in product {i}',
                    'message': 'Product details must be an object'
                }), 400
        
        # Process products
        logger.info(f"Starting bulk population of {len(products_data)} products from {g.client_ip}")
        population_start_time = time.time()
        
        identifier = app_state.product_identifier
        processed_products = identifier.populate_database(products_data)
        
        processing_time = time.time() - population_start_time
        
        # Prepare response
        response = {
            'success': True,
            'products_processed': len(processed_products),
            'processing_time_seconds': round(processing_time, 3),
            'summary': {
                'total_submitted': len(products_data),
                'successfully_processed': len(processed_products),
                'processing_rate': round(len(processed_products) / processing_time, 2) if processing_time > 0 else 0
            },
            'sample_products': [p.to_dict() for p in processed_products[-5:]],  # Last 5 for verification
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Population completed: {len(processed_products)} products processed in {processing_time:.2f}s")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Population error from {g.client_ip}: {e}", exc_info=True)
        return jsonify({
            'error': 'Population failed',
            'message': 'An error occurred during bulk upload',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/products', methods=['GET'])
def get_products():
    """Enhanced product listing endpoint with pagination support."""
    try:
        # Parse query parameters with validation
        try:
            limit = min(max(int(request.args.get('limit', 50)), 1), 100)
            offset = max(int(request.args.get('offset', 0)), 0)
        except ValueError:
            return jsonify({
                'error': 'Invalid parameters',
                'message': 'limit and offset must be valid integers'
            }), 400
        
        category = request.args.get('category', '').strip() or None
        brand = request.args.get('brand', '').strip() or None
        
        identifier = app_state.product_identifier
        stats = identifier.get_database_stats()
        
        # For this version, return comprehensive metadata
        # In a full implementation, you'd fetch actual product records
        response = {
            'message': 'Product listing endpoint',
            'note': 'Full product listing with pagination will be implemented in next version',
            'pagination': {
                'limit': limit,
                'offset': offset,
                'total_available': stats.get('total_products', 0)
            },
            'filters': {
                'category': category,
                'brand': brand
            },
            'database_stats': stats,
            'available_filters': {
                'categories': identifier.get_categories(),
                'brands': identifier.get_brands()
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in product listing: {e}")
        return jsonify({'error': 'Failed to retrieve product information'}), 500

# Application factory with enhanced initialization
def create_app() -> Flask:
    """Create and configure Flask application with all enhancements."""
    logger.info("Creating Flask application...")
    
    # Start background cleanup task
    cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
    cleanup_thread.start()
    app_state._cleanup_thread = cleanup_thread
    logger.info("Started background file cleanup task")
    
    # Initialize the core application
    initialization_success = initialize_application()
    
    if initialization_success:
        logger.info("🚀 Flask application initialized successfully!")
    else:
        logger.error("⚠️ Flask application started with limited functionality")
        logger.error("Please check your database configuration and dependencies")
    
    return app

# Production WSGI entry point
def create_production_app():
    """Create production-ready Flask application."""
    # Set production configurations
    app.config.update({
        'DEBUG': False,
        'TESTING': False,
        'ENV': 'production'
    })
    
    # Enhanced logging for production
    if not app.debug:
        from logging.handlers import SysLogHandler
        syslog_handler = SysLogHandler()
        syslog_handler.setLevel(logging.WARNING)
        app.logger.addHandler(syslog_handler)
    
    return create_app()

if __name__ == '__main__':
    # Development server with enhanced configuration
    app = create_app()
    
    # Get configuration from environment
    HOST = os.environ.get('FLASK_HOST', '127.0.0.1')
    PORT = int(os.environ.get('FLASK_PORT', '5000'))
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() in ('true', '1', 'yes')
    
    logger.info(f"Starting Flask development server on {HOST}:{PORT} (debug={DEBUG})")
    
    app.run(
        host=HOST,
        port=PORT,
        debug=DEBUG,
        threaded=True,
        use_reloader=DEBUG
    )