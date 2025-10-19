import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from scipy.spatial.distance import cosine
import cv2
import logging
from typing import List, Dict, Optional, Union, Tuple, Any
import time
import tempfile
from pathlib import Path

# Database imports
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from datetime import datetime
import psycopg2
from urllib.parse import urlparse

# Set up logging with more detailed configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('product_identifier.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Database Setup
Base = declarative_base()

class Product(Base):
    """
    Enhanced database model for a product, including its deep learning features and metadata.
    """
    __tablename__ = 'products'
    
    # Primary Key and Indexing columns
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_path = Column(String(500), unique=True, nullable=False, index=True)
    
    # Features (Stored as a PostgreSQL array for NumPy compatibility)
    features = Column(ARRAY(Float(precision=32), dimensions=1), nullable=True)
    feature_dim = Column(Integer, nullable=True)
    
    # Product Details
    name = Column(String(255), nullable=False, index=True)
    price = Column(Float, nullable=True)
    link = Column(Text, nullable=True)
    brand = Column(String(100), nullable=True, index=True)
    category = Column(String(100), nullable=True, index=True)
    rating = Column(Float, nullable=True)
    availability = Column(String(50), nullable=True)
    reviews_count = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)
    
    # Metadata fields
    feature_extraction_success = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Product(id={self.id}, name='{self.name}', brand='{self.brand}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Product object to dictionary for API responses."""
        return {
            'id': self.id,
            'name': self.name,
            'price': self.price,
            'link': self.link,
            'brand': self.brand,
            'category': self.category,
            'rating': self.rating,
            'availability': self.availability,
            'reviews_count': self.reviews_count,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class DatabaseManager:
    """
    Enhanced database manager with comprehensive error handling and connection management.
    """
    
    def __init__(self, db_url: str, pool_size: int = 10, max_overflow: int = 20):
        """
        Initialize the database manager.
        
        Args:
            db_url (str): Database connection URL
            pool_size (int): Size of the connection pool
            max_overflow (int): Maximum overflow connections
        """
        self.db_url = db_url
        self.engine = None
        self.Session = None
        self._initialize_database(pool_size, max_overflow)
    
    def _initialize_database(self, pool_size: int, max_overflow: int) -> None:
        """Initialize database connection and create tables."""
        try:
            # Parse database URL to extract components
            parsed_url = urlparse(self.db_url)
            
            logger.info(f"Connecting to PostgreSQL database: {parsed_url.hostname}:{parsed_url.port}/{parsed_url.path[1:]}")
            
            # Create engine with connection pooling
            self.engine = create_engine(
                self.db_url,
                echo=False,  # Set to True for SQL debugging
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=3600,   # Recycle connections every hour
                connect_args={
                    "options": "-c timezone=UTC",
                    "connect_timeout": 30
                }
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            
            # Create all tables
            Base.metadata.create_all(self.engine)
            
            # Create session factory
            self.Session = sessionmaker(bind=self.engine)
            
            logger.info("Database connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise RuntimeError(f"Database initialization failed: {e}")
    
    def get_session(self):
        """Returns a new database session with error handling."""
        if not self.Session:
            raise RuntimeError("Database not initialized")
        return self.Session()
    
    def test_connection(self) -> bool:
        """Test database connectivity."""
        try:
            session = self.get_session()
            session.execute("SELECT 1")
            session.close()
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def product_exists(self, session, image_path: str) -> bool:
        """Check if a product with the given image path already exists."""
        try:
            return session.query(Product).filter_by(image_path=image_path).first() is not None
        except SQLAlchemyError as e:
            logger.error(f"Error checking product existence: {e}")
            return False
    
    def get_all_products(self, session, limit: Optional[int] = None) -> List[Product]:
        """
        Fetch all product objects from the database.
        
        Args:
            session: Database session
            limit (Optional[int]): Maximum number of products to return
            
        Returns:
            List[Product]: List of Product objects
        """
        try:
            query = session.query(Product).filter(Product.features.isnot(None))
            if limit:
                query = query.limit(limit)
            return query.all()
        except SQLAlchemyError as e:
            logger.error(f"Error fetching products: {e}")
            return []
    
    def get_products_by_category(self, session, category: str) -> List[Product]:
        """Get products filtered by category."""
        try:
            return session.query(Product).filter(
                Product.category == category,
                Product.features.isnot(None)
            ).all()
        except SQLAlchemyError as e:
            logger.error(f"Error fetching products by category: {e}")
            return []
    
    def get_database_stats(self, session) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            total_products = session.query(Product).count()
            products_with_features = session.query(Product).filter(Product.features.isnot(None)).count()
            categories = session.query(Product.category).distinct().all()
            brands = session.query(Product.brand).distinct().all()
            
            return {
                'total_products': total_products,
                'products_with_features': products_with_features,
                'unique_categories': len([c[0] for c in categories if c[0]]),
                'unique_brands': len([b[0] for b in brands if b[0]]),
                'feature_extraction_rate': (products_with_features / total_products * 100) if total_products > 0 else 0
            }
        except SQLAlchemyError as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

class ProductIdentifier:
    """
    Enhanced ProductIdentifier class with PostgreSQL integration for finding similar products 
    using ResNet50 feature extraction.
    """
    
    def __init__(self, device: str = None, model_name: str = 'resnet50', db_url: str = None):
        """
        Initialize the product identifier with ResNet50 model and PostgreSQL database.
        
        Args:
            device (str): Device to run the model on ('cpu', 'cuda', or 'auto')
            model_name (str): Model architecture to use (currently supports 'resnet50')
            db_url (str): PostgreSQL database URL
        """
        # Model initialization
        self.device = self._get_device(device)
        self.model_name = model_name
        self.model = None
        self.transform = None
        self.feature_dim = 2048  # ResNet50 feature dimension
        
        # Database initialization
        self.db_url = db_url or os.environ.get(
            'DATABASE_URL',
            'postgresql+psycopg2://user:password@localhost:5432/product_db'
        )
        
        logger.info(f"Initializing ProductIdentifier with {model_name} on {self.device}")
        
        # Initialize components
        self.load_model()
        self.db_manager = DatabaseManager(self.db_url)
        
        # Test database connection
        if not self.db_manager.test_connection():
            logger.warning("Database connection test failed, some features may be limited")
    
    def _get_device(self, device: str = None) -> str:
        """Determine the best device to use for inference."""
        if device == 'auto' or device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                logger.info("Using CPU for inference")
        elif device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = 'cpu'
        
        return device
    
    def load_model(self) -> None:
        """Load and configure the ResNet50 model for feature extraction."""
        try:
            logger.info(f"Loading {self.model_name} model...")
            
            # Load the pre-trained ResNet50 model
            if self.model_name == 'resnet50':
                self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                self.feature_dim = 2048
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            # Remove the final classification layer to get features
            self.model.fc = nn.Identity()
            self.model.to(self.device)
            self.model.eval()
            
            # Define image preprocessing transformations
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            # Warm up the model with a dummy input
            self._warmup_model()
            
            logger.info(f"Model loaded successfully! Feature dimension: {self.feature_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")

    def _warmup_model(self) -> None:
        """Warm up the model with a dummy input for better performance."""
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
            logger.debug("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def validate_image(self, image_path: str) -> bool:
        """
        Comprehensive image validation.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            bool: True if image is valid and processable
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return False
            
            # Check file size (max 50MB)
            file_size = os.path.getsize(image_path)
            max_size = 50 * 1024 * 1024  # 50MB
            if file_size > max_size:
                logger.error(f"Image file too large: {file_size / (1024*1024):.1f}MB > {max_size / (1024*1024)}MB")
                return False
            
            if file_size == 0:
                logger.error("Image file is empty")
                return False
            
            # Validate image format and integrity
            try:
                with Image.open(image_path) as img:
                    img.verify()  # Verify the image integrity
            except Exception as e:
                logger.error(f"Image verification failed: {e}")
                return False
            
            # Re-open for additional checks (verify closes the image)
            try:
                with Image.open(image_path) as img:
                    # Check dimensions
                    if img.size[0] == 0 or img.size[1] == 0:
                        logger.error(f"Invalid image dimensions: {img.size}")
                        return False
                    
                    # Check if dimensions are reasonable (not too small or too large)
                    min_dim, max_dim = 10, 10000
                    if img.size[0] < min_dim or img.size[1] < min_dim:
                        logger.error(f"Image too small: {img.size}")
                        return False
                    
                    if img.size[0] > max_dim or img.size[1] > max_dim:
                        logger.error(f"Image too large: {img.size}")
                        return False
                    
                    # Check color mode
                    if img.mode not in ['RGB', 'RGBA', 'L', 'P']:
                        logger.warning(f"Unusual image mode: {img.mode}, will attempt to convert")
                    
            except Exception as e:
                logger.error(f"Image format validation failed: {e}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Image validation error: {e}")
            return False

    def get_image_features(self, image_path: str, normalize: bool = True) -> Optional[np.ndarray]:
        """
        Extract features from an image using the pre-trained ResNet50 model.
        
        Args:
            image_path (str): Path to the image file
            normalize (bool): Whether to normalize the feature vector
            
        Returns:
            Optional[np.ndarray]: Feature vector or None if extraction fails
        """
        start_time = time.time()
        
        try:
            if not self.validate_image(image_path):
                return None
            
            # Load and preprocess the image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                original_mode = image.mode
                image = image.convert('RGB')
                logger.debug(f"Converted image from {original_mode} to RGB")
            
            # Apply transformations
            image_tensor = self.transform(image)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)
            
            # Convert to numpy
            features_np = features.squeeze().cpu().numpy()
            
            # Normalize features if requested
            if normalize:
                norm = np.linalg.norm(features_np)
                if norm > 0:
                    features_np = features_np / norm
            
            processing_time = time.time() - start_time
            logger.debug(f"Feature extraction completed in {processing_time:.3f}s, shape: {features_np.shape}")
            
            return features_np
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None

    def populate_database(self, product_dicts: List[Dict]) -> List[Product]:
        """
        Populate the PostgreSQL database with products and their features.

        Args:
            product_dicts (List[Dict]): List of product dictionaries (metadata)

        Returns:
            List[Product]: List of Product SQLAlchemy objects loaded/saved to the DB.
        """
        logger.info("Starting database population/feature extraction for persistence...")
        start_time = time.time()

        session = self.db_manager.get_session()
        successful_extractions = 0
        skipped_existing = 0

        try:
            for product_dict in product_dicts:
                image_path = product_dict['image_path']
                product_details = product_dict['details']

                # Check if product already exists (to avoid re-extraction on restart)
                if self.db_manager.product_exists(session, image_path):
                    logger.debug(f"Product {image_path} already exists, skipping feature extraction.")
                    skipped_existing += 1
                    continue

                # Extract features
                features_np = self.get_image_features(image_path)

                if features_np is not None:
                    # Create the SQLAlchemy Product object
                    new_product = Product(
                        image_path=image_path,
                        features=features_np.tolist(),  # Convert numpy array to list for PostgreSQL ARRAY
                        feature_dim=len(features_np),
                        name=product_details.get('name', 'N/A'),
                        price=product_details.get('price'),
                        link=product_details.get('link'),
                        brand=product_details.get('brand'),
                        category=product_details.get('category'),
                        rating=product_details.get('rating'),
                        availability=product_details.get('availability', 'Unknown'),
                        reviews_count=product_details.get('reviews_count', product_details.get('reviews', 0)),
                        description=product_details.get('description'),
                        feature_extraction_success=True
                    )

                    # Add and commit to database
                    session.add(new_product)
                    successful_extractions += 1
                    logger.info(f"✓ Saved features for: {new_product.name}")
                else:
                    # Still save the product without features for completeness
                    failed_product = Product(
                        image_path=image_path,
                        features=None,
                        feature_dim=0,
                        name=product_details.get('name', 'N/A'),
                        price=product_details.get('price'),
                        link=product_details.get('link'),
                        brand=product_details.get('brand'),
                        category=product_details.get('category'),
                        rating=product_details.get('rating'),
                        availability=product_details.get('availability', 'Unknown'),
                        reviews_count=product_details.get('reviews_count', product_details.get('reviews', 0)),
                        description=product_details.get('description'),
                        feature_extraction_success=False
                    )
                    session.add(failed_product)
                    logger.error(f"✗ Failed to extract features for {image_path}. Saved without features.")

            session.commit()

            total_time = time.time() - start_time
            logger.info(f"DB population completed: {successful_extractions} new products processed, "
                       f"{skipped_existing} existing products skipped in {total_time:.2f}s")

            # Return all products from the DB for use in the search function
            return self.db_manager.get_all_products(session)

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error during population: {e}")
            return []
        except Exception as e:
            session.rollback()
            logger.error(f"Unexpected error during database population: {e}")
            return []
        finally:
            session.close()

    def find_similar_products(
        self, 
        query_features: np.ndarray, 
        product_database: Optional[List[Product]] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_metric: str = 'cosine',
        category_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Find the most similar products in the database based on their features.
        
        Args:
            query_features (np.ndarray): Feature vector of the query image
            product_database (Optional[List[Product]]): List of Product objects (if None, fetch from DB)
            top_k (int): Number of top results to return
            similarity_threshold (float): Minimum similarity score to include in results
            similarity_metric (str): Similarity metric to use ('cosine', 'euclidean', 'dot')
            category_filter (Optional[str]): Filter results by category
            
        Returns:
            List[Dict]: Sorted list of similar products with metadata
        """
        if query_features is None:
            logger.error("Query features are None")
            return []

        # If no product database provided, fetch from PostgreSQL
        if product_database is None:
            session = self.db_manager.get_session()
            try:
                if category_filter:
                    product_database = self.db_manager.get_products_by_category(session, category_filter)
                else:
                    product_database = self.db_manager.get_all_products(session)
            finally:
                session.close()

        if not product_database:
            logger.warning("Product database is empty")
            return []

        results = []
        valid_products = 0

        for product in product_database:
            # Check if features exist and are not empty
            if product.features is None or len(product.features) == 0:
                logger.debug(f"Product {product.id} has no features, skipping...")
                continue

            valid_products += 1

            try:
                # Convert feature list back to numpy array for comparison
                db_features_np = np.array(product.features)

                # Calculate similarity based on chosen metric
                if similarity_metric == 'cosine':
                    similarity = 1 - cosine(query_features, db_features_np)
                elif similarity_metric == 'euclidean':
                    # Convert euclidean distance to similarity (0-1 range)
                    distance = np.linalg.norm(query_features - db_features_np)
                    similarity = 1 / (1 + distance)
                elif similarity_metric == 'dot':
                    similarity = np.dot(query_features, db_features_np)
                else:
                    raise ValueError(f"Unsupported similarity metric: {similarity_metric}")

                # Only include results above the threshold
                if similarity >= similarity_threshold:
                    results.append({
                        'id': product.id,
                        'similarity': float(similarity),
                        'details': product.to_dict(),
                        'confidence': self._get_confidence_level(similarity),
                        'similarity_metric': similarity_metric
                    })

            except Exception as e:
                logger.error(f"Error computing similarity for product {product.id}: {e}")
                continue

        # Sort results from most similar to least similar
        results.sort(key=lambda x: x['similarity'], reverse=True)

        logger.info(f"Found {len(results)} similar products from {valid_products} valid products "
                   f"(threshold: {similarity_threshold})")

        return results[:top_k]

    def _get_confidence_level(self, similarity: float) -> str:
        """
        Convert similarity score to confidence level description.
        
        Args:
            similarity (float): Similarity score between 0 and 1
            
        Returns:
            str: Confidence level description
        """
        if similarity >= 0.95:
            return "Excellent Match"
        elif similarity >= 0.85:
            return "Very High"
        elif similarity >= 0.70:
            return "High" 
        elif similarity >= 0.50:
            return "Medium"
        elif similarity >= 0.30:
            return "Low"
        else:
            return "Very Low"

    def create_dummy_images(self, image_size: Tuple[int, int] = (300, 300)) -> None:
        """
        Create realistic dummy colored images for testing.
        
        Args:
            image_size (Tuple[int, int]): Size of the images to create (width, height)
        """
        logger.info(f"Creating dummy images with size {image_size}...")
        
        images_to_create = [
            ('blue_shirt.png', [255, 100, 100]),    # Blue with some variation
            ('red_shirt.png', [100, 100, 255]),     # Red with some variation  
            ('yellow_shirt.png', [100, 255, 255])   # Yellow with some variation
        ]
        
        for filename, base_color in images_to_create:
            if not os.path.exists(filename):
                try:
                    # Create a more realistic looking image with gradients and noise
                    img = np.full((image_size[1], image_size[0], 3), base_color, dtype=np.uint8)
                    
                    # Add gradient effect
                    for i in range(image_size[1]):
                        factor = 0.7 + 0.3 * (i / image_size[1])  # Gradient from 0.7 to 1.0
                        img[i, :] = np.clip(img[i, :] * factor, 0, 255).astype(np.uint8)
                    
                    # Add some texture/noise to make it more realistic
                    noise = np.random.randint(-30, 30, img.shape, dtype=np.int16)
                    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    
                    # Add some geometric patterns for more variety
                    center = (image_size[0] // 2, image_size[1] // 2)
                    cv2.circle(img, center, min(image_size) // 6, 
                              [int(c * 0.8) for c in base_color], -1)
                    
                    cv2.imwrite(filename, img)
                    logger.info(f"Created {filename} ({image_size[0]}x{image_size[1]})")
                    
                except Exception as e:
                    logger.error(f"Failed to create {filename}: {e}")
            else:
                logger.debug(f"{filename} already exists, skipping creation")

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        session = self.db_manager.get_session()
        try:
            stats = self.db_manager.get_database_stats(session)
            return stats
        finally:
            session.close()

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the loaded model and database."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'feature_dimension': self.feature_dim,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'cuda_available': torch.cuda.is_available(),
            'torch_version': torch.__version__,
            'database_url': self.db_url.split('@')[-1] if '@' in self.db_url else 'Not configured',
            'database_connected': self.db_manager.test_connection()
        }

    def batch_extract_features(self, image_paths: List[str], batch_size: int = 8) -> List[Optional[np.ndarray]]:
        """
        Extract features from multiple images in batches for improved efficiency.
        
        Args:
            image_paths (List[str]): List of image paths
            batch_size (int): Number of images to process in each batch
            
        Returns:
            List[Optional[np.ndarray]]: List of feature vectors (None for failed extractions)
        """
        logger.info(f"Batch extracting features for {len(image_paths)} images (batch_size={batch_size})")
        
        all_features = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_features = []
            
            for path in batch_paths:
                features = self.get_image_features(path)
                batch_features.append(features)
            
            all_features.extend(batch_features)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")
        
        return all_features

# Example usage and testing
def main():
    """Main function to demonstrate the enhanced product identification system with PostgreSQL."""
    try:
        # Initialize with automatic device selection and database URL
        db_url = os.environ.get(
            'DATABASE_URL',
            'postgresql+psycopg2://username:password@localhost:5432/product_db'
        )
        
        identifier = ProductIdentifier(device='auto', db_url=db_url)
        
        # Print model and database information
        model_info = identifier.get_model_info()
        logger.info(f"Model info: {model_info}")
        
        # Create dummy database with enhanced details
        dummy_database = [
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
                    'description': 'A classic blue cotton shirt perfect for any occasion'
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
        
        # Create dummy images
        identifier.create_dummy_images((400, 400))
        
        # Populate database with features
        products = identifier.populate_database(dummy_database)
        
        # Get database statistics
        stats = identifier.get_database_stats()
        logger.info(f"Database stats: {stats}")
        
        logger.info("Enhanced product identification system with PostgreSQL ready!")
        
        # Example similarity search
        query_features = identifier.get_image_features('blue_shirt.png')
        if query_features is not None:
            results = identifier.find_similar_products(
                query_features, 
                top_k=3,
                similarity_threshold=0.5
            )
            
            logger.info("Similar products found:")
            for i, result in enumerate(results, 1):
                logger.info(f"{i}. {result['details']['name']} - "
                           f"Similarity: {result['similarity']:.3f} "
                           f"({result['confidence']})")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()