"""
Professional Product Identification System with PostgreSQL and pgvector

This module provides a comprehensive solution for visual product similarity search
using deep learning feature extraction and efficient vector similarity search
with PostgreSQL and the pgvector extension.

Features:
- Deep learning feature extraction using pre-trained ResNet models
- Efficient vector similarity search with pgvector
- Comprehensive database management with SQLAlchemy
- Production-ready error handling and logging
- Scalable architecture for large product catalogs
"""

import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from contextlib import contextmanager
import hashlib

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Database imports
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, Boolean, Index
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy_utils import database_exists, create_database
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    raise ImportError("pgvector package is required. Install with: pip install pgvector")

# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_FEATURE_DIM = 512  # ResNet18 feature dimension
DEFAULT_IMAGE_SIZE = (224, 224)
MAX_IMAGE_SIZE_MB = 10
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

# SQLAlchemy base
Base = declarative_base()

class Product(Base):
    """
    SQLAlchemy model for product metadata with comprehensive fields and indexing.
    """
    __tablename__ = 'products'
    
    # Primary key and unique identifiers
    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # Product metadata
    name = Column(String(500), nullable=False, index=True)
    price = Column(Float, nullable=True)
    brand = Column(String(200), nullable=True, index=True)
    category = Column(String(200), nullable=True, index=True)
    link = Column(Text, nullable=True)
    rating = Column(Float, nullable=True)
    reviews_count = Column(Integer, default=0)
    availability = Column(String(100), default='Unknown')
    description = Column(Text, nullable=True)
    
    # Image and processing metadata
    image_path = Column(String(1000), nullable=True)
    image_hash = Column(String(64), nullable=True, index=True)  # For deduplication
    
    # Feature processing status
    feature_extracted = Column(Boolean, default=False, index=True)
    feature_extraction_error = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationship to feature vector
    feature_vector = relationship("FeatureVector", uselist=False, back_populates="product")
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_product_category_rating', 'category', 'rating'),
        Index('idx_product_brand_category', 'brand', 'category'),
        Index('idx_product_created_at', 'created_at'),
    )
    
    def to_dict(self, include_similarity: bool = False) -> Dict[str, Any]:
        """Convert product to dictionary for API responses."""
        data = {
            'id': self.id,
            'product_id': self.product_id,
            'name': self.name,
            'price': self.price,
            'brand': self.brand,
            'category': self.category,
            'link': self.link,
            'rating': self.rating,
            'reviews_count': self.reviews_count,
            'availability': self.availability,
            'description': self.description,
            'feature_extracted': self.feature_extracted,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
        
        # Include similarity score if available (added during search)
        if include_similarity and hasattr(self, 'similarity_score'):
            data['similarity_score'] = self.similarity_score
            data['confidence_level'] = self.confidence_level
        
        return data
    
    def __repr__(self):
        return f"<Product(id={self.id}, product_id='{self.product_id}', name='{self.name[:30]}...')>"

class FeatureVector(Base):
    """
    SQLAlchemy model for storing feature vectors with pgvector integration.
    """
    __tablename__ = 'feature_vectors'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String(255), nullable=False, unique=True, index=True)
    
    # Vector column using pgvector
    vector = Column(Vector(DEFAULT_FEATURE_DIM), nullable=False)
    vector_norm = Column(Float, nullable=True)  # Store L2 norm for optimization
    
    # Processing metadata
    extraction_method = Column(String(100), default='ResNet18')
    extraction_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationship back to product
    product = relationship("Product", back_populates="feature_vector")
    
    # Indexes for vector similarity search
    __table_args__ = (
        Index('idx_feature_vectors_product_id', 'product_id'),
        # Vector index will be created separately due to pgvector requirements
    )
    
    def __repr__(self):
        return f"<FeatureVector(id={self.id}, product_id='{self.product_id}')>"

class DatabaseManager:
    """
    Professional database manager with connection pooling, error handling, and monitoring.
    """
    
    def __init__(self, db_url: str, pool_size: int = 10, max_overflow: int = 20, echo: bool = False):
        """
        Initialize database manager with comprehensive configuration.
        
        Args:
            db_url: PostgreSQL database URL
            pool_size: Number of connections to maintain in pool
            max_overflow: Maximum number of connections beyond pool_size
            echo: Whether to echo SQL statements (for debugging)
        """
        self.db_url = db_url
        self.engine = None
        self.Session = None
        self._initialize_engine(pool_size, max_overflow, echo)
        self._initialize_database()
    
    def _initialize_engine(self, pool_size: int, max_overflow: int, echo: bool) -> None:
        """Initialize SQLAlchemy engine with connection pooling."""
        try:
            # Ensure database exists
            if not database_exists(self.db_url):
                logger.info("Creating database...")
                create_database(self.db_url)
            
            # Create engine with optimized settings
            self.engine = create_engine(
                self.db_url,
                echo=echo,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,  # Recycle connections every hour
                connect_args={
                    "options": "-c timezone=UTC",
                    "connect_timeout": 30
                }
            )
            
            # Create session factory
            self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
            
            logger.info(f"Database engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise RuntimeError(f"Database initialization failed: {e}")
    
    def _initialize_database(self) -> None:
        """Initialize database schema and pgvector extension."""
        try:
            with self.engine.connect() as conn:
                # Enable pgvector extension
                try:
                    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    conn.commit()
                    logger.info("pgvector extension enabled")
                except Exception as e:
                    logger.warning(f"Could not enable pgvector extension: {e}")
                
                # Create all tables
                Base.metadata.create_all(self.engine)
                
                # Create vector index for efficient similarity search
                try:
                    conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_feature_vectors_cosine 
                        ON feature_vectors USING hnsw (vector vector_cosine_ops) 
                        WITH (m = 16, ef_construction = 64)
                    """)
                    
                    conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_feature_vectors_l2 
                        ON feature_vectors USING hnsw (vector vector_l2_ops) 
                        WITH (m = 16, ef_construction = 64)
                    """)
                    
                    conn.commit()
                    logger.info("Vector indexes created successfully")
                    
                except Exception as e:
                    logger.warning(f"Could not create vector indexes: {e}")
                    
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """Context manager for database sessions with automatic cleanup."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test database connectivity."""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        try:
            with self.get_session() as session:
                # Basic counts
                total_products = session.query(Product).count()
                products_with_features = session.query(Product).filter(Product.feature_extracted == True).count()
                total_categories = session.query(Product.category).distinct().count()
                total_brands = session.query(Product.brand).distinct().count()
                
                # Recent activity
                recent_products = session.query(Product).filter(
                    Product.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                ).count()
                
                # Feature extraction rate
                extraction_rate = (products_with_features / total_products * 100) if total_products > 0 else 0
                
                return {
                    'total_products': total_products,
                    'products_with_features': products_with_features,
                    'extraction_rate_percent': round(extraction_rate, 2),
                    'unique_categories': total_categories,
                    'unique_brands': total_brands,
                    'products_added_today': recent_products,
                    'database_type': 'PostgreSQL with pgvector',
                    'last_updated': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get database statistics: {e}")
            return {'error': str(e)}

class FeatureExtractor:
    """
    Professional feature extraction class with multiple model support and optimization.
    """
    
    def __init__(self, model_name: str = 'resnet18', device: Optional[str] = None):
        """
        Initialize feature extractor with specified model.
        
        Args:
            model_name: Name of the model to use ('resnet18', 'resnet50', etc.)
            device: Device to use ('cpu', 'cuda', or None for auto-detection)
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.transform = None
        self.feature_dim = DEFAULT_FEATURE_DIM
        
        self._load_model()
        self._setup_transforms()
        self._warmup_model()
    
    def _get_device(self, device: Optional[str]) -> torch.device:
        """Determine optimal device for inference."""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = 'cpu'
        
        torch_device = torch.device(device)
        
        if torch_device.type == 'cuda':
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU for inference")
        
        return torch_device
    
    def _load_model(self) -> None:
        """Load and configure the feature extraction model."""
        try:
            if self.model_name == 'resnet18':
                self.model = models.resnet18(pretrained=True)
                self.feature_dim = 512
            elif self.model_name == 'resnet50':
                self.model = models.resnet50(pretrained=True)
                self.feature_dim = 2048
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            # Remove final classification layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            self.model.to(self.device)
            
            logger.info(f"Model {self.model_name} loaded successfully, feature_dim: {self.feature_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _setup_transforms(self) -> None:
        """Setup image preprocessing transforms."""
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _warmup_model(self) -> None:
        """Warmup model with dummy input for consistent performance."""
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
            logger.debug("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def extract_features(self, image_path: str, normalize: bool = True) -> Optional[np.ndarray]:
        """
        Extract feature vector from image.
        
        Args:
            image_path: Path to image file
            normalize: Whether to L2 normalize the features
            
        Returns:
            Feature vector as numpy array or None if extraction fails
        """
        start_time = time.time()
        
        try:
            # Validate image
            if not self._validate_image(image_path):
                return None
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.squeeze().cpu().numpy()
            
            # Normalize if requested
            if normalize:
                norm = np.linalg.norm(features)
                if norm > 0:
                    features = features / norm
            
            extraction_time = time.time() - start_time
            logger.debug(f"Feature extraction completed in {extraction_time:.3f}s, shape: {features.shape}")
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {image_path}: {e}")
            return None
    
    def _validate_image(self, image_path: str) -> bool:
        """Validate image file."""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return False
            
            # Check file size
            file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
            if file_size > MAX_IMAGE_SIZE_MB:
                logger.error(f"Image too large: {file_size:.1f}MB > {MAX_IMAGE_SIZE_MB}MB")
                return False
            
            # Check file extension
            _, ext = os.path.splitext(image_path.lower())
            if ext not in SUPPORTED_IMAGE_FORMATS:
                logger.error(f"Unsupported image format: {ext}")
                return False
            
            # Validate image integrity
            with Image.open(image_path) as img:
                img.verify()
            
            return True
            
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get feature extractor information."""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'feature_dimension': self.feature_dim,
            'supported_formats': list(SUPPORTED_IMAGE_FORMATS),
            'max_image_size_mb': MAX_IMAGE_SIZE_MB
        }

class ProductIdentifier:
    """
    Professional product identification system with comprehensive functionality.
    """
    
    def __init__(self, db_url: str, model_name: str = 'resnet18', device: Optional[str] = None):
        """
        Initialize product identifier.
        
        Args:
            db_url: PostgreSQL database URL
            model_name: Feature extraction model name
            device: Computing device to use
        """
        self.db_manager = DatabaseManager(db_url)
        self.feature_extractor = FeatureExtractor(model_name, device)
        
        logger.info("ProductIdentifier initialized successfully")
    
    def create_dummy_images(self, size: Tuple[int, int] = (400, 400)) -> None:
        """Create dummy images for testing purposes."""
        dummy_images = [
            ('blue_shirt.png', (0, 100, 255)),
            ('red_shirt.png', (255, 50, 50)),
            ('yellow_shirt.png', (255, 255, 0))
        ]
        
        for filename, color in dummy_images:
            if not os.path.exists(filename):
                try:
                    # Create image with some texture
                    img = Image.new('RGB', size, color=color)
                    # Add some noise for more realistic features
                    pixels = img.load()
                    for i in range(0, size[0], 10):
                        for j in range(0, size[1], 10):
                            noise = np.random.randint(-20, 20, 3)
                            new_color = tuple(np.clip(np.array(color) + noise, 0, 255))
                            for di in range(min(10, size[0] - i)):
                                for dj in range(min(10, size[1] - j)):
                                    pixels[i + di, j + dj] = new_color
                    
                    img.save(filename)
                    logger.info(f"Created dummy image: {filename}")
                except Exception as e:
                    logger.error(f"Failed to create dummy image {filename}: {e}")
    
    def _compute_image_hash(self, image_path: str) -> Optional[str]:
        """Compute MD5 hash of image for deduplication."""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to compute image hash: {e}")
            return None
    
    def populate_database(self, products_data: List[Dict[str, Any]]) -> List[Product]:
        """
        Populate database with products and extract features.
        
        Args:
            products_data: List of product dictionaries
            
        Returns:
            List of Product objects from database
        """
        logger.info(f"Starting database population with {len(products_data)} products...")
        start_time = time.time()
        
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        try:
            with self.db_manager.get_session() as session:
                for item in products_data:
                    try:
                        image_path = item['image_path']
                        details = item['details']
                        
                        # Generate product_id if not provided
                        product_id = details.get('product_id', os.path.splitext(os.path.basename(image_path))[0])
                        
                        # Check if product already exists
                        existing_product = session.query(Product).filter_by(product_id=product_id).first()
                        
                        if existing_product and existing_product.feature_extracted:
                            logger.debug(f"Product {product_id} already processed, skipping")
                            skipped_count += 1
                            continue
                        
                        # Validate image exists
                        if not os.path.exists(image_path):
                            logger.warning(f"Image not found: {image_path}")
                            error_count += 1
                            continue
                        
                        # Compute image hash for deduplication
                        image_hash = self._compute_image_hash(image_path)
                        
                        # Extract features
                        features = self.feature_extractor.extract_features(image_path)
                        
                        if features is None:
                            logger.error(f"Feature extraction failed for {product_id}")
                            error_count += 1
                            continue
                        
                        # Create or update product
                        if existing_product:
                            product = existing_product
                        else:
                            product = Product(product_id=product_id)
                        
                        # Update product details
                        for key, value in details.items():
                            if hasattr(product, key) and key != 'product_id':
                                setattr(product, key, value)
                        
                        product.image_path = image_path
                        product.image_hash = image_hash
                        product.feature_extracted = True
                        product.feature_extraction_error = None
                        product.updated_at = datetime.utcnow()
                        
                        # Create or update feature vector
                        feature_vector = session.query(FeatureVector).filter_by(product_id=product_id).first()
                        if feature_vector:
                            feature_vector.vector = features.tolist()
                            feature_vector.extraction_timestamp = datetime.utcnow()
                        else:
                            feature_vector = FeatureVector(
                                product_id=product_id,
                                vector=features.tolist(),
                                vector_norm=float(np.linalg.norm(features)),
                                extraction_method=self.feature_extractor.model_name
                            )
                        
                        # Add to session
                        session.merge(product)
                        session.merge(feature_vector)
                        
                        processed_count += 1
                        logger.info(f"Processed product: {product_id}")
                        
                    except Exception as e:
                        logger.error(f"Error processing product: {e}")
                        error_count += 1
                        continue
            
            # Fetch all products for return
            with self.db_manager.get_session() as session:
                products = session.query(Product).all()
            
            total_time = time.time() - start_time
            logger.info(f"Database population completed in {total_time:.2f}s: "
                       f"{processed_count} processed, {skipped_count} skipped, {error_count} errors")
            
            return products
            
        except Exception as e:
            logger.error(f"Database population failed: {e}")
            raise
    
    def find_similar_products(
        self,
        query_features: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.1,
        similarity_metric: str = 'cosine',
        category_filter: Optional[str] = None,
        brand_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find similar products using vector similarity search.
        
        Args:
            query_features: Feature vector of query image
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            similarity_metric: Distance metric ('cosine' or 'l2')
            category_filter: Filter by category
            brand_filter: Filter by brand
            
        Returns:
            List of similar products with similarity scores
        """
        logger.info(f"Starting similarity search with {similarity_metric} metric, top_k={top_k}")
        start_time = time.time()
        
        try:
            with self.db_manager.get_session() as session:
                # Build base query
                query = session.query(Product, FeatureVector).join(
                    FeatureVector, Product.product_id == FeatureVector.product_id
                ).filter(Product.feature_extracted == True)
                
                # Apply filters
                if category_filter:
                    query = query.filter(Product.category == category_filter)
                if brand_filter:
                    query = query.filter(Product.brand == brand_filter)
                
                # Choose distance operator based on metric
                if similarity_metric == 'cosine':
                    distance_op = FeatureVector.vector.cosine_distance(query_features.tolist())
                else:  # l2
                    distance_op = FeatureVector.vector.l2_distance(query_features.tolist())
                
                # Order by distance and limit results
                query = query.order_by(distance_op).limit(top_k * 2)  # Get more for filtering
                
                results = []
                for product, feature_vector in query.all():
                    # Calculate similarity score
                    if similarity_metric == 'cosine':
                        # For cosine distance: similarity = 1 - distance
                        distance = np.dot(query_features, np.array(feature_vector.vector)) / (
                            np.linalg.norm(query_features) * np.linalg.norm(feature_vector.vector)
                        )
                        similarity = max(0, distance)  # Ensure non-negative
                    else:  # l2
                        distance = np.linalg.norm(query_features - np.array(feature_vector.vector))
                        similarity = 1 / (1 + distance)
                    
                    # Apply similarity threshold
                    if similarity >= similarity_threshold:
                        # Add similarity attributes to product
                        product.similarity_score = round(float(similarity), 4)
                        product.confidence_level = self._get_confidence_level(similarity)
                        
                        result = product.to_dict(include_similarity=True)
                        result['similarity_metric'] = similarity_metric
                        results.append(result)
                    
                    # Stop when we have enough results
                    if len(results) >= top_k:
                        break
                
                search_time = time.time() - start_time
                logger.info(f"Search completed in {search_time:.3f}s, found {len(results)} results")
                
                return results
                
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise
    
    def _get_confidence_level(self, similarity: float) -> str:
        """Convert similarity score to confidence level."""
        if similarity >= 0.95:
            return "Excellent"
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
    
    def get_categories(self) -> List[str]:
        """Get all available product categories."""
        try:
            with self.db_manager.get_session() as session:
                categories = session.query(Product.category).distinct().all()
                return [cat[0] for cat in categories if cat[0]]
        except Exception as e:
            logger.error(f"Failed to get categories: {e}")
            return []
    
    def get_brands(self) -> List[str]:
        """Get all available product brands."""
        try:
            with self.db_manager.get_session() as session:
                brands = session.query(Product.brand).distinct().all()
                return [brand[0] for brand in brands if brand[0]]
        except Exception as e:
            logger.error(f"Failed to get brands: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return self.db_manager.get_statistics()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        model_info = self.feature_extractor.get_info()
        model_info['database_connected'] = self.db_manager.test_connection()
        return model_info
    
    def validate_image(self, image_path: str) -> bool:
        """Validate image file."""
        return self.feature_extractor._validate_image(image_path)
    
    def get_image_features(self, image_path: str) -> Optional[np.ndarray]:
        """Extract features from image."""
        return self.feature_extractor.extract_features(image_path)

# Example usage
def main():
    """Example usage of the ProductIdentifier system."""
    # Database configuration
    DATABASE_URL = os.environ.get(
        'DATABASE_URL',
        'postgresql+psycopg2://username:password@localhost:5432/product_db'
    )
    
    try:
        # Initialize system
        identifier = ProductIdentifier(db_url=DATABASE_URL, model_name='resnet18')
        
        # Create dummy images for testing
        identifier.create_dummy_images()
        
        # Sample product data
        products_data = [
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
            }
        ]
        
        # Populate database
        products = identifier.populate_database(products_data)
        logger.info(f"Populated database with {len(products)} products")
        
        # Get statistics
        stats = identifier.get_database_stats()
        logger.info(f"Database statistics: {stats}")
        
        # Example search
        if os.path.exists('blue_shirt.png'):
            query_features = identifier.get_image_features('blue_shirt.png')
            if query_features is not None:
                results = identifier.find_similar_products(query_features, top_k=2)
                logger.info("Search results:")
                for result in results:
                    logger.info(f"  {result['name']} - Similarity: {result['similarity_score']}")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        raise

if __name__ == '__main__':
    main()