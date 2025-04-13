import numpy as np
import json
import os
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import geopy.distance
from collections import defaultdict
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

class CivicIssueDuplicateDetector:
    def __init__(self, n_clusters=None, location_threshold=0.1, text_similarity_threshold=0.8):
        """
        Initialize the duplicate detection model using unsupervised clustering
        
        Args:
            n_clusters: Number of clusters for K-means (default: None - will be determined dynamically)
            location_threshold: Max distance in km to consider location similar (default: 0.1 km = 100m)
            text_similarity_threshold: Threshold for text similarity (default: 0.8)
        """
        # Initialize image feature extractor (ResNet50)
        self.image_model = models.resnet50(weights='DEFAULT')
        self.image_model.eval()
        # Remove the classification layer
        self.image_model = torch.nn.Sequential(*(list(self.image_model.children())[:-1]))
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Text embedding model
        self.text_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # TF-IDF vectorizer as alternative text representation
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        
        # Clustering parameters
        self.n_clusters = n_clusters
        self.location_threshold = location_threshold
        self.text_similarity_threshold = text_similarity_threshold
        
        # Storage for processed data and clusters
        self.image_features_db = []
        self.location_db = []
        self.text_embeddings_db = []
        self.issue_types_db = []
        self.reports_db = []
        
        # Cluster models
        self.image_kmeans = None
        self.text_kmeans = None
        
        # Cluster assignments
        self.image_clusters = []
        self.location_clusters = defaultdict(list)  # Will store indices by location grid
        self.issue_type_clusters = defaultdict(list)  # Will store indices by issue type
        
        # XGBoost model and scaler
        self.xgb_model = None
        self.scaler = None
        self.has_enough_data_for_xgboost = False
        
    def extract_image_features(self, image_path):
        """Extract image features using ResNet50"""
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.image_transform(image).unsqueeze(0)
            
            with torch.no_grad():
                features = self.image_model(image)
            
            return features.squeeze().numpy()
        except Exception as e:
            # Silent error handling
            return np.zeros(2048)
    
    def extract_text_features(self, text):
        """Extract text embeddings using Sentence-BERT"""
        return self.text_model.encode(text)
    
    def location_to_grid(self, location):
        """Convert location to grid cell for clustering"""
        # Using a simple grid approach for location clustering
        # Each grid cell is approximately location_threshold x location_threshold km
        lat, lon = location
        lat_grid = int(lat / self.location_threshold)
        lon_grid = int(lon / self.location_threshold)
        return (lat_grid, lon_grid)
    
    def add_report(self, report):
        """
        Add a new report to the database
        
        Args:
            report: Dictionary with 'text', 'image_path', 'location', 'issue_type'
        """
        # Extract features
        image_features = self.extract_image_features(report['image_path'])
        text_embedding = self.extract_text_features(report['text'])
        location = report['location']
        issue_type = report['issue_type']
        
        # Store features and report
        index = len(self.reports_db)
        self.image_features_db.append(image_features)
        self.text_embeddings_db.append(text_embedding)
        self.location_db.append(location)
        self.issue_types_db.append(issue_type)
        self.reports_db.append(report)
        
        # Add to location grid
        location_grid = self.location_to_grid(location)
        self.location_clusters[location_grid].append(index)
        
        # Add to issue type clusters
        self.issue_type_clusters[issue_type].append(index)
        
        # Check if we have enough data to train XGBoost
        self.check_and_train_xgboost()
        
        # Return the added index
        return index
    
    def build_clusters(self):
        """Build clusters from all added reports"""
        # Determine number of clusters - even with small datasets
        if self.n_clusters is None:
            # Use at least 2 clusters, but not more than half the data points
            self.n_clusters = max(2, min(int(len(self.reports_db) / 2), 50))
        
        # Proceed with clustering even with small datasets
        if len(self.reports_db) >= 2:  # Need at least 2 reports to cluster
            self.image_kmeans = KMeans(
                n_clusters=min(self.n_clusters, len(self.reports_db)), 
                random_state=42
            )
            self.image_clusters = self.image_kmeans.fit_predict(np.array(self.image_features_db))
    
    def check_and_train_xgboost(self):
        """Check if we have enough data to train XGBoost and train if possible"""
        # Check if we have enough reports of the same type in similar locations
        issue_type_counts = {}
        for issue_type, indices in self.issue_type_clusters.items():
            if len(indices) >= 5:  # We need at least 5 reports of the same type
                issue_type_counts[issue_type] = len(indices)
        
        # If we have enough data, train XGBoost
        if issue_type_counts and not self.has_enough_data_for_xgboost:
            self.train_xgboost_model()
            self.has_enough_data_for_xgboost = True
    
    def train_xgboost_model(self):
        """Train XGBoost model using pseudo-labels from current similarity metrics"""
        # Create feature vectors for each report pair
        X = []
        y = []  # Pseudo-labels based on current similarity metrics
        
        # Compare each report with every other report
        for i in range(len(self.reports_db)):
            for j in range(i+1, len(self.reports_db)):
                # Skip if different issue types
                if self.issue_types_db[i] != self.issue_types_db[j]:
                    continue
                    
                # Extract features for this pair
                text_sim = cosine_similarity([self.text_embeddings_db[i]], [self.text_embeddings_db[j]])[0][0]
                image_sim = cosine_similarity([self.image_features_db[i]], [self.image_features_db[j]])[0][0]
                
                # Calculate location similarity
                loc1 = self.location_db[i]
                loc2 = self.location_db[j]
                dist = geopy.distance.distance(loc1, loc2).kilometers
                loc_sim = 1.0 - min(1.0, dist/self.location_threshold)
                
                # Create feature vector for this pair
                features = [text_sim, image_sim, loc_sim, 
                           int(self.issue_types_db[i] == self.issue_types_db[j])]
                X.append(features)
                
                # Create pseudo-label using current similarity formula
                current_sim = 0.4 * text_sim + 0.3 * (1.0 if image_sim > 0.9 else 0.0) + 0.3 * loc_sim
                is_duplicate = 1 if current_sim >= 0.6 else 0
                y.append(is_duplicate)
        
        # Train XGBoost model if we have enough pairs
        if len(X) > 5:
            # Normalize features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train XGBoost model
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                objective='binary:logistic',
                random_state=42
            )
            self.xgb_model.fit(X_scaled, y)
    
    def find_duplicates(self, new_report):
        """
        Find if a new report is a duplicate of any existing report
        
        Args:
            new_report: Dictionary with 'text', 'image_path', 'location', 'issue_type'
            
        Returns:
            is_duplicate: Boolean indicating if this is a duplicate
            similar_reports: List of indices of similar reports
            confidence: Confidence score of duplicate detection
        """
        # Extract features from new report
        new_image_features = self.extract_image_features(new_report['image_path'])
        new_text_embedding = self.extract_text_features(new_report['text'])
        new_location = new_report['location']
        new_issue_type = new_report['issue_type']
        
        # Storage for results
        similarities = []
        
        # Check each report in the database
        for idx, report in enumerate(self.reports_db):
            # Check issue type match first
            if report['issue_type'] != new_issue_type:
                continue
                
            # Check location proximity
            dist = geopy.distance.distance(new_location, self.location_db[idx]).kilometers
            if dist > self.location_threshold:
                continue
                
            # Text similarity
            text_sim = cosine_similarity([new_text_embedding], [self.text_embeddings_db[idx]])[0][0]
            
            # Image similarity
            image_sim = cosine_similarity([new_image_features], [self.image_features_db[idx]])[0][0]
            
            # If we have images and they're identical, give high similarity
            image_name1 = os.path.basename(new_report['image_path'])
            image_name2 = os.path.basename(report['image_path'])
            same_image = image_name1 == image_name2
            
            # Use XGBoost model if available and we have enough data
            if self.xgb_model is not None and self.has_enough_data_for_xgboost:
                # Create feature vector
                features = [[text_sim, image_sim, 1.0 - min(1.0, dist/self.location_threshold),
                           int(new_issue_type == self.issue_types_db[idx])]]
                
                # Scale features
                features_scaled = self.scaler.transform(features)
                
                # Get XGBoost prediction probability
                prob = self.xgb_model.predict_proba(features_scaled)[0][1]  # Probability of being duplicate
                
                if prob >= 0.5:  # Threshold for XGBoost confidence
                    similarities.append((idx, prob))
            else:
                # Calculate overall similarity score using original method
                overall_sim = 0.4 * text_sim + 0.3 * (1.0 if same_image else 0.0) + 0.3 * (1.0 - min(1.0, dist/self.location_threshold))
                
                if overall_sim >= 0.6:  # Threshold for duplicate
                    similarities.append((idx, overall_sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        if similarities:
            return True, [idx for idx, _ in similarities], similarities[0][1]
        else:
            return False, [], 0.0
    
    def process_json_input(self, json_data):
        """
        Process JSON input to determine if a report is a duplicate
        
        Args:
            json_data: String containing JSON data or dictionary
            
        Returns:
            Dictionary with duplicate status and original report ID
        """
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        
        is_duplicate, similar_reports, confidence = self.find_duplicates(data)
        
        # Format the response with only required information
        response = {
            "is_duplicate": 1 if is_duplicate else 0
        }
        
        # Include only the original report ID, not the index
        if similar_reports and is_duplicate:
            # Map indices to report IDs
            original_ids = [self.reports_db[idx]['id'] for idx in similar_reports]
            response["original_report_id"] = original_ids[0]  # Return only the most similar report ID
        
        return response
    
    def rebuild_clusters_if_needed(self, force=False):
        """Rebuild clusters if database has grown significantly"""
        # Simple heuristic: rebuild if database has grown by 20%
        if force or (self.image_kmeans is not None and 
                    len(self.reports_db) > 1.2 * len(self.image_clusters)):
            self.build_clusters()
    
    def load_reports_from_json(self, json_file):
        """Load reports from a JSON file"""
        with open(json_file, 'r') as f:
            reports = json.load(f)
        
        for report in reports:
            self.add_report(report)
        
        # Build clusters after loading
        if len(reports) >= 2:
            self.build_clusters()

# Example usage
if __name__ == "__main__":
    # Initialize the detector
    detector = CivicIssueDuplicateDetector(
        n_clusters=10,  # Start with 10 clusters
        location_threshold=0.1,  # 100 meters
        text_similarity_threshold=0.7  # Text similarity threshold
    )
    
    # Example reports
    example_reports = [
        {
            'id': 'R001',
            'text': 'Large pothole on Main Street causing traffic issues',
            'image_path': '/content/bad-road-cracked.webp',
            'location': (28.6139, 77.2090),  # Delhi coordinates
            'issue_type': 'pothole',
            'timestamp': '2025-04-12T14:30:00'
        },
        {
            'id': 'R002',
            'text': 'Dangerous pothole on Main St needs immediate fixing',
            'image_path': '/content/high-angle-view-pot-hole-street_1048944-10365481.webp',
            'location': (28.6141, 77.2088),  # Very close to first report
            'issue_type': 'pothole',
            'timestamp': '2025-04-12T15:45:00'
        },
        {
            'id': 'R003',
            'text': 'Broken manhole cover near central park requires repair',
            'image_path': '/content/manhole.webp',
            'location': (28.5355, 77.2410),  # Different location
            'issue_type': 'manhole',
            'timestamp': '2025-04-10T09:15:00'
        },
        {
            'id': 'R004',
            'text': 'Garbage accumulation blocking the drainage system',
            'image_path': '/content/urban-waste-overflow-stockcake.webp',
            'location': (28.5358, 77.2412),  # Near report 3
            'issue_type': 'garbage',
            'timestamp': '2025-04-10T10:30:00'
        }
    ]
    
    # Add reports to the database
    for report in example_reports:
        detector.add_report(report)
    
    # Build clusters
    detector.build_clusters()
    
    # Check for duplicate
    new_report = {
        'text': 'There is a large pothole on Main Street that needs fixing',
        'image_path': '/content/bad-road-cracked.webp',
        'location': (28.6140, 77.2091),  # Very close to reports 1 and 2
        'issue_type': 'pothole',
        'timestamp': '2025-04-13T08:15:00'
    }
    
    # Process as JSON
    result = detector.process_json_input(new_report)
    print(json.dumps(result, indent=2))