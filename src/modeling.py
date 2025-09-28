"""
Modeling Module for E-Commerce Analytics
Machine learning models and evaluation functions

Author: Soumitra Upadhyay
Date: September 2025
Project: E-Commerce Analytics Capstone
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report,
                           mean_squared_error, mean_absolute_error, r2_score)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.tree import DecisionTreeClassifier
import joblib
import warnings

warnings.filterwarnings('ignore')

class ECommerceModeling:
    """
    Comprehensive modeling class for e-commerce analytics
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def customer_segmentation_kmeans(self, data: pd.DataFrame, 
                                   features: List[str],
                                   n_clusters: int = 5,
                                   model_name: str = 'customer_segments') -> Dict:
        """
        Perform customer segmentation using K-means clustering
        
        Args:
            data (pd.DataFrame): Customer data with features
            features (List[str]): Features to use for clustering
            n_clusters (int): Number of clusters
            model_name (str): Name to store the model
            
        Returns:
            Dict: Clustering results and metrics
        """
        # Prepare data
        X = data[features].copy()
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Store model and scaler
        self.models[model_name] = kmeans
        self.scalers[f"{model_name}_scaler"] = scaler
        
        # Add cluster labels to data
        result_data = data.copy()
        result_data['cluster'] = clusters
        
        # Calculate cluster characteristics
        cluster_summary = result_data.groupby('cluster')[features].agg(['mean', 'count']).round(2)
        
        # Calculate silhouette score
        from sklearn.metrics import silhouette_score
        silhouette_avg = silhouette_score(X_scaled, clusters)
        
        # Inertia (within-cluster sum of squares)
        inertia = kmeans.inertia_
        
        results = {
            'model': kmeans,
            'scaler': scaler,
            'data_with_clusters': result_data,
            'cluster_summary': cluster_summary,
            'silhouette_score': silhouette_avg,
            'inertia': inertia,
            'n_clusters': n_clusters
        }
        
        self.results[model_name] = results
        
        print(f"✅ K-means clustering completed:")
        print(f"   Silhouette Score: {silhouette_avg:.3f}")
        print(f"   Inertia: {inertia:.2f}")
        
        return results
    
    def find_optimal_clusters(self, data: pd.DataFrame, features: List[str],
                            max_clusters: int = 10) -> Dict:
        """
        Find optimal number of clusters using elbow method and silhouette analysis
        
        Args:
            data (pd.DataFrame): Customer data
            features (List[str]): Features for clustering
            max_clusters (int): Maximum number of clusters to test
            
        Returns:
            Dict: Optimization results
        """
        X = data[features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        
        from sklearn.metrics import silhouette_score
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, clusters))
        
        # Find elbow point (simple approach)
        # Calculate the rate of change in inertia
        inertia_diff = np.diff(inertias)
        inertia_diff2 = np.diff(inertia_diff)
        elbow_point = k_range[np.argmax(inertia_diff2) + 1]
        
        # Find best silhouette score
        best_silhouette_k = k_range[np.argmax(silhouette_scores)]
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Elbow plot
        ax1.plot(k_range, inertias, 'bo-')
        ax1.axvline(x=elbow_point, color='red', linestyle='--', label=f'Elbow at k={elbow_point}')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.legend()
        ax1.grid(True)
        
        # Silhouette plot
        ax2.plot(k_range, silhouette_scores, 'ro-')
        ax2.axvline(x=best_silhouette_k, color='blue', linestyle='--', 
                   label=f'Best Silhouette at k={best_silhouette_k}')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis for Optimal k')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        results = {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'elbow_point': elbow_point,
            'best_silhouette_k': best_silhouette_k,
            'max_silhouette_score': max(silhouette_scores)
        }
        
        print(f"✅ Optimal cluster analysis completed:")
        print(f"   Elbow method suggests: k = {elbow_point}")
        print(f"   Best silhouette score at: k = {best_silhouette_k} (score: {max(silhouette_scores):.3f})")
        
        return results
    
    def churn_prediction_model(self, data: pd.DataFrame, 
                             target_column: str,
                             feature_columns: List[str] = None,
                             model_type: str = 'random_forest',
                             test_size: float = 0.2) -> Dict:
        """
        Build and evaluate churn prediction model
        
        Args:
            data (pd.DataFrame): Customer data with churn labels
            target_column (str): Target variable (churn indicator)
            feature_columns (List[str]): Features for prediction
            model_type (str): Model type ('random_forest', 'logistic', 'gradient_boosting')
            test_size (float): Test set size
            
        Returns:
            Dict: Model results and evaluation metrics
        """
        # Prepare data
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        X = data[feature_columns].copy()
        y = data[target_column].copy()
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Select model
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        elif model_type == 'logistic':
            model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        elif model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(random_state=self.random_state)
        
        # Train model
        if model_type == 'logistic':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Feature importance (for tree-based models)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store model
        model_name = f'churn_prediction_{model_type}'
        self.models[model_name] = model
        self.scalers[f"{model_name}_scaler"] = scaler
        
        results = {
            'model': model,
            'scaler': scaler,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba,
            'test_data': {'X_test': X_test, 'y_test': y_test}
        }
        
        self.results[model_name] = results
        
        print(f"✅ Churn prediction model ({model_type}) completed:")
        for metric, value in metrics.items():
            print(f"   {metric.title()}: {value:.3f}")
        
        return results
    
    def customer_lifetime_value_prediction(self, data: pd.DataFrame,
                                         target_column: str,
                                         feature_columns: List[str] = None,
                                         model_type: str = 'random_forest',
                                         test_size: float = 0.2) -> Dict:
        """
        Build CLV prediction model
        
        Args:
            data (pd.DataFrame): Customer data with CLV values
            target_column (str): CLV target variable
            feature_columns (List[str]): Features for prediction
            model_type (str): Model type ('random_forest', 'linear', 'gradient_boosting')
            test_size (float): Test set size
            
        Returns:
            Dict: Model results and evaluation metrics
        """
        # Prepare data
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        X = data[feature_columns].copy()
        y = data[target_column].copy()
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Select model
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        elif model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(random_state=self.random_state)
        
        # Train model
        if model_type == 'linear':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred)
        }
        
        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Store model
        model_name = f'clv_prediction_{model_type}'
        self.models[model_name] = model
        self.scalers[f"{model_name}_scaler"] = scaler
        
        results = {
            'model': model,
            'scaler': scaler,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'test_data': {'X_test': X_test, 'y_test': y_test}
        }
        
        self.results[model_name] = results
        
        print(f"✅ CLV prediction model ({model_type}) completed:")
        for metric, value in metrics.items():
            print(f"   {metric.upper()}: {value:.3f}")
        
        return results
    
    def hyperparameter_tuning(self, model, param_grid: Dict, 
                            X_train, y_train,
                            cv: int = 5,
                            scoring: str = 'accuracy') -> Dict:
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            model: ML model to tune
            param_grid (Dict): Parameter grid for tuning
            X_train: Training features
            y_train: Training target
            cv (int): Cross-validation folds
            scoring (str): Scoring metric
            
        Returns:
            Dict: Tuning results
        """
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring, 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        results = {
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'grid_search': grid_search
        }
        
        print(f"✅ Hyperparameter tuning completed:")
        print(f"   Best Score: {grid_search.best_score_:.3f}")
        print(f"   Best Parameters: {grid_search.best_params_}")
        
        return results
    
    def plot_model_performance(self, model_results: Dict, model_name: str):
        """
        Plot model performance metrics and visualizations
        
        Args:
            model_results (Dict): Results from model training
            model_name (str): Name of the model for titles
        """
        if 'confusion_matrix' in model_results:
            # Classification model
            self._plot_classification_results(model_results, model_name)
        else:
            # Regression model
            self._plot_regression_results(model_results, model_name)
    
    def _plot_classification_results(self, results: Dict, model_name: str):
        """Plot classification model results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} - Classification Results', fontsize=16, fontweight='bold')
        
        # Confusion Matrix
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_ylabel('True Label')
        axes[0,0].set_xlabel('Predicted Label')
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        y_test = results['test_data']['y_test']
        y_proba = results['prediction_probabilities']
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        axes[0,1].plot(fpr, tpr, label=f'ROC Curve (AUC = {results["metrics"]["roc_auc"]:.3f})')
        axes[0,1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Feature Importance
        if results['feature_importance'] is not None:
            top_features = results['feature_importance'].head(10)
            axes[1,0].barh(top_features['feature'], top_features['importance'])
            axes[1,0].set_title('Top 10 Feature Importance')
            axes[1,0].set_xlabel('Importance')
        
        # Metrics Bar Plot
        metrics = results['metrics']
        axes[1,1].bar(metrics.keys(), metrics.values())
        axes[1,1].set_title('Model Metrics')
        axes[1,1].set_ylabel('Score')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_regression_results(self, results: Dict, model_name: str):
        """Plot regression model results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} - Regression Results', fontsize=16, fontweight='bold')
        
        y_test = results['test_data']['y_test']
        y_pred = results['predictions']
        
        # Actual vs Predicted
        axes[0,0].scatter(y_test, y_pred, alpha=0.6)
        axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0,0].set_xlabel('Actual Values')
        axes[0,0].set_ylabel('Predicted Values')
        axes[0,0].set_title('Actual vs Predicted Values')
        
        # Residuals Plot
        residuals = y_test - y_pred
        axes[0,1].scatter(y_pred, residuals, alpha=0.6)
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        axes[0,1].set_xlabel('Predicted Values')
        axes[0,1].set_ylabel('Residuals')
        axes[0,1].set_title('Residuals Plot')
        
        # Feature Importance
        if results['feature_importance'] is not None:
            top_features = results['feature_importance'].head(10)
            axes[1,0].barh(top_features['feature'], top_features['importance'])
            axes[1,0].set_title('Top 10 Feature Importance')
            axes[1,0].set_xlabel('Importance')
        
        # Metrics Bar Plot
        metrics = results['metrics']
        axes[1,1].bar(metrics.keys(), metrics.values())
        axes[1,1].set_title('Model Metrics')
        axes[1,1].set_ylabel('Score')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_name: str, filepath: str):
        """
        Save a trained model to disk
        
        Args:
            model_name (str): Name of the model to save
            filepath (str): Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model_data = {
            'model': self.models[model_name],
            'scaler': self.scalers.get(f"{model_name}_scaler"),
            'results': self.results.get(model_name)
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model '{model_name}' saved to {filepath}")
    
    def load_model(self, filepath: str, model_name: str):
        """
        Load a saved model from disk
        
        Args:
            filepath (str): Path to the saved model
            model_name (str): Name to assign to the loaded model
        """
        model_data = joblib.load(filepath)
        
        self.models[model_name] = model_data['model']
        if model_data['scaler']:
            self.scalers[f"{model_name}_scaler"] = model_data['scaler']
        if model_data['results']:
            self.results[model_name] = model_data['results']
        
        print(f"✅ Model loaded as '{model_name}' from {filepath}")
    
    def predict_new_data(self, model_name: str, new_data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data using a trained model
        
        Args:
            model_name (str): Name of the trained model
            new_data (pd.DataFrame): New data for predictions
            
        Returns:
            np.ndarray: Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        scaler = self.scalers.get(f"{model_name}_scaler")
        
        X_new = new_data.copy()
        
        # Handle categorical variables (basic approach)
        categorical_cols = X_new.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X_new[col] = le.fit_transform(X_new[col].astype(str))
        
        # Scale if scaler exists
        if scaler:
            X_new_scaled = scaler.transform(X_new)
            predictions = model.predict(X_new_scaled)
        else:
            predictions = model.predict(X_new)
        
        return predictions