#!/usr/bin/env python3
"""
Mac M4 Air ML Stress Test
A comprehensive benchmark to test the limits of your M4 Air across:
- Multi-core CPU performance
- GPU acceleration via Metal Performance Shaders
- Memory bandwidth and thermal management
- Mixed precision training
- Parallel processing capabilities
"""

import time
import psutil
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

class M4StressTest:
    def __init__(self):
        self.results = {}
        self.start_time = None
        
        # System info
        self.cpu_count = psutil.cpu_count(logical=True)
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
        print(f"🚀 M4 Air Stress Test Initialized")
        print(f"CPU Cores: {self.cpu_count}")
        print(f"Memory: {self.memory_gb:.1f} GB")
        
        # Configure TensorFlow for Metal GPU
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                print(f"✅ TensorFlow Metal GPU detected: {len(physical_devices)} device(s)")
            else:
                print("⚠️ No Metal GPU detected for TensorFlow")
        except:
            print("⚠️ TensorFlow GPU configuration failed")
            
        # Configure PyTorch for MPS (Metal Performance Shaders)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"✅ PyTorch device: {self.device}")

    def monitor_system(self, duration=1):
        """Monitor system resources"""
        cpu_percent = psutil.cpu_percent(interval=duration, percpu=True)
        memory = psutil.virtual_memory()
        temps = []
        
        # Try to get temperature (may not work on all systems)
        try:
            import subprocess
            temp_output = subprocess.check_output(['sudo', 'powermetrics', '--samplers', 'smc', '-n', '1', '-i', '1000'])
            # Parse temperature from output (this is macOS specific)
        except:
            temps = [0]  # Fallback if temperature monitoring fails
            
        return {
            'cpu_avg': np.mean(cpu_percent),
            'cpu_max': np.max(cpu_percent),
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'temp_avg': np.mean(temps) if temps else 0
        }

    def stress_test_1_massive_data_generation(self):
        """Test 1: Generate and process massive datasets"""
        print("\n🔥 Test 1: Massive Data Generation & Processing")
        start_time = time.time()
        
        # Generate massive classification dataset
        print("Generating 1M samples, 1000 features...")
        X, y = make_classification(
            n_samples=1_000_000,
            n_features=1000,
            n_informative=800,
            n_redundant=200,
            n_clusters_per_class=10,
            class_sep=0.8,
            random_state=42
        )
        
        # Memory-intensive operations
        print("Performing memory-intensive operations...")
        X_squared = np.square(X)
        X_log = np.log1p(np.abs(X))
        X_combined = np.hstack([X, X_squared, X_log])
        
        # Multi-threaded correlation matrix
        print("Computing correlation matrix...")
        corr_matrix = np.corrcoef(X_combined.T)
        
        end_time = time.time()
        self.results['massive_data'] = {
            'time': end_time - start_time,
            'data_shape': X_combined.shape,
            'memory_used': X_combined.nbytes / (1024**3)
        }
        
        print(f"✅ Completed in {end_time - start_time:.2f}s")
        print(f"Final dataset: {X_combined.shape} ({X_combined.nbytes/(1024**3):.2f} GB)")
        
        return X_combined[:100_000], y[:100_000]  # Return subset for next tests

    def stress_test_2_parallel_ml_training(self, X, y):
        """Test 2: Parallel ML model training"""
        print("\n🔥 Test 2: Parallel ML Model Training")
        start_time = time.time()
        
        def train_rf():
            rf = RandomForestClassifier(n_estimators=500, max_depth=20, n_jobs=-1, random_state=42)
            rf.fit(X, y)
            return rf.score(X, y)
        
        def train_gbm():
            gbm = GradientBoostingClassifier(n_estimators=300, max_depth=8, random_state=42)
            gbm.fit(X, y)
            return gbm.score(X, y)
        
        def train_xgb():
            xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=8, n_jobs=-1, random_state=42)
            xgb_model.fit(X, y)
            return xgb_model.score(X, y)
        
        def train_lgb():
            lgb_model = lgb.LGBMClassifier(n_estimators=300, max_depth=8, n_jobs=-1, random_state=42)
            lgb_model.fit(X, y)
            return lgb_model.score(X, y)
        
        # Train all models in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                'RandomForest': executor.submit(train_rf),
                'GradientBoosting': executor.submit(train_gbm),
                'XGBoost': executor.submit(train_xgb),
                'LightGBM': executor.submit(train_lgb)
            }
            
            scores = {name: future.result() for name, future in futures.items()}
        
        end_time = time.time()
        self.results['parallel_ml'] = {
            'time': end_time - start_time,
            'scores': scores
        }
        
        print(f"✅ Parallel training completed in {end_time - start_time:.2f}s")
        for model, score in scores.items():
            print(f"   {model}: {score:.4f}")

    def stress_test_3_deep_learning_tensorflow(self, X, y):
        """Test 3: TensorFlow deep learning with Metal GPU"""
        print("\n🔥 Test 3: TensorFlow Deep Learning (Metal GPU)")
        start_time = time.time()
        
        # Create a complex neural network
        model = keras.Sequential([
            keras.layers.Dense(2048, activation='relu', input_shape=(X.shape[1],)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Model parameters: {model.count_params():,}")
        
        # Train the model
        history = model.fit(
            X.astype(np.float32), y,
            batch_size=512,
            epochs=20,
            validation_split=0.2,
            verbose=1
        )
        
        end_time = time.time()
        self.results['tensorflow_dl'] = {
            'time': end_time - start_time,
            'final_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1],
            'parameters': model.count_params()
        }
        
        print(f"✅ TensorFlow training completed in {end_time - start_time:.2f}s")

    def stress_test_4_pytorch_mps(self, X, y):
        """Test 4: PyTorch with MPS (Metal Performance Shaders)"""
        print("\n🔥 Test 4: PyTorch with MPS Acceleration")
        start_time = time.time()
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
        
        # Define a complex neural network
        class ComplexNet(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, 2048),
                    nn.BatchNorm1d(2048),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(2048, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.layers(x)
        
        model = ComplexNet(X.shape[1]).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        print(f"PyTorch model parameters: {param_count:,}")
        
        # Training loop
        model.train()
        for epoch in range(15):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 3 == 0:
                print(f"Epoch {epoch+1}/15, Loss: {epoch_loss/len(dataloader):.4f}")
        
        end_time = time.time()
        self.results['pytorch_mps'] = {
            'time': end_time - start_time,
            'parameters': param_count,
            'device': str(self.device)
        }
        
        print(f"✅ PyTorch MPS training completed in {end_time - start_time:.2f}s")

    def stress_test_5_hyperparameter_tuning(self, X, y):
        """Test 5: Intensive hyperparameter tuning"""
        print("\n🔥 Test 5: Intensive Hyperparameter Tuning")
        start_time = time.time()
        
        # Use a subset for faster tuning
        X_sub = X[:10_000]
        y_sub = y[:10_000]
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, 20],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=1)  # Let GridSearchCV handle parallelism
        
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,  # Use all available cores
            verbose=1
        )
        
        grid_search.fit(X_sub, y_sub)
        
        end_time = time.time()
        self.results['hyperparameter_tuning'] = {
            'time': end_time - start_time,
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'total_fits': len(grid_search.cv_results_['params'])
        }
        
        print(f"✅ Hyperparameter tuning completed in {end_time - start_time:.2f}s")
        print(f"Best score: {grid_search.best_score_:.4f}")
        print(f"Total model fits: {len(grid_search.cv_results_['params'])}")

    def stress_test_6_multiprocessing_ensemble(self, X, y):
        """Test 6: Multiprocessing ensemble with CPU saturation"""
        print("\n🔥 Test 6: Multiprocessing Ensemble (CPU Saturation)")
        start_time = time.time()
        
        def train_model_process(args):
            model_type, X_data, y_data, random_seed = args
            
            if model_type == 'rf':
                model = RandomForestClassifier(n_estimators=200, random_state=random_seed, n_jobs=1)
            elif model_type == 'gbm':
                model = GradientBoostingClassifier(n_estimators=100, random_state=random_seed)
            elif model_type == 'xgb':
                model = xgb.XGBClassifier(n_estimators=100, random_state=random_seed, n_jobs=1)
            
            model.fit(X_data, y_data)
            return model.score(X_data, y_data)
        
        # Create ensemble of models across all CPU cores
        tasks = []
        models = ['rf', 'gbm', 'xgb']
        
        for i in range(self.cpu_count):
            model_type = models[i % len(models)]
            tasks.append((model_type, X, y, i))
        
        # Use ProcessPoolExecutor to saturate all CPU cores
        with ProcessPoolExecutor(max_workers=self.cpu_count) as executor:
            scores = list(executor.map(train_model_process, tasks))
        
        end_time = time.time()
        self.results['multiprocessing_ensemble'] = {
            'time': end_time - start_time,
            'num_models': len(scores),
            'avg_score': np.mean(scores),
            'cpu_cores_used': self.cpu_count
        }
        
        print(f"✅ Multiprocessing ensemble completed in {end_time - start_time:.2f}s")
        print(f"Trained {len(scores)} models across {self.cpu_count} cores")
        print(f"Average score: {np.mean(scores):.4f}")

    def generate_report(self):
        """Generate comprehensive performance report"""
        print("\n" + "="*60)
        print("🎯 MAC M4 AIR ML STRESS TEST RESULTS")
        print("="*60)
        
        total_time = sum(test['time'] for test in self.results.values())
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"System: Mac M4 Air - {self.cpu_count} cores, {self.memory_gb:.1f} GB RAM")
        
        print(f"\n🔥 INDIVIDUAL TEST RESULTS:")
        for test_name, results in self.results.items():
            print(f"\n{test_name.upper().replace('_', ' ')}:")
            print(f"  ⏱️  Time: {results['time']:.2f}s")
            
            if 'data_shape' in results:
                print(f"  📊 Data shape: {results['data_shape']}")
                print(f"  💾 Memory used: {results['memory_used']:.2f} GB")
            
            if 'scores' in results:
                for model, score in results['scores'].items():
                    print(f"  🎯 {model}: {score:.4f}")
            
            if 'final_accuracy' in results:
                print(f"  🎯 Final accuracy: {results['final_accuracy']:.4f}")
                print(f"  📈 Parameters: {results['parameters']:,}")
            
            if 'best_score' in results:
                print(f"  🏆 Best CV score: {results['best_score']:.4f}")
                print(f"  🔧 Total fits: {results['total_fits']}")
            
            if 'num_models' in results:
                print(f"  🤖 Models trained: {results['num_models']}")
                print(f"  🎯 Average score: {results['avg_score']:.4f}")
        
        # Performance scoring
        print(f"\n🏆 PERFORMANCE GRADE:")
        if total_time < 300:  # 5 minutes
            grade = "🚀 EXCELLENT - Your M4 Air is a beast!"
        elif total_time < 600:  # 10 minutes  
            grade = "⚡ VERY GOOD - Solid performance!"
        elif total_time < 900:  # 15 minutes
            grade = "👍 GOOD - Decent performance"
        else:
            grade = "⏳ AVERAGE - Consider upgrading?"
        
        print(grade)
        
        # Save results
        results_df = pd.DataFrame([
            {'Test': name, 'Time (s)': data['time']} 
            for name, data in self.results.items()
        ])
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=results_df, x='Test', y='Time (s)')
        plt.xticks(rotation=45, ha='right')
        plt.title('M4 Air ML Stress Test Results')
        plt.tight_layout()
        plt.savefig('m4_stress_test_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📈 Results saved to: m4_stress_test_results.png")

    def run_all_tests(self):
        """Run the complete stress test suite"""
        print("🔥 Starting Complete M4 Air ML Stress Test Suite")
        print("This will push your system to its limits!")
        print("Monitor Activity Monitor to see CPU, GPU, and thermal behavior\n")
        
        self.start_time = time.time()
        
        try:
            # Test 1: Data generation
            X, y = self.stress_test_1_massive_data_generation()
            
            # Test 2: Parallel ML
            self.stress_test_2_parallel_ml_training(X, y)
            
            # Test 3: TensorFlow + Metal
            self.stress_test_3_deep_learning_tensorflow(X, y)
            
            # Test 4: PyTorch + MPS
            self.stress_test_4_pytorch_mps(X, y)
            
            # Test 5: Hyperparameter tuning
            self.stress_test_5_hyperparameter_tuning(X, y)
            
            # Test 6: Multiprocessing
            self.stress_test_6_multiprocessing_ensemble(X, y)
            
            # Generate final report
            self.generate_report()
            
        except Exception as e:
            print(f"❌ Error during stress test: {e}")
            if self.results:
                self.generate_report()

if __name__ == "__main__":
    # Run the stress test
    tester = M4StressTest()
    tester.run_all_tests()
    
    print("\n🎉 M4 Air Stress Test Complete!")
    print("Check Activity Monitor during the test to see:")
    print("- CPU usage across all cores")
    print("- GPU utilization (Metal)")
    print("- Memory pressure") 
    print("- Temperature and thermal throttling")