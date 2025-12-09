# utils.py — Enhanced version with comprehensive data cleaning
# Includes: error detection, duplicate handling, outlier detection, missing value treatment

import numpy as np
import pandas as pd
import torch
import os
from datetime import datetime

class DataCleaningReport:
    """Store and display data cleaning operations"""
    def __init__(self):
        self.operations = []
    
    def add(self, operation, details):
        self.operations.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'operation': operation,
            'details': details
        })
    
    def save(self, filepath='data_cleaning_report.txt'):
        with open(filepath, 'w') as f:
            f.write("="*60 + "\n")
            f.write("DATA CLEANING REPORT\n")
            f.write("="*60 + "\n\n")
            for op in self.operations:
                f.write(f"[{op['timestamp']}] {op['operation']}\n")
                f.write(f"  {op['details']}\n\n")
        print(f"Data cleaning report saved to {filepath}")

class Data_utility(object):
    def __init__(self, args):
        self.data_path = args.data
        self.train_ratio = args.train
        self.valid_ratio = args.valid
        self.horizon = args.horizon
        self.window = args.window
        self.normalize = args.normalize
        
        # Initialize cleaning report
        self.cleaning_report = DataCleaningReport()
        
        # ======================= Load and Clean Data =======================
        print("="*60)
        print("STEP 1: Loading Raw Data")
        print("="*60)
        
        raw_df = pd.read_csv(self.data_path)
        print(f"✓ Raw data loaded: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
        self.cleaning_report.add("Data Loading", 
                                f"Loaded {raw_df.shape[0]} rows × {raw_df.shape[1]} columns from {self.data_path}")
        
        # Store date column if exists
        self.dates = raw_df.iloc[:, 0].values if raw_df.shape[1] > 1 else None
        
        # Extract numerical data (skip first column which is typically date)
        data = raw_df.iloc[:, 1:].values.astype(np.float32)
        
        print("\n" + "="*60)
        print("STEP 2: Identifying and Handling Errors")
        print("="*60)
        
        # Check for infinite values
        inf_mask = np.isinf(data)
        if np.any(inf_mask):
            n_inf = np.sum(inf_mask)
            print(f"⚠ Found {n_inf} infinite values - replacing with NaN")
            data[inf_mask] = np.nan
            self.cleaning_report.add("Error Detection", 
                                    f"Found and replaced {n_inf} infinite values")
        else:
            print("✓ No infinite values detected")
            self.cleaning_report.add("Error Detection", "No infinite values found")
        
        # Check for negative values (ILI data should be non-negative)
        negative_mask = data < 0
        if np.any(negative_mask):
            n_negative = np.sum(negative_mask)
            print(f"⚠ Found {n_negative} negative values - replacing with NaN")
            data[negative_mask] = np.nan
            self.cleaning_report.add("Error Detection", 
                                    f"Found and replaced {n_negative} negative values")
        else:
            print("✓ No negative values detected")
            self.cleaning_report.add("Error Detection", "No negative values found")
        
        print("\n" + "="*60)
        print("STEP 3: Detecting Duplicates")
        print("="*60)
        
        # Check for duplicate rows (if dates are available)
        if self.dates is not None:
            unique_dates, counts = np.unique(self.dates, return_counts=True)
            duplicates = unique_dates[counts > 1]
            if len(duplicates) > 0:
                print(f"⚠ Found {len(duplicates)} duplicate dates")
                # Keep first occurrence of duplicates
                _, unique_indices = np.unique(self.dates, return_index=True)
                unique_indices = np.sort(unique_indices)
                data = data[unique_indices]
                self.dates = self.dates[unique_indices]
                self.cleaning_report.add("Duplicate Handling", 
                                        f"Removed {len(self.dates) - len(unique_indices)} duplicate rows")
            else:
                print("✓ No duplicate dates found")
                self.cleaning_report.add("Duplicate Handling", "No duplicates detected")
        
        print("\n" + "="*60)
        print("STEP 4: Outlier Detection")
        print("="*60)
        
        # Detect outliers using IQR method
        outliers_detected = 0
        for region_idx in range(data.shape[1]):
            region_data = data[:, region_idx]
            valid_data = region_data[~np.isnan(region_data)]
            
            if len(valid_data) > 0:
                Q1 = np.percentile(valid_data, 25)
                Q3 = np.percentile(valid_data, 75)
                IQR = Q3 - Q1
                
                # Define outliers as values beyond 3*IQR (more lenient for disease data)
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outlier_mask = (region_data < lower_bound) | (region_data > upper_bound)
                outlier_mask = outlier_mask & ~np.isnan(region_data)
                
                n_outliers = np.sum(outlier_mask)
                if n_outliers > 0:
                    outliers_detected += n_outliers
                    # Mark outliers as NaN for imputation
                    data[outlier_mask, region_idx] = np.nan
        
        if outliers_detected > 0:
            print(f"⚠ Detected {outliers_detected} outliers (marked for imputation)")
            self.cleaning_report.add("Outlier Detection", 
                                    f"Detected {outliers_detected} outliers using IQR method (3×IQR threshold)")
        else:
            print("✓ No outliers detected")
            self.cleaning_report.add("Outlier Detection", "No outliers detected")
        
        print("\n" + "="*60)
        print("STEP 5: Handling Missing Values")
        print("="*60)
        
        # Count missing values
        n_missing = np.sum(np.isnan(data))
        missing_pct = 100 * n_missing / data.size
        print(f"Missing values: {n_missing} ({missing_pct:.2f}% of data)")
        
        if n_missing > 0:
            # Impute missing values using linear interpolation
            data_imputed = self._impute_missing_values(data)
            print(f"✓ Imputed {n_missing} missing values using linear interpolation")
            self.cleaning_report.add("Missing Value Treatment", 
                                    f"Imputed {n_missing} ({missing_pct:.2f}%) missing values using linear interpolation")
            data = data_imputed
        else:
            print("✓ No missing values to handle")
            self.cleaning_report.add("Missing Value Treatment", "No missing values found")
        
        # Store cleaned data
        self.rawdat = data
        self.T, self.m = self.rawdat.shape
        
        print("\n" + "="*60)
        print("STEP 6: Data Validation")
        print("="*60)
        
        # Validate cleaned data
        assert not np.any(np.isnan(self.rawdat)), "❌ NaN values still present after cleaning!"
        assert not np.any(np.isinf(self.rawdat)), "❌ Infinite values still present after cleaning!"
        assert not np.any(self.rawdat < 0), "❌ Negative values still present after cleaning!"
        
        print(f"✓ Final cleaned dataset validated: {self.T} weeks × {self.m} regions")
        print("✓ No NaN, infinite, or negative values")
        self.cleaning_report.add("Data Validation", 
                                f"Cleaned dataset validated: {self.T} weeks × {self.m} regions, all quality checks passed")
        
        # ======================= Load Adjacency Matrix =======================
        print("\n" + "="*60)
        print("Loading Adjacency Matrix")
        print("="*60)
        
        adj_file = 'data/us_hhs/ind_mat2.txt'
        if not os.path.exists(adj_file):
            raise FileNotFoundError(f"Missing adjacency matrix: {adj_file}")
        self.adj = np.loadtxt(adj_file, delimiter=',')
        print(f"✓ Loaded binary adjacency matrix from {adj_file} (shape: {self.adj.shape})")
        
        # ======================= Normalization =======================
        print("\n" + "="*60)
        print("STEP 7: Normalization")
        print("="*60)
        
        train_len = int(self.T * self.train_ratio)
        train_data = self.rawdat[:train_len]
        
        if self.normalize == 2:
            max_val = np.max(train_data, axis=0)
            max_val[max_val == 0] = 1.0
            self.dat = self.rawdat / max_val[None, :]
            self.scale = torch.from_numpy(max_val).float()
            print(f"✓ Applied max normalization (max values per region)")
            self.cleaning_report.add("Normalization", 
                                    "Applied max normalization using training set maximum values")
        else:
            self.dat = self.rawdat.copy()
            self.scale = torch.ones(self.m)
            print("✓ No normalization applied")
            self.cleaning_report.add("Normalization", "No normalization applied")
        
        # ======================= Create Datasets =======================
        print("\n" + "="*60)
        print("STEP 8: Splitting into Train/Validation/Test Sets")
        print("="*60)
        
        def create_dataset(start_idx, end_idx):
            seq_len_needed = self.window + self.horizon
            if end_idx - start_idx < seq_len_needed:
                return [], []
            
            inputs, targets = [], []
            for i in range(start_idx, end_idx - seq_len_needed + 1):
                x = self.dat[i:i + self.window]
                y = self.dat[i + self.window:i + self.window + self.horizon]
                inputs.append(x)
                targets.append(y)
            return inputs, targets
        
        # Split points
        train_end = int(self.T * self.train_ratio)
        valid_end = train_end + int(self.T * self.valid_ratio)
        
        train_x, train_y = create_dataset(0, train_end)
        valid_x, valid_y = create_dataset(train_end, valid_end)
        test_x, test_y = create_dataset(valid_end, self.T)
        
        # Convert to list of torch tensors
        self.train = [(torch.FloatTensor(x), torch.FloatTensor(y)) for x, y in zip(train_x, train_y)]
        self.valid = [(torch.FloatTensor(x), torch.FloatTensor(y)) for x, y in zip(valid_x, valid_y)]
        self.test = [(torch.FloatTensor(x), torch.FloatTensor(y)) for x, y in zip(test_x, test_y)]
        
        print(f"✓ Train set: {len(self.train)} samples (weeks 0-{train_end})")
        print(f"✓ Validation set: {len(self.valid)} samples (weeks {train_end}-{valid_end})")
        print(f"✓ Test set: {len(self.test)} samples (weeks {valid_end}-{self.T})")
        
        self.cleaning_report.add("Dataset Splitting", 
                                f"Train: {len(self.train)} | Valid: {len(self.valid)} | Test: {len(self.test)} samples")
        
        # Move scale to GPU if available
        if torch.cuda.is_available():
            self.scale = self.scale.cuda()
        
        # RSE denominator
        if len(self.train) > 0:
            train_targets = torch.stack([y for _, y in self.train])
            self.rse_denominator = torch.sqrt(torch.mean(train_targets**2, dim=(0,1))) + 1e-8
            if torch.cuda.is_available():
                self.rse_denominator = self.rse_denominator.cuda()
        else:
            self.rse_denominator = torch.ones(self.m)
        
        # Save cleaning report
        print("\n" + "="*60)
        self.cleaning_report.save()
        print("="*60)
    
    def _impute_missing_values(self, data):
        """Impute missing values using linear interpolation"""
        data_imputed = data.copy()
        
        for region_idx in range(data.shape[1]):
            region_data = data[:, region_idx]
            
            # Find missing indices
            missing_mask = np.isnan(region_data)
            
            if np.any(missing_mask) and not np.all(missing_mask):
                # Get valid indices and values
                valid_indices = np.where(~missing_mask)[0]
                valid_values = region_data[~missing_mask]
                
                # Interpolate
                missing_indices = np.where(missing_mask)[0]
                interpolated_values = np.interp(missing_indices, valid_indices, valid_values)
                
                # Fill in missing values
                data_imputed[missing_indices, region_idx] = interpolated_values
            
            elif np.all(missing_mask):
                # If entire region is missing, use zeros
                data_imputed[:, region_idx] = 0.0
        
        return data_imputed
    
    def get_batches(self, data_list, batch_size, shuffle=True):
        """Generate batches from dataset"""
        if shuffle:
            indices = np.random.permutation(len(data_list))
        else:
            indices = np.arange(len(data_list))
        
        for i in range(0, len(data_list), batch_size):
            batch = [data_list[j] for j in indices[i:i + batch_size]]
            if len(batch) == 0:
                continue
            batch_x = torch.stack([x for x, y in batch])
            batch_y = torch.stack([y for x, y in batch])
            yield batch_x, batch_y
    
    def _relative_error(self, true, pred):
        return np.sqrt(np.mean((true - pred)**2, axis=(0,1))) / (self.rse_denominator.cpu().numpy() + 1e-8)
    
    def _relative_absolute_error(self, true, pred):
        return np.mean(np.abs(true - pred), axis=(0,1)) / (np.mean(np.abs(true), axis=(0,1)) + 1e-8)