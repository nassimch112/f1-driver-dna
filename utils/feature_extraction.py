"""
Utility functions for extracting driver DNA features from F1 telemetry data.
"""

import numpy as np
import pandas as pd

def extract_driver_dna(driver, lap, circuit_info=None):
    """
    Extract driver DNA features from a lap's telemetry data.
    
    Parameters:
    -----------
    driver : str
        Driver identifier code (e.g., 'VER', 'HAM')
    lap : fastf1.core.Lap
        Lap object containing telemetry data
    circuit_info : fastf1.core.CircuitInfo, optional
        Circuit information if available
        
    Returns:
    --------
    dict
        Dictionary of extracted driver DNA features
    """
    tel = lap.get_telemetry()
    features = {}
    
    try:
        # 1. Braking features
        brake_intensity = []
        for i in range(1, len(tel)-5):
            brake_i = 1 if tel['Brake'].iloc[i] else 0
            brake_prev = 1 if tel['Brake'].iloc[i-1] else 0
            
            if brake_i > 0 and brake_prev == 0:
                if i+5 < len(tel):
                    brake_count = sum(1 for j in range(i, i+6) if tel['Brake'].iloc[j])
                    intensity = brake_count / 5.0
                    brake_intensity.append(intensity)
        
        # 2. Throttle application
        throttle_intensity = []
        for i in range(1, len(tel)-5):
            if tel['Throttle'].iloc[i] > 20 and tel['Throttle'].iloc[i-1] <= 20:
                if i+5 < len(tel):
                    start_throttle = tel['Throttle'].iloc[i]
                    end_throttle = tel['Throttle'].iloc[i+5]
                    intensity = (end_throttle - start_throttle) / 5
                    throttle_intensity.append(intensity)
        
        # 3. Speed characteristics
        speed_var = tel['Speed'].std() / tel['Speed'].mean() if tel['Speed'].mean() > 0 else np.nan
        
        # 4. Gear selection features
        gear_changes = 0
        short_shifts = 0
        
        for i in range(1, len(tel)-1):
            if tel['nGear'].iloc[i] != tel['nGear'].iloc[i-1]:
                gear_changes += 1
                
                if tel['nGear'].iloc[i] > tel['nGear'].iloc[i-1]:
                    if 'RPM' in tel.columns:
                        max_rpm = tel['RPM'].max()
                        if tel['RPM'].iloc[i-1] < 0.95 * max_rpm:
                            short_shifts += 1
        
        # 5. Corner analysis using minimum speed approach
        speeds = tel['Speed'].values
        distances = tel['Distance'].values if 'Distance' in tel.columns else np.arange(len(speeds))
        normalized_dist = distances / distances[-1] if len(distances) > 0 and distances[-1] > 0 else np.arange(len(speeds)) / len(speeds)
        
        # Find local speed minima (corners)
        corner_indices = []
        for i in range(2, len(speeds)-2):
            if speeds[i] < speeds[i-1] and speeds[i] < speeds[i+1] and speeds[i] < 0.7 * np.max(speeds):
                corner_indices.append(i)
        
        # Calculate corner metrics
        entry_exit_ratios = []
        speed_reductions = []
        
        for idx in corner_indices:
            # Find entry and exit points
            entry_idx = idx
            for j in range(idx, max(0, idx-20), -1):
                if speeds[j] > speeds[j-1]:
                    entry_idx = j
                    break
            
            exit_idx = idx
            for j in range(idx, min(len(speeds)-1, idx+20)):
                if speeds[j] < speeds[j+1]:
                    exit_idx = j
                    break
            
            # Calculate metrics
            if entry_idx < exit_idx:
                entry_speed = speeds[entry_idx]
                min_speed = speeds[idx]
                exit_speed = speeds[exit_idx]
                
                # Only include significant corners
                if entry_speed > min_speed * 1.2 or exit_speed > min_speed * 1.2:
                    entry_exit_ratio = entry_speed / exit_speed if exit_speed > 0 else np.nan
                    speed_reduction = min_speed / max(entry_speed, exit_speed) if max(entry_speed, exit_speed) > 0 else np.nan
                    
                    if not np.isnan(entry_exit_ratio) and not np.isnan(speed_reduction):
                        entry_exit_ratios.append(entry_exit_ratio)
                        speed_reductions.append(speed_reduction)
        
        # 6. Racing line features
        if 'X' in tel.columns and 'Y' in tel.columns:
            # Calculate path smoothness
            dx = np.diff(tel['X'])
            dy = np.diff(tel['Y'])
            distances = np.sqrt(dx**2 + dy**2)
            
            if len(distances) > 5:
                smoothness = np.std(distances) / np.mean(distances)
                features['path_smoothness'] = smoothness
        
        # Compile features
        features.update({
            'avg_brake_intensity': np.mean(brake_intensity) if brake_intensity else np.nan,
            'avg_throttle_intensity': np.mean(throttle_intensity) if throttle_intensity else np.nan,
            'speed_variability': speed_var,
            'gear_changes': gear_changes / (distances[-1]/1000) if 'Distance' in tel.columns and distances[-1] > 0 else gear_changes / 5,
            'short_shift_ratio': short_shifts / max(1, gear_changes),
            'entry_exit_bias': np.mean(entry_exit_ratios) if entry_exit_ratios else np.nan,
            'avg_corner_speed_reduction': np.mean(speed_reductions) if speed_reductions else np.nan,
        })
    
    except Exception as e:
        print(f"Error extracting features for {driver}: {e}")
    
    return features