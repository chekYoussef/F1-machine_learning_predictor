# ===== IMPORTS =====
# Import your data preparation functions
import fastf1
from hamilton_predictor import (
    get_hamilton_historical_data,
    explore_clean_data, 
    create_features,
    prepare_for_model
)

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, skipping...")

# ===== PREDICTION FUNCTIONS =====
def calculate_circuit_average(df, circuit_name):
    """Calculate Hamilton's average finish at a specific circuit."""
    circuit_races = df[df['circuit'] == circuit_name]
    if len(circuit_races) > 0:
        return circuit_races['finish_position'].mean()
    return df['finish_position'].mean()

def calculate_recent_form(df):
    """Calculate average of last 3 finishes."""
    recent_races = df.sort_values(by=['year', 'round']).tail(3)
    return recent_races['finish_position'].mean()

def make_prediction_with_actual_data(best_model, df_with_features, next_circuit_name, actual_grid_position, scaler=None):
    """Make prediction using actual grid position data."""
    print(f"=== MAKING PREDICTION FOR {next_circuit_name.upper()} ===")
    
    # Get the most recent race data
    latest_race = df_with_features.sort_values(by=['year', 'round']).iloc[-1]
    
    # Calculate features for the next race
    next_race_features = {
        'grid_position': actual_grid_position,
        'circuit_avg_finish': calculate_circuit_average(df_with_features, next_circuit_name),
        'recent_form': calculate_recent_form(df_with_features),
        'previous_finish': latest_race['finish_position']
    }
    
    print("📊 Features for next race:")
    for feature, value in next_race_features.items():
        print(f"  {feature}: {value}")
    
    # Convert to array for prediction
    feature_array = [next_race_features['grid_position'], 
                    next_race_features['circuit_avg_finish'],
                    next_race_features['recent_form'],
                    next_race_features['previous_finish']]
    
    # Apply scaling if provided
    if scaler is not None:
        feature_array = scaler.transform([feature_array])
    
    # Make prediction
    podium_prob = best_model.predict_proba(feature_array)[0][1]
    
    print(f"\n🎯 LEWIS HAMILTON PODIUM PROBABILITY: {podium_prob:.2%}")
    
    # Interpretation
    if podium_prob > 0.7:
        print("🏆 HIGH CHANCE OF PODIUM! Great qualifying position!")
    elif podium_prob > 0.5:
        print("📈 GOOD CHANCE OF PODIUM")
    elif podium_prob > 0.3:
        print("📊 MODERATE CHANCE OF PODIUM")
    else:
        print("📉 LOW CHANCE OF PODIUM")
    
    return podium_prob, next_race_features

def visualize_qualifying_impact(best_model, df_with_features, next_circuit_name, scaler=None):
    """
    Visualizes how podium probability changes based on qualifying position.
    """
    print(f"\n=== VISUALIZING QUALIFYING IMPACT FOR {next_circuit_name.upper()} ===")
    
    # Get the most recent race data for other features
    latest_race = df_with_features.sort_values(by=['year', 'round']).iloc[-1]
    circuit_avg = calculate_circuit_average(df_with_features, next_circuit_name)
    recent_form = calculate_recent_form(df_with_features)
    previous_finish = latest_race['finish_position']
    
    # Test different qualifying positions (P1 to P20)
    qualifying_positions = list(range(1, 21))
    probabilities = []
    
    for grid_pos in qualifying_positions:
        # Create feature array for this qualifying position
        feature_array = [grid_pos, circuit_avg, recent_form, previous_finish]
        
        # Apply scaling if needed
        if scaler is not None:
            feature_array = scaler.transform([feature_array])
        
        # Get probability
        prob = best_model.predict_proba(feature_array)[0][1]
        probabilities.append(prob)
    
    # Create the visualization
    plt.figure(figsize=(12, 8))
    
    # Plot the probability curve
    plt.plot(qualifying_positions, probabilities, 'b-', linewidth=3, marker='o', markersize=6)
    plt.fill_between(qualifying_positions, probabilities, alpha=0.2, color='blue')
    
    # Add markers for key positions
    key_positions = [1, 5, 10, 15, 20]
    for pos in key_positions:
        idx = pos - 1
        plt.annotate(f'{probabilities[idx]:.1%}', 
                    xy=(pos, probabilities[idx]),
                    xytext=(5, 5), textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
                    fontweight='bold')
    
    # Add reference lines
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% Probability')
    plt.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='30% Probability')
    plt.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='70% Probability')
    
    # Customize the plot
    plt.xlabel('Qualifying Position', fontsize=14, fontweight='bold')
    plt.ylabel('Podium Probability', fontsize=14, fontweight='bold')
    plt.title(f'Lewis Hamilton Podium Probability vs Qualifying Position\n{next_circuit_name.title()} Grand Prix', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(qualifying_positions)
    plt.ylim(0, 1)
    
    insight_text = (
        f"Qualifying Impact Insights:\n"
        f"• P1: {probabilities[0]:.1%} podium chance\n"
        f"• P5: {probabilities[4]:.1%} podium chance\n" 
        f"• P10: {probabilities[9]:.1%} podium chance\n"
        f"• P15: {probabilities[14]:.1%} podium chance\n"
        f"• P20: {probabilities[19]:.1%} podium chance\n"
        f"• Critical threshold: P{next((i for i, p in enumerate(probabilities, 1) if p < 0.5), '>20')}+"
    )

    plt.text(1.02, 0.5, insight_text, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
    
    plt.tight_layout()
    plt.show()
    
    return qualifying_positions, probabilities




# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    # Data Preparation
    print("=== STEP 1: DATA COLLECTION ===")
    historical_data = get_hamilton_historical_data(4)

    print("\n=== STEP 2: DATA CLEANING ===")
    cleaned_data = explore_clean_data(historical_data)

    print("\n=== STEP 3: FEATURE ENGINEERING ===")
    df_with_features = create_features(cleaned_data)

    print("\n=== STEP 4: MODEL PREPARATION ===")
    X, y, feature_names = prepare_for_model(df_with_features)

    # Model Training
    print("\n=== STEP 5: MODEL TRAINING ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

    print("\n=== MODEL TRAINING RESULTS ===")
    results = {}
    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'uses_scaling': name == 'Logistic Regression'
        }
        
        print(f"{name}:")
        print(f"  Accuracy: {accuracy:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}, ROC AUC: {roc_auc:.3f}")

    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
    best_model = results[best_model_name]['model']
    uses_scaling = results[best_model_name]['uses_scaling']
    print(f"\n🎯 Best model: {best_model_name} (ROC AUC: {results[best_model_name]['roc_auc']:.3f})")

    next_circuit_name = "Azerbaijan"
    actual_grid_position = 12

    # Use your function
    scaling_to_use = scaler if uses_scaling else None
    probability, features = make_prediction_with_actual_data(
    best_model, 
    df_with_features, 
    next_circuit_name, 
    actual_grid_position,  
    scaler=scaling_to_use
    )

    print(f"\nBased on:")
    print(f"• Qualifying: P{actual_grid_position}")
    print(f"• Recent form: {features['recent_form']:.1f} avg finish")
    print(f"• Previous race: P{features['previous_finish']}")
    print(f"• Circuit history: {features['circuit_avg_finish']:.1f} avg at {next_circuit_name}")

    visualize_qualifying_impact(best_model, df_with_features, next_circuit_name, 
                           scaler=scaler if uses_scaling else None)