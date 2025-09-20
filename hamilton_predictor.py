import datetime
import fastf1
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

fastf1.Cache.enable_cache('./f1_cache')


def get_hamilton_historical_data(years_back=5):
    """Fetches Luis Hamilton's race results for previous seasons.
    "Returns a pandas DataFrame.
    """
    current_year = datetime.datetime.now().year
    start_year = current_year - years_back
    years = list(range(start_year, current_year+1))

    race_data_past5years = []
    hamilton_data = {}
    
    for year in years:
        print(f'Fetching data for {year}:')
        try:
            schedule = fastf1.get_event_schedule(year)

            # filter races
            race_events = schedule[schedule['EventName'].str.contains('Grand Prix', case=False, na=False)]

            if race_events.empty:
                print(f" No Grand Prix events found for {year}")
                continue


            for index, event in race_events.iterrows():
                round_num = event['RoundNumber']
                circuit_name = event['Location']

                try:
                    session = fastf1.get_session(year, round_num, 'R')
                    session.load()

                    #find Hamiltons data 
                    ham_results = session.results[session.results['DriverNumber'] == '44']


                    if not ham_results.empty:
                        # fll data in dict
                        race_info ={
                            'year': year,
                            'round': round_num,
                            'circuit': circuit_name,
                            'grid_position': ham_results['GridPosition'].iloc[0],
                            'finish_position': ham_results['Position'].iloc[0],
                            'points': ham_results['Points'].iloc[0],
                            'team': ham_results['TeamName'].iloc[0],
                            'podium': 1 if ham_results['Position'].iloc[0]<= 3 else 0,
                            'status': ham_results['Status'].iloc[0]
                        }

                        race_data_past5years.append(race_info)
                        print(f"  - Round {round_num} ({circuit_name}) :P{race_info['finish_position']}")
                except Exception as e:
                    print(f"   -Could not load Round {round_num} ({circuit_name}): {e}")

                    continue  
        except Exception as e:
            print(f"Could not load schedule for {year}: {e}")
            continue
    

    # Convert the list of dictonaries to DataFrame
    df = pd.DataFrame(race_data_past5years)
    print(f"\nData collection complete! Collected {len(df)} races.")
    return df


historical_data = get_hamilton_historical_data(4)
print(historical_data.head())

def explore_clean_data(df):
    """Explores raw formula1 Dataframe and cleans it by removing DNF races.
    Returns a cleaned pandas Dataframe
    """
    # 1. INITIAL EXPLORATION
    print("\n1. DataFrame Info:")
    print(df.info())
    
    print("\n2. Statistical Summary:")
    print(df.describe())
    
    print("\n3. First 5 rows:")
    print(df.head())
    
    # 2. IDENTIFY DATA ISSUES
    print("\n4. Missing Values Check:")
    missing_values = df.isnull().sum()
    print(missing_values)
    
    print("\n5. Identifying DNF Races:")
    dnf_races = df[df['status'] != 'Finished']
    print(f"Number of DNF races: {len(dnf_races)}")
    print("DNF races:")
    print(dnf_races[['year', 'round', 'circuit', 'status']])
    
    # 3. DATA CLEANING
    print("\n6. Cleaning Data - Removing DNF races...")
    df_clean = df[df['status'] == 'Finished'].copy()
    
    print("\n7. Dropping 'status' column (no longer needed)...")
    df_clean = df_clean.drop(columns=['status'])
    
    print("\n8. Resetting index...")
    df_clean = df_clean.reset_index(drop=True)  # FIXED: Using drop=True parameter
    
    # 4. FINAL CHECK
    print("\n9. Final Cleaned DataFrame Info:")
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {df_clean.shape}")
    print(f"Races removed: {len(df) - len(df_clean)}")
    
    print("\n10. First 5 rows of cleaned data:")
    print(df_clean.head())
    
    return df_clean

# Let's run it on our collected data
print("Starting data cleaning process...")
cleaned_data = explore_clean_data(historical_data)

def create_features(input_df):
    """
        PURPOSE: Creates new predictive features from the raw race data.

        INPUT: input_df (The cleaned DataFrame from Step 3)
        OUTPUT: A DataFrame with new engineered features
    """
    df = input_df.copy()
    # sort data by year and round
    df = df.sort_values(by=['year', 'round']).reset_index(drop=True)
    
    #Circuit history features

    print("Creating 'circuit_avg-finish', feature...")
    df['circuit_avg_finish'] = np.nan  # Initialize with NaN

    for circuit in df['circuit'].unique():
        circuit_mask = df['circuit'] == circuit
        # Calculate expanding mean for this circuit only
        expanding_means = df[circuit_mask]['finish_position'].expanding().mean().shift(1)
        # Assign the values back to the main DataFrame
        df.loc[circuit_mask, 'circuit_avg_finish'] = expanding_means.values
    # Recent form feature (average of last 3 finishes)
    print("3. Creating 'recent_form' feature...")
    df['recent_form'] = df['finish_position'].rolling(window=3, min_periods=1).mean().shift(1)

    # Previous race feature
    print("4. Creating 'previous_finish' feature...")
    df['previous_finish'] = df['finish_position'].shift(1)
    
    # handle missing values
    print(" Handling missing values...")
    overall_avg_finish =df['finish_position'].mean()
    df['circuit_avg_finish'].fillna(overall_avg_finish, inplace=True)
    df['recent_form'].fillna(overall_avg_finish, inplace=True)
    df['previous_finish'].fillna(overall_avg_finish, inplace=True)

    print("Final dataFrame with new features:")
    print(f"Original columns: {list(input_df.columns)}")
    print(f"New columns: {list(df.columns)}")
    print("\nFirst 10 rows with new features:")
    print(df[['year', 'round', 'circuit', 'finish_position', 
              'circuit_avg_finish', 'recent_form', 'previous_finish']].head(10))
    
    return df

print("Starting feature engineering...")
df_with_features = create_features(cleaned_data)


def prepare_for_model(df_with_features):
    """
    Prepares the engineered features for machine learning training.
    Returns X (features), y (target), and feature names.
    """
    print("=== STEP 5: PREPARING FOR MACHINE LEARNING ===")
    
    # 1. DEFINE OUR FEATURES
    print("1. Selecting features for prediction...")
    feature_columns = ['grid_position', 'circuit_avg_finish', 'recent_form', 'previous_finish']
    
    # 2. DEFINE OUR TARGET
    print("2. Defining target variable...")
    target_column = 'podium'  # 1 = podium, 0 = no podium
    
    # 3. CREATE FEATURE MATRIX (X)
    print("3. Creating feature matrix (X)...")
    X = df_with_features[feature_columns]
    
    # 4. CREATE TARGET VECTOR (y)  
    print("4. Creating target vector (y)...")
    y = df_with_features[target_column] 
    # 5. CHECK DATA SHAPES
    print("5. Checking data shapes:")
    print(f"   X shape: {X.shape}") 
    print(f"   y shape: {y.shape}")
    print(f"   They should have the same number of rows: {X.shape[0] == y.shape[0]}")
    
    # 6. PRINT SUMMARY
    print("6. Data Summary:")
    print(f"   - Using {len(feature_columns)} features: {feature_columns}")
    print(f"   - Predicting: {target_column}")
    print(f"   - Total samples: {len(X)}")
    print(f"   - Podium races: {y.sum()}")  # Hint: .sum() to count 1s
    print(f"   - Non-podium races: {len(y) - y.sum()}")
    print(f"   - Podium rate: {y.mean():.2%}")
    
    print("\nFirst 5 rows of features (X):")
    print(X.head())  # Hint: Show first 5 rows
    
    return X, y, feature_columns

# Let's run it on our feature-engineered data
print("Preparing data for machine learning...")
X, y, feature_names = prepare_for_model(df_with_features)
# Verify we have 2025 data
print("\n=== UPDATED DATA TIMEFRAME ===")
print("Years in dataset:", sorted(df_with_features['year'].unique()))
print("Number of races per year:")
print(df_with_features['year'].value_counts().sort_index())

def prepare_for_next_race_prediction(df_with_features):
    """
    Prepares data for predicting the NEXT race based on current form.
    """
    print("=== PREPARING FOR NEXT-RACE PREDICTION ===")
    
    # Sort by date (year and round)
    df = df_with_features.sort_values(by=['year', 'round']).reset_index(drop=True)
    
    # For next-race prediction, we want to use ALL available data
    # The model will learn patterns from all seasons
    feature_columns = ['grid_position', 'circuit_avg_finish', 'recent_form', 'previous_finish']
    target_column = 'podium'
    
    X = df[feature_columns]
    y = df[target_column]
    
    print(f"Using data from {df['year'].min()} to {df['year'].max()}")
    print(f"Total races: {len(df)}")
    print(f"2025 races included: {len(df[df['year'] == 2025])}")
    
    return X, y, feature_columns

# Prepare the data
X, y, feature_names = prepare_for_next_race_prediction(df_with_features)


 
                    

    
