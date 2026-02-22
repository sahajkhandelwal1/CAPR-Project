import pandas as pd
import os
import sys
import argparse
from collections import defaultdict

def clean_filename(filename):
    """Remove quotes and whitespace from filename."""
    if not filename:
        return filename
    filename = filename.strip()
    # Remove surrounding quotes (single or double)
    if (filename.startswith('"') and filename.endswith('"')) or \
       (filename.startswith("'") and filename.endswith("'")):
        filename = filename[1:-1]
    return filename.strip()

def display_columns(columns):
    """Display all available columns with indices."""
    print("\n" + "="*60)
    print("Available Columns:")
    print("="*60)
    for i, col in enumerate(columns, 1):
        print(f"{i:2d}. {col}")

def get_unique_values(df, column):
    """Get unique values for a column, excluding NaN."""
    unique_vals = df[column].dropna().unique().tolist()
    # Convert to string and sort for consistent display
    unique_vals = sorted([str(v) for v in unique_vals])
    return unique_vals

def score_column_values(df, column):
    """Interactive scoring of unique values in a column."""
    print(f"\n{'='*60}")
    print(f"Scoring Column: {column}")
    print(f"{'='*60}")
    
    unique_vals = get_unique_values(df, column)
    
    if not unique_vals:
        print(f"No values found in column '{column}'. Skipping...")
        return {}
    
    print(f"\nFound {len(unique_vals)} unique values in '{column}':")
    print("-" * 60)
    
    # Display values with indices
    value_scores = {}
    for i, val in enumerate(unique_vals, 1):
        print(f"{i:3d}. {val}")
    
    print("\nEnter scores for each value.")
    print("Score Range: 1-100 (1 = safest/lowest risk, 100 = most dangerous/highest risk)")
    print("Format: value_index:score (e.g., 1:10, 2:75, 3:95)")
    print("Or enter 'skip' to skip this column, or 'auto' to score all at once.")
    print("For 'auto', enter a single score (1-100) that will apply to all values.")
    
    while True:
        user_input = input("\n> ").strip().lower()
        
        if user_input == 'skip':
            return {}
        
        if user_input == 'auto':
            try:
                score = float(input("Enter score to apply to all values (1-100): "))
                if not (1 <= score <= 100):
                    print("❌ Score must be between 1 and 100. Please try again.")
                    continue
                value_scores = {val: score for val in unique_vals}
                print(f"✓ Applied score {score} to all {len(unique_vals)} values")
                break
            except ValueError:
                print("❌ Invalid score. Please enter a number between 1 and 100.")
                continue
        
        # Parse individual scores
        try:
            pairs = user_input.split(',')
            value_scores = {}
            for pair in pairs:
                pair = pair.strip()
                if ':' not in pair:
                    print(f"Invalid format: {pair}. Use 'index:score' format.")
                    continue
                idx_str, score_str = pair.split(':', 1)
                idx = int(idx_str.strip()) - 1
                score = float(score_str.strip())
                
                if 0 <= idx < len(unique_vals):
                    val = unique_vals[idx]
                    if not (1 <= score <= 100):
                        print(f"❌ Score {score} for '{val}' must be between 1 and 100. Skipping.")
                        continue
                    value_scores[val] = score
                    print(f"✓ {val}: {score}")
                else:
                    print(f"❌ Invalid index: {idx + 1}. Must be between 1 and {len(unique_vals)}")
            
            if value_scores:
                # Check if all values are scored
                unscored = [v for v in unique_vals if v not in value_scores]
                if unscored:
                    print(f"\n⚠ Warning: {len(unscored)} values not scored: {unscored[:5]}{'...' if len(unscored) > 5 else ''}")
                    response = input("Continue anyway? (y/n): ").strip().lower()
                    if response != 'y':
                        continue
                break
        except ValueError as e:
            print(f"❌ Invalid input: {e}. Please enter numbers in format 'index:score' where score is 1-100.")
    
    return value_scores

def get_column_weights(columns):
    """Get weights for each column for weighted average calculation."""
    print(f"\n{'='*60}")
    print("Column Weighting")
    print(f"{'='*60}")
    print("\nEnter weights for each column (used in weighted average calculation).")
    print("Weights can be any positive numbers. They will be normalized automatically.")
    print("Format: column_index:weight (e.g., 1:0.5, 2:1.0, 3:2.0)")
    print("Or enter 'equal' to give all columns equal weight.")
    
    weights = {}
    
    while True:
        user_input = input("\n> ").strip().lower()
        
        if user_input == 'equal':
            equal_weight = 1.0 / len(columns)
            weights = {col: equal_weight for col in columns}
            print(f"✓ Applied equal weight ({equal_weight:.4f}) to all {len(columns)} columns")
            break
        
        try:
            pairs = user_input.split(',')
            weights = {}
            for pair in pairs:
                pair = pair.strip()
                if ':' not in pair:
                    print(f"Invalid format: {pair}. Use 'index:weight' format.")
                    continue
                idx_str, weight_str = pair.split(':', 1)
                idx = int(idx_str.strip()) - 1
                weight = float(weight_str.strip())
                
                if weight < 0:
                    print(f"Warning: Negative weight for column {idx + 1}. Using absolute value.")
                    weight = abs(weight)
                
                if 0 <= idx < len(columns):
                    col = columns[idx]
                    weights[col] = weight
                    print(f"✓ {col}: {weight}")
                else:
                    print(f"Invalid index: {idx + 1}. Must be between 1 and {len(columns)}")
            
            if weights:
                # Normalize weights
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {col: w / total_weight for col, w in weights.items()}
                    print(f"\n✓ Weights normalized (total: {sum(weights.values()):.4f})")
                break
        except ValueError as e:
            print(f"❌ Invalid input: {e}. Please enter numbers in format 'index:weight' (e.g., 1:0.5, 2:1.0).")
    
    return weights

def calculate_scores(df, column_scores, column_weights):
    """Calculate weighted average score for each row."""
    print("\n" + "="*60)
    print("Calculating Scores...")
    print("="*60)
    
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Calculate score for each row
    row_scores = []
    
    for idx, row in df.iterrows():
        total_score = 0.0
        total_weight = 0.0
        
        for column, weight in column_weights.items():
            if column not in column_scores or column not in df.columns:
                continue
            
            value = str(row[column]) if pd.notna(row[column]) else None
            
            if value and value in column_scores[column]:
                score = column_scores[column][value]
                total_score += score * weight
                total_weight += weight
        
        # Calculate weighted average (handle division by zero)
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.0
        
        row_scores.append(final_score)
    
    # Add score column
    result_df['Score'] = row_scores
    
    print(f"✓ Calculated scores for {len(row_scores):,} rows")
    print(f"  Score range: {min(row_scores):.2f} - {max(row_scores):.2f} (1=safest, 100=most dangerous)")
    print(f"  Average score: {sum(row_scores) / len(row_scores):.2f}")
    
    return result_df

def score_csv(input_file, output_file):
    """Main function to score a CSV file."""
    # Load CSV
    print("\n" + "="*60)
    print("CSV Scoring System")
    print("="*60)
    print(f"\nLoading CSV: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Loaded {len(df):,} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    columns = df.columns.tolist()
    
    # Display columns
    display_columns(columns)
    
    # Select columns to score
    print("\n" + "-"*60)
    print("Select columns to score (leave empty to score all columns):")
    print("Format: column indices separated by commas (e.g., 1,2,5,8)")
    selection = input("> ").strip()
    
    if selection:
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(",")]
            selected_cols = [columns[i] for i in indices if 0 <= i < len(columns)]
            print(f"✓ Selected {len(selected_cols)} columns: {', '.join(selected_cols)}")
        except:
            print("Invalid selection. Using all columns.")
            selected_cols = columns
    else:
        selected_cols = columns
        print(f"✓ Using all {len(columns)} columns")
    
    # Score values for each column
    column_scores = {}
    
    print("\n" + "="*60)
    print("VALUE SCORING PHASE")
    print("="*60)
    
    for col in selected_cols:
        scores = score_column_values(df, col)
        if scores:
            column_scores[col] = scores
    
    if not column_scores:
        print("\n❌ No columns were scored. Exiting.")
        return
    
    print(f"\n✓ Scored {len(column_scores)} columns")
    
    # Get column weights
    scored_columns = list(column_scores.keys())
    column_weights = get_column_weights(scored_columns)
    
    if not column_weights:
        print("\n❌ No column weights provided. Exiting.")
        return
    
    # Calculate scores
    result_df = calculate_scores(df, column_scores, column_weights)
    
    # Save output
    print("\n" + "="*60)
    print("Saving Results...")
    print("="*60)
    
    try:
        result_df.to_csv(output_file, index=False)
        
        # Get file size
        file_size_bytes = os.path.getsize(output_file)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        print(f"\n✅ Success! Scored file saved as: {output_file}")
        print(f"📁 File size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
        print(f"📊 Dataset contains {len(result_df):,} records")
        print(f"📈 Score column added: 'Score'")
        
        # Display summary statistics
        print("\n" + "-"*60)
        print("Score Summary Statistics (1=safest, 100=most dangerous):")
        print("-"*60)
        print(f"  Minimum Score: {result_df['Score'].min():.4f}")
        print(f"  Maximum Score: {result_df['Score'].max():.4f}")
        print(f"  Average Score: {result_df['Score'].mean():.4f}")
        print(f"  Median Score:  {result_df['Score'].median():.4f}")
        print(f"  Std Deviation: {result_df['Score'].std():.4f}")
        
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")

def main():
    parser = argparse.ArgumentParser(description='CSV Scoring System - Score and weight CSV columns')
    parser.add_argument('input_file', nargs='?', help='Input CSV filename')
    parser.add_argument('output_file', nargs='?', help='Output CSV filename')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("CSV Scoring System")
    print("="*60)
    
    # Get input file
    if args.input_file:
        input_file = clean_filename(args.input_file)
    else:
        input_file = clean_filename(input("\nEnter input CSV filename: "))
        if not input_file:
            print("❌ Input filename required.")
            return
    
    # Get output file
    if args.output_file:
        output_file = clean_filename(args.output_file)
    else:
        output_file = clean_filename(input("Enter output CSV filename: "))
        if not output_file:
            print("❌ Output filename required.")
            return
    
    score_csv(input_file, output_file)

if __name__ == "__main__":
    main()
