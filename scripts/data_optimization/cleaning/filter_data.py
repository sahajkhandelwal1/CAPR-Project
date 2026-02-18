import pandas as pd
import os

def display_columns(columns):
    print("\nAvailable Columns:")
    for i, col in enumerate(columns):
        print(f"{i+1}. {col}")

def get_column_selection(columns):
    while True:
        print("\nSelect columns to KEEP (example: 1,2,5,8):")
        selection = input("> ").replace(" ", "")

        try:
            indices = [int(x) - 1 for x in selection.split(",")]
            selected = [columns[i] for i in indices]
            return selected
        except:
            print("Invalid input. Try again.")

def filter_csv(input_file, output_file):
    # Load CSV
    print("\nLoading CSV...")
    df = pd.read_csv(input_file)
    columns = df.columns.tolist()

    # Display column names
    display_columns(columns)

    # Get user selection
    selected_cols = get_column_selection(columns)

    # Filter dataframe by columns
    print("\nFiltering CSV by selected columns...")
    filtered_df = df[selected_cols]
    
    # Filter by date - only keep crimes after 2020
    print("Filtering for crimes after 2020...")
    if 'Incident Year' in filtered_df.columns:
        initial_count = len(filtered_df)
        filtered_df = filtered_df[filtered_df['Incident Year'] > 2020]
        final_count = len(filtered_df)
        print(f"Filtered from {initial_count:,} to {final_count:,} records (removed {initial_count - final_count:,} records from 2020 and earlier)")
    else:
        print("Warning: 'Incident Year' column not found. Date filtering skipped.")

    # Save output
    filtered_df.to_csv(output_file, index=False)
    
    # Get file size in MB
    file_size_bytes = os.path.getsize(output_file)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print(f"\n✅ Success! Filtered file saved as: {output_file}")
    print(f"📁 File size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
    print(f"📊 Final dataset contains {len(filtered_df):,} records")

def main():
    print("\n=== Police Crime Data Cleaner ===")
    input_file = input("Enter input CSV filename: ").strip()
    output_file = input("Enter output CSV filename: ").strip()

    filter_csv(input_file, output_file)

if __name__ == "__main__":
    main()
