import pandas as pd

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

    # Filter dataframe
    print("\nFiltering CSV...")
    filtered_df = df[selected_cols]

    # Save output
    filtered_df.to_csv(output_file, index=False)
    print(f"\n✅ Success! Filtered file saved as: {output_file}")

def main():
    print("\n=== Police Crime Data Cleaner ===")
    input_file = input("Enter input CSV filename: ").strip()
    output_file = input("Enter output CSV filename: ").strip()

    filter_csv(input_file, output_file)

if __name__ == "__main__":
    main()
