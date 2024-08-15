import polars as pl
import time
import argparse

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Process ArXiv JSON dataset.")
    parser.add_argument("--input_json", type=str, default = "/kaggle/input/arxiv/arxiv-metadata-oai-snapshot.json", help="Path to the input NDJSON file.")
    parser.add_argument("--output_json", type=str, default = "/kaggle/working/arxiv_cs.json", help="Path to the output NDJSON file.")
    return parser.parse_args()

# Function to convert the latest 'created' date from the 'versions' field to a specific format
def get_latest_time(element):
    return time.strftime("%d %b %Y", time.strptime(element[-1]['created'], "%a, %d %b %Y %H:%M:%S %Z"))

# Function to convert the 'update_date' field to a specific format
def get_latest_date(element):
    return time.strftime("%d %b %Y", time.strptime(element, "%Y-%m-%d"))

def main():
    args = parse_args()

    # Loading the entire JSON dataset from the input file path
    cs_arxiv_df = pl.read_ndjson(args.input_json)

    # Filtering rows where the 'categories' column contains specific computer science categories
    cs_arxiv_df = cs_arxiv_df.filter(pl.col("categories").str.contains(r"\b(?:cs\.(?:CV|LG|CL|AI|NE|RO))\b", strict=True))

    # Initializing a new column '_time' with default value 0
    cs_arxiv_df = cs_arxiv_df.with_columns(pl.lit(0, dtype=pl.Int64).alias('_time'))

    # Updating the '_time' column with the latest version date or update date
    cs_arxiv_df = cs_arxiv_df.with_columns(
        pl.when(cs_arxiv_df['versions'].is_not_null())
        .then(cs_arxiv_df['versions'].map_elements(get_latest_time,  return_dtype=pl.Utf8))
        .otherwise(cs_arxiv_df['update_date'].map_elements(get_latest_date,  return_dtype=pl.Utf8))
        .alias('_time')
    )

    # Columns to be dropped from the DataFrame
    columns_to_drop = ['versions', 'authors_parsed', 'report-no', 'license', 'submitter']

    # Dropping the specified columns based on the DataFrame type
    cs_arxiv_df = cs_arxiv_df.drop(columns_to_drop)

    # Writing the processed DataFrame to a new NDJSON file specified by the output path
    cs_arxiv_df.write_ndjson(args.output_json)

if __name__ == "__main__":
    main()
