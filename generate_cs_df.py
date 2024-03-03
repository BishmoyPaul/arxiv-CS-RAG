import argparse
import polars as pl
import pandas as pd
import time

if __name__ == "__main__":
    # Code generation prompt
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--source_json', type=str, help='Path to the input file')
    parser.add_argument('--output_json', type=str, help='Path to the output file')

    args = parser.parse_args()
    
    source_json = args.source_json
    output_json = args.output_json

    cs_arxiv_df = pl.read_ndjson(source_json)
    cs_arxiv_df = cs_arxiv_df.filter(pl.col("categories").str.contains(r"\b(?:cs\.(?:CV|LG|CL|AI|NE|RO))\b", strict = True))

    def get_latest_time(element):
        return time.strftime("%d %b %Y",time.strptime(element[-1]['created'], "%a, %d %b %Y %H:%M:%S %Z"))
        #return element[-1]['created']

    def get_latest_date(element):
        return time.strftime("%d %b %Y", time.strptime(element, "%Y-%m-%d"))

    cs_arxiv_df = cs_arxiv_df.with_columns(pl.lit(0, dtype = pl.Int64).alias('_time'))

    cs_arxiv_df = cs_arxiv_df.with_columns(
        pl.when(cs_arxiv_df['versions'].is_not_null())
        .then(cs_arxiv_df['versions'].map_elements(get_latest_time))
        .otherwise(cs_arxiv_df['update_date'].map_elements(get_latest_date))
        .alias('_time')
    )

    columns_to_drop = ['versions','authors_parsed','report-no','license','submitter']
    if isinstance(cs_arxiv_df, pd.DataFrame):
        cs_arxiv_df = cs_arxiv_df.drop(labels = columns_to_drop, axis = 1)
    elif isinstance(cs_arxiv_df, pl.DataFrame):
        cs_arxiv_df = cs_arxiv_df.drop(columns_to_drop)
    else:
        raise Exception
        
    cs_arxiv_df.write_ndjson(output_json)

