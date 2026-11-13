import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.data.load_data import (
    load_csv,
    create_data_version
)


def main():

    # Dataset v1
    df_v1 = load_csv(
        "data/raw/dataset_v1.csv"
    )

    metadata_v1 = create_data_version(
        df=df_v1,
        version_id="v1",
        description=(
            "Reference dataset used "
            "for model training"
        ),
        output_dir="data_versions"
    )

    print("Dataset v1 metadata:")
    print(metadata_v1)

    # Dataset v2
    df_v2 = load_csv(
        "data/raw/dataset_v2.csv"
    )

    metadata_v2 = create_data_version(
        df=df_v2,
        version_id="v2",
        description=(
            "Second dataset version with "
            "more samples and feature distribution shift"
        ),
        output_dir="data_versions"
    )

    print("\nDataset v2 metadata:")
    print(metadata_v2)


if __name__ == "__main__":
    main()