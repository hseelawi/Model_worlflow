#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    This component takes in a csv and does the necessary cleaning steps in preperation for training

    args:
        input_artifact str: path to csv file on Weights and Biases
        ouput_artifact str: path to save cleaned csv file on Weights and Biases
        output_type str: type of output artifact to be logged to Weights and Biases
        output_description str: a verbose description of the output file to be logged to Weight and Biases
        min_price float: a minimum cutoff for instance price
        max_price float: a maximum cutoff for instance price
    """

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Loading csv file")
    df = pd.read_csv(artifact_local_path)

    logger.info("Converting 'last_review' to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    logger.info("Removing price outliers")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    logger.info("Saving csv to disk")
    df.to_csv(args.output_artifact, index=False)

    logger.info("Logs csv to Weights and Biases")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="this component is used to do basic cleaning on the dataset")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="the path to the csv file on Weights and Biases that needs to be cleaned",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="the path to the csv file on Weights and Biases where the cleaned version is uploaded",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="the type of the artifact that will be logged to Weights and Biases",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="description of the output dataset to be logged as part of the artifact to Weights and Biases",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="minimum property price below which a sample is considered and outlier",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="maximum property price above which a sample is considered and outlier",
        required=True
    )

    args = parser.parse_args()

    go(args)
