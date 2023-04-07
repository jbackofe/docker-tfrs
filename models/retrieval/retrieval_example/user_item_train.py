import os
import pprint
import tempfile
import argparse
import logging
import sys

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs

tfds.disable_progress_bar()

#Use this format (%Y-%m-%dT%H:%M:%SZ) to record timestamp of the metrics
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    level=logging.DEBUG)

def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', 
                      type=str, 
                      default='/logs',
                      help='Name of the model folder.')
    parser.add_argument('--embedding_dim',
                      type=int,
                      default=32,
                      help='The embedding dimension for both user and item')
    parser.add_argument('--epochs',
                      type=int,
                      default=3,
                      help='The number of training passes')
    parser.add_argument('--learning_rate',
                      type=float,
                      default=0.1,
                      help='Learning rate for training.')
    parser.add_argument('--export_folder',
                      type=str,
                      default='model',
                      help='folder to save model')

    args, _ = parser.parse_known_args(args=argv[1:])

    return args

# Katib parses metrics in this format: <metric-name>=<metric-value>.
class StdOutCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logging.info(
            "Epoch {:4d}: factorized_top_k/top_1_categorical_accuracy={:.4f}".format(
                epoch+1, logs["factorized_top_k/top_1_categorical_accuracy"]
            )
        )

########## IMPORT DATA ##########
def importData():
    # Ratings data.
    ratings = tfds.load("movielens/100k-ratings", split="train")
    # Features of all the available movies.
    movies = tfds.load("movielens/100k-movies", split="train")

    return ratings, movies

########## PREPARE DATASET ##########
def prepareData(ratings, movies):
    ratings = ratings.map(lambda x: {
        "movie_title": x["movie_title"],
        "user_id": x["user_id"],
    })
    movies = movies.map(lambda x: x["movie_title"])

    return ratings, movies

# Shuffle and split into train/test
def splitData(ratings, seed=42):
    tf.random.set_seed(seed)
    shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
    train = shuffled.take(80_000)
    test = shuffled.skip(80_000).take(20_000)
    return train, test

# Get unique user ids and movie titles
# This is for mapping raw feature values to embeddings
def getUnique(ratings, movies):
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])
    movie_titles = movies.batch(1_000)

    unique_user_ids = np.unique(np.concatenate(list(user_ids)))
    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    return unique_user_ids, unique_movie_titles

# Construct the full model
# Takes in raw features and returns a loss.
class MovielensModel(tfrs.Model):
    def __init__(self, user_model, movie_model, task):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        positive_movie_embeddings = self.movie_model(features["movie_title"])

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_movie_embeddings)

def main(argv=None):
    args = parse_arguments(sys.argv if argv is None else argv)

    # Import MovieLens dataset
    ratings, movies = importData()

    # Prepare the dataset
    ratings, movies = prepareData(ratings, movies)

    # Train/Test split
    train, test = splitData(ratings)

    # Get unique user ids and movie titles
    unique_user_ids, unique_movie_titles = getUnique(ratings, movies)

    # Corresponds to a classic matrix factorization approach
    user_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
        # We add an additional embedding to account for unknown tokens.
        tf.keras.layers.Embedding(len(unique_user_ids) + 1, args.embedding_dim)
    ])

    # The candidate tower
    movie_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_movie_titles, mask_token=None),
        tf.keras.layers.Embedding(len(unique_movie_titles) + 1, args.embedding_dim)
    ])

    # Use FactorizedTopK metric to compare affinity score of positive (user, movie) pairs
    # that the model calculates to the scores of all other possible candidates.
    # If the score for the positive pair is higher than for all other candidates, our model is highly accurate
    metrics = tfrs.metrics.FactorizedTopK(
        candidates=movies.batch(128).map(movie_model)
    )

    # Define a task (combination of loss function and metric computation)
    # Default loss for Retrieval is CategoricalCrossentropy.
    task = tfrs.tasks.Retrieval(
        metrics=metrics
    )
    
    # Instantiate the model
    model = MovielensModel(user_model, movie_model, task)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(args.learning_rate))

    # Shuffle, batch, and cache the training and evaluation data
    cached_train = train.shuffle(100_000).batch(8192).cache()
    cached_test = test.batch(4096).cache()

    # Define callbacks for logging
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.log_dir)

    std_out_callback = StdOutCallback()

    # Train the model 
    # Note: turn metric calculation for training off when training large dataset
    model.fit(cached_train, epochs=args.epochs, callbacks=[tensorboard_callback, std_out_callback])

    # Evaluate model
    eval_dict = model.evaluate(cached_test, return_dict=True)

    ########## SAVE MODEL ##########
    # Create a model that takes in raw query features
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)

    # Save the index.
    bruteForce_path = os.path.join(args.export_folder, 'bruteForce')
    tf.saved_model.save(index, bruteForce_path)

    # Create approximate retrieval index (faster than BruteForce)
    scann_index = tfrs.layers.factorized_top_k.ScaNN(model.user_model)

    # Save the index.
    scann_path = os.path.join(args.export_folder, 'scann')
    tf.saved_model.save(
        scann_index,
        scann_path,
        options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"])
    )

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  main()