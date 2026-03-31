import logging
import sys
from pathlib import Path
from typing import List

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

from ..data_io import read_pair_data
from .feature_extractor import FeatureExtractor
from ..paths import MODEL_DIR, SYSTEMOUTPUTS_DIR, TEST_DATA_PATH, TRAIN_DATA_PATH

logger = logging.getLogger(__name__)

BEST_MODEL_PATH = MODEL_DIR / "best_svc_model_C_0.01.pkl"



    
def train_best_model(
    X: np.ndarray,
    y: np.ndarray,
    c_values: List[float],
    model_save_path: Path,
) -> LinearSVC:
    """Train and save the best LinearSVC with grid search."""
    param_grid = {"C": c_values}

    # Initialize the LinearSVC classifier
    svm_clf = LinearSVC(dual=False, max_iter=10000, class_weight='balanced')

    # Initialize GridSearchCV with 3-fold cross-validation, parallel processing, and accuracy as the scoring metric
    grid_search = GridSearchCV(svm_clf, param_grid, cv=3, n_jobs=-1, scoring='accuracy')

    # Fit GridSearchCV
    grid_search.fit(X, y)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_C = grid_search.best_params_['C']
    best_accuracy = grid_search.best_score_

    # Save the best model to disk
    joblib.dump(best_model, model_save_path)
    logger.info("Best model with C=%s saved as %s", best_C, model_save_path)
    logger.info("Best accuracy: %s", best_accuracy)

    return best_model



def output_predictions(modelfile: Path, testfilename: Path, outfile: Path) -> None:
    # Read in the test data and extract features
    test_data, _ = read_pair_data(testfilename)
    feature_extractor = FeatureExtractor()
    feature_extractor.load_data(test_data, extract_labels=False)
    X_test, _ = feature_extractor.extract_features()

    logger.info(
        "Extracted %s test instances with %s features ...",
        X_test.shape[0],
        X_test.shape[1],
    )

    if X_test.shape[0] == 0:
        logger.info("No test instances to predict. Exiting ...")
        sys.exit(0)
    
    # Load the pre-trained model
    classifier = joblib.load(modelfile)

    # Open output file to write results
    with open(outfile, "w") as outf:
        for features in X_test:
            prob = classifier._predict_proba_lr([features])[0][1]

            if prob >= 0.5:
                outf.write(f"true\t{prob:.4f}\n")
            else:
                outf.write(f"false\t{prob:.4f}\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train_data, _ = read_pair_data(TRAIN_DATA_PATH)

    # remove the data with None label
    filtered_train_data = [item for item in train_data if item[0] is not None]
    logger.info("Read in %s valid training data ...", len(filtered_train_data))

    # check whether the best model already exists
    # if not, train the model, search for the best regularization parameter C and save the best model
    if not BEST_MODEL_PATH.exists():
        feature_extractor = FeatureExtractor()
        feature_extractor.load_data(filtered_train_data)
        X, y = feature_extractor.extract_features()
        logger.info("Extracted %s training instances with %s features ...", X.shape[0], X.shape[1])
        logger.info("Training the model ...")
        c_values = [0.01, 0.1, 1, 10, 100]
        train_best_model(X, y, c_values, BEST_MODEL_PATH)
    else:
        logger.info("Best pre-trained model already exists at %s", BEST_MODEL_PATH)

    # Output predictions on the test data
    output_predictions(BEST_MODEL_PATH, TEST_DATA_PATH, SYSTEMOUTPUTS_DIR / "PIT2015_SVC_04.output")


if __name__ == "__main__":
    main()


