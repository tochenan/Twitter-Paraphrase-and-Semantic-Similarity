import numpy as np
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import joblib
from feature_extractor import FeatureExtractor
import sys

# Set the paths to the training and test data, and the path to save the best model
project_path = Path(__file__).parent.parent
train_data_path = project_path / "data/train.data"
test_data_path = project_path / "data/test.data"
best_model_path = project_path / "model/best_svc_model_C_0.01.pkl"


# read in data, getting the readin data function from baseline_logisticregression.py
def readInData(filename):

    data = []
    trends = set([])
    
    (trendid, trendname, origsent, candsent, judge, origsenttag, candsenttag) = (None, None, None, None, None, None, None)
    
    for line in open(filename):
        line = line.strip()
        #read in training or dev data with labels
        if len(line.split('\t')) == 7:
            (trendid, trendname, origsent, candsent, judge, origsenttag, candsenttag) = line.split('\t')
        #read in test data without labels
        elif len(line.split('\t')) == 6:
            (trendid, trendname, origsent, candsent, origsenttag, candsenttag) = line.split('\t')
        else:
            continue
        
        #if origsent == candsent:
        #    continue
        
        trends.add(trendid)
         
        if judge == None:
            data.append((judge, origsent, candsent, trendid))
            continue

        # ignoring the training/test data that has middle label 
        if judge[0] == '(':  # labelled by Amazon Mechanical Turk in format like "(2,3)"
            nYes = eval(judge)[0]            
            if nYes >= 3:
                amt_label = True
                data.append(( amt_label, origsent, candsent, trendid))
            elif nYes <= 1:
                amt_label = False
                data.append(( amt_label, origsent, candsent, trendid))   
        elif judge[0].isdigit():   # labelled by expert in format like "2"
            nYes = int(judge[0])
            if nYes >= 4:
                expert_label = True
                data.append(( expert_label, origsent, candsent, trendid))
            elif nYes <= 2:
                expert_label = False
                data.append(( expert_label, origsent, candsent, trendid))     
            else:
                expert_label = None
                data.append(( expert_label, origsent, candsent, trendid))
        else:
            continue
                
    return data, trends



    
def obtain_best_model(X, y, C_values, model_save_path):
    
    param_grid = {'C': C_values}

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
    print(f"Best model with C={best_C} saved as {model_save_path}")
    print(f"Best accuracy: {best_accuracy}")

    return best_model



def OutputPredictions(modelfile, testfilename, outfile):
    # Read in the test data and extract features
    test_data, _ = readInData(testfilename)  
    feature_extractor = FeatureExtractor()
    feature_extractor.load_data(test_data,extract_labels=False)
    X_test, _ = feature_extractor.extract_features()


    print(f"Extracted {X_test.shape[0]} test instances with {X_test.shape[1]} features ...")

    if X_test.shape[0] == 0:
        print("No test instances to predict. Exiting ...")
        sys.exit(0)
    
    # Load the pre-trained model
    with open(modelfile, 'rb') as inmodel:
        classifier = joblib.load(inmodel)

    # Open output file to write results
    with open(outfile, 'w') as outf:
        for i, features in enumerate(X_test):
            prob = classifier._predict_proba_lr([features])[0][1]  # Probability of the positive class

            if prob >= 0.5:
                outf.write(f"true\t{prob:.4f}\n")
            else:
                outf.write(f"false\t{prob:.4f}\n")

    
if __name__ == "__main__":

    train_data, _ = readInData(train_data_path)

    # remove the data with None label
    filtered_train_data = [item for item in train_data if item[0] is not None]
    
    print(f"Read in {len(filtered_train_data)} valid training data ...")
  
    # check whether the best model already exists
    # if not, train the model, search for the best regularization parameter C and save the best model
    if not best_model_path.exists():
        feature_extractor = FeatureExtractor()
        feature_extractor.load_data(filtered_train_data)
        X, y = feature_extractor.extract_features()
        print(f"Extracted {X.shape[0]} training instances with {X.shape[1]} features ...")
        print(f"Training the model ...")
        C_values = [0.01, 0.1, 1, 10, 100]
        obtain_best_model(X, y, C_values,best_model_path)
    else:
        print(f"Best pre-trained model already exists at {best_model_path}")

    # Output predictions on the test data
    OutputPredictions(best_model_path, test_data_path, f"{project_path}/systemoutputs/PIT2015_SVC_04.output")


