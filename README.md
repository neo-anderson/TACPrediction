# TACPrediction
Predicting TAC values based on accelerometer data

## Instructions
- The analysis is based on [Bar Crawl: Detecting Heavy Drinking Data Set](http://archive.ics.uci.edu/ml/datasets/Bar+Crawl%3A+Detecting+Heavy+Drinking) from UCI.
- Download the dataset and copy the data folder into `src` directory
- Make sure you have the required libraries installed if you want to run the code. Use the included `requirements.txt` to install the libraries using pip. `pip install -r requirements.txt`
- Open Analysis.ipynb for the full analysis details.
- Run `python app.py` to run the final python code. Run `python app.py --preprocess` to preprocess the data and regenerate the cache if needed.

## Sample Run
```
Training the neural network
Distribution of Intoxicated class in the dataset is 30.0%
Cross validation F1 scores: [0.83306031 0.82328565 0.83571902 0.8187543  0.83191972]
Average cross validation F1: 0.8285477974485742
Cross validation time: 111.68637895584106 seconds
Time taken to train on full training set: 18.890563011169434 seconds

              precision    recall  f1-score   support

       Sober       0.91      0.85      0.88      3236
 Intoxicated       0.69      0.80      0.74      1388

    accuracy                           0.83      4624
   macro avg       0.80      0.82      0.81      4624
weighted avg       0.84      0.83      0.84      4624

Overall accuracy: 0.833044982698962
Sober accuracy: 0.8464153275648949
Intoxicated accuracy: 0.8018731988472623
Saved predictions to cache/predictions.csv
Total time taken: 113.113 seconds
```

