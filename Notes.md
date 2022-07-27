# Notes
## Data Exploration

### Examine class balance
 -> Large class imbalance
    -> Update the data
    -> Synthetic Generation
    -> Dynamic Sampling
    -> Change the prediction-function threshold
    -> Favour models that allow for class imbalance

#### Data Truncation
We find that ~3500 documents that are truncated examples of larger documents

This means that most documents are not fully contextualised.

### Examine typical data

## AI Best Practices
1. No cross validation - > Pure overfitting
    -> Enable 5 fold validation

2. Define your metrics: Accuracy is a bias trap.
    -> Get a more well rounded view off accuracy.
    -> Precision, Recall, F1

3. Set a baseline model, and benchmark against it.

4. Pipelining the Model Training

5. Early Stopping

## NLP
1. Real pre-processing.
2. Bad featurization
 -> Top 1000 are mostly meaningless.
 -> no regularisation
 -> Numbers
 -> Punctuation
 -> bad tokenisation
 -> No stopword filtering

## Representation
1. TFIDF over words
2. Distributed semantics
3. Feature Extraction

### Document features
-> Pull out titles using heuristics.
-> Pull out and Map Acronyms.

### External Knoweledge
-> Ontology Tagging
-> -> Leads to more interesting graph representations

## Modelling
2. Standard Models
3. Graph representations
4. Distributed Semantics
1. Anomaly Detection


## Evalution
1. Accuracy may not be domain specific
2. At a minimum look at the confusion matrix.

## Programming practice
1. looped list look ups are very innefficient, cast as a set as a minimum.
2. all of this should be pipelined really for feature generation



