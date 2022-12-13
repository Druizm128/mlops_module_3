# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is trained to classify persons in two groups. Those that earn more than 50K annually, and those that do not, based on census attribute such as workclass, education, marital-status occupation, relationship, race, sex, native-country, among others.
## Intended Use

The model will be used to predict the income of people, that will serve for marketing purposes as an input of another process.
## Training Data

Data comes from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). The original set contains 32561 rows. The data dates from 1994.
## Evaluation Data

The model was trained with 80% of the data and evaluated with 20%.
## Metrics

* Precision: 79%
* Recall: 60%

## Ethical Considerations

## Caveats and Recommendations

* This model was built as a Proof of Concept.
* To use it today, census data needs to be updated to reflect the current situation.
* The model was trained with census data from 1994.