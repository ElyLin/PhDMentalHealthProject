# PhD Student Mental Health Project
![PhD Mental Health](https://steemitimages.com/DQmSvy1C8BaSreRGnrJcL9qaAEn7cfCgPgcMq6f87Cwg1nQ/Capture.PNG)

## Project Overview
This notebook contains analysis of survey data conducted by Shift Learning on PhD students. The project uses multiple machine learning models such as logistic regression, decision trees and naive bayes to draw insights into what risk factors contribute to mental health problems in PhD students and make recommendations on how to improve.

## Data Overview
The data is aggregated by surveying and interviewing 6,320 current PhD students all over the world in 2019. There are around 60 questions about their personal information and their opinions and concerns about their PhD program. Important questions include age, country of origin, country of study, whether/by whom they were bullied, whether/reason they need to work outside of their program, whether/which kind of discrimination the student faced, their career expectations, worries about the program/their future, etc.

We are modeling on the assumption that if students need help with anxiety and/or depression, they would ask for it.

data source: https://figshare.com/s/74a5ea79d76ad66a8af8?file=18543320

## Methodology
Feature Selection & Engineering: 
- Since the data is a general survey of PhD students and not mental health specific, I only selected the columns that I think would have an impact on their mental health.

Models:
- logistic regression, decision tree, categorical naive bayes, catboost, xgboost.

## Model Result
I'm able to achieve 67% accuracy on both training data and testing data on naive bayes model without overfitting. 

## Author
- Ely Lin

