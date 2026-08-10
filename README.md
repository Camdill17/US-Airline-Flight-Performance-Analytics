# U.S. Airline Flight Performance & Delay Analytics — Q1 2026

## Project Overview

This project analyzes U.S. airline flight performance during the first quarter of 2026. The analysis examines flight delays, cancellations, airline performance, airport and route performance, day-of-week patterns, time-of-day patterns, and major causes of delays.

The project also applies machine learning to determine whether scheduled flight characteristics can be used to predict whether a flight will arrive at least 15 minutes late.

## Business Problem

Flight delays and cancellations create operational challenges for airlines, airports, and travelers. Understanding when and where disruptions occur can help identify patterns associated with unreliable flight performance.

This project addresses several questions:

- Which airlines experience the highest delay and cancellation rates?
- Which airports and routes have the greatest delay risk?
- How does flight performance change by day of the week?
- How does scheduled departure time affect delay risk?
- What are the primary causes of flight delays?
- Can machine learning predict whether a flight will arrive 15 or more minutes late?

## Dataset

The analysis uses U.S. airline on-time performance data for January, February, and March 2026.

After combining the monthly datasets, the Q1 dataset contained:

- **1,847,242 total flight records**
- **1,780,195 completed flights used for predictive modeling**
- **362 origin airports**
- **362 destination airports**
- **6,422 unique routes**

A flight was classified as delayed when its arrival delay was **15 minutes or greater**.

## Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- SciPy
- Google Colab
- GitHub

## Exploratory Data Analysis

The project evaluates flight performance across several operational dimensions, including:

- Airline performance
- Airport performance
- Route performance
- Day of week
- Scheduled departure time
- Arrival and departure delays
- Flight cancellation rates
- Delay causes

### Time-of-Day Patterns

Flight performance showed a clear relationship between scheduled departure time and delay risk.

| Time of Day | Delay Rate | On-Time Rate | Avg. Arrival Delay |
|---|---:|---:|---:|
| Early Morning | 13.60% | 86.40% | -0.06 min |
| Morning | 18.72% | 81.28% | 4.10 min |
| Afternoon | 23.92% | 76.08% | 9.49 min |
| Evening | 28.02% | 71.98% | 13.42 min |
| Late Night | 25.36% | 74.64% | 11.55 min |

Early-morning flights had the strongest on-time performance, while evening flights experienced the highest delay rate.

## Day-of-Week Patterns

Sunday had the highest arrival delay rate at **25.49%** and the highest cancellation rate at **6.96%**.

Wednesday performed considerably better, with a delay rate of **17.93%** and cancellation rate of only **1.22%**.

These results suggest that the timing of a flight is an important factor when evaluating operational reliability.

## Route Analysis

Routes with at least 500 flights were evaluated to reduce the influence of routes with very small numbers of observations.

Among the routes analyzed, **CVG → DCA** had the highest arrival delay rate at **46.74%**.

The relationship between route flight volume and delay rate was relatively weak, with a correlation of approximately **0.124**, suggesting that high flight volume alone does not explain route-level delay performance.

## Delay Cause Analysis

The largest shares of attributed delay minutes were:

| Delay Cause | Share of Delay Minutes |
|---|---:|
| Late Aircraft | 38.51% |
| Carrier | 35.25% |
| National Airspace System | 19.07% |
| Weather | 7.00% |
| Security | 0.17% |

Late-aircraft delays represented the largest share of total attributed delay minutes.

Weather delays occurred less frequently but were particularly severe when they occurred, averaging approximately **81 minutes per affected flight**.

## Predictive Modeling

The predictive modeling portion classified completed flights as:

- **0 — On Time:** Arrival delay below 15 minutes
- **1 — Delayed:** Arrival delay of 15 minutes or greater

The modeling dataset contained **1,780,195 completed flights**.

The target distribution was:

- **78.41% on time**
- **21.59% delayed**

A stratified 80/20 train-test split produced:

- **1,424,156 training observations**
- **356,039 testing observations**

Predictors included:

- Month
- Day of week
- Marketing carrier
- Origin airport
- Destination airport
- Scheduled departure hour
- Distance

Categorical variables were encoded during preprocessing, producing **768 predictor columns**.

## Machine Learning Models

Four classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6004 | 0.2966 | **0.6204** | 0.4013 | 0.6482 |
| Decision Tree | **0.6292** | 0.2927 | 0.5063 | 0.3710 | 0.6263 |
| Random Forest | 0.6035 | 0.2980 | 0.6170 | 0.4019 | 0.6523 |
| Gradient Boosting | 0.6272 | **0.3149** | 0.6179 | **0.4172** | **0.6722** |

## Best Model

**Gradient Boosting** provided the strongest overall predictive performance.

Although the Decision Tree achieved slightly higher overall accuracy, Gradient Boosting produced the highest:

- Precision: **0.3149**
- F1 Score: **0.4172**
- ROC-AUC: **0.6722**

It also achieved a recall of **0.6179**, identifying approximately 62% of delayed flights in the test dataset.

The Gradient Boosting model correctly predicted:

- **175,789 on-time flights**
- **47,504 delayed flights**

Its ROC-AUC of **0.6722** indicates moderate ability to distinguish between delayed and on-time flights using information available before departure.

## Feature Importance

Feature importance analysis for the Gradient Boosting model identified scheduled departure hour as the strongest predictor.

| Feature | Importance |
|---|---:|
| Departure Hour | 0.0800 |
| Marketing Carrier | 0.0299 |
| Day of Week | 0.0291 |
| Origin Airport | 0.0254 |
| Month | 0.0232 |
| Destination Airport | 0.0222 |
| Distance | 0.0125 |

The results support the exploratory analysis showing that flight timing plays an important role in delay risk.

## Key Findings

The analysis produced several important findings:

- Flight delays generally became more common later in the day.
- Early-morning flights had the strongest on-time performance.
- Evening flights experienced the highest delay rate.
- Sunday had the highest delay and cancellation rates among the days of the week.
- Late-aircraft and carrier delays accounted for the majority of attributed delay minutes.
- Route flight volume had only a weak relationship with route delay rate.
- Scheduled departure hour was the most influential feature in the Gradient Boosting model.
- Gradient Boosting provided the strongest overall predictive performance of the four models tested.

## Conclusion

The results demonstrate that U.S. airline delays are associated with identifiable temporal, operational, airline, airport, and route characteristics. In particular, scheduled departure time showed a meaningful relationship with flight reliability, with delays generally becoming more common as the day progressed.

The predictive models also demonstrate that scheduled flight information contains useful information for estimating arrival-delay risk. Gradient Boosting achieved the strongest overall balance of classification metrics, although its performance also shows that flight delays cannot be completely explained using scheduled flight characteristics alone.

Future work could incorporate additional information such as weather conditions, airport congestion, aircraft rotations, seasonal patterns, and real-time operational data to improve predictive performance.

## Repository Contents

`US_Airline_Flight_Performance_Analytics_Q1_2026.ipynb` — Complete Python analysis, visualization, preprocessing, and machine learning workflow.

## Author

**Cameron Dillard**

Data Analytics | Python | Machine Learning | Data Visualization
