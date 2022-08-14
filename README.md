# Enhance the Stock Market Prediction Using Twitter Sentiment analysis

## Contents:
* [Problem Formulation](#Problem-Formulation)
* [Background](#Background)
  * [What's ARIMA(Autoregressive integrated moving average) ?](#What's-ARIMA(Autoregressive-integrated-moving-average)-?)
  * [Why CNN at time series?](#Why-CNN-at-time-series?)
  * [Why LSTM at time series?](#Why-LSTM-at-time-series?)
* [Methodology](#Methodology)
  * [The Project Structure](#The-Project-Structure)
  * [Dataset Collection](#Dataset-Collection)
  * [Data Preprocessing](#Data-Preprocessing)
  * [Sentiment Analysis](#Sentiment-Analysis)
  * [Data preparation](#Data-preparation)
    * [ARIMA Preparation](#ARIMA-Preparation)
    * [CNN-LSTM Preparation](#CNN-LSTM-Preparation)

  * [Modeling](#Modeling)
    * [CNN-LSTM Based approach](#CNN-LSTM-Based-approach)
    * [ARIMA (Autoregressive integrated moving average) approach](#ARIMA-(Autoregressive-integrated-moving-average)-approach)
* [Results](#Install)
* [Conclusion](#Install)

* [Usage](#Usage)
  * [Scraping tweets ](#Scraping-tweets )
  * [Applying the Sentiment analysis on the tweets](#Applying-the-Sentiment-analysis-on-the-tweets)
* [References and Related Work](#References-and-Related-Work)
* [Team Members](#Team-Members)

## Problem Formulation:
The stock market is a focus for investors to maximize their potential
profits and consequently, the interest shown from the technical
and financial sides in stock market prediction is always on the rise.

However, stock market prediction is a problem known for its challenging
nature due to its dependency on diverse factors that affect 
the market, these factors are unpredictable and cannot be taken into
consideration such as political variables, and social media effects
such as twitter on the stock market.
