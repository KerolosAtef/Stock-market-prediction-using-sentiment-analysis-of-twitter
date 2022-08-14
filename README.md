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

## Background:
### What's ARIMA(Autoregressive integrated moving average) ?
> - ARIMA is a statistical analysis model that predicts the future of a variable with respect to time or for a better understanding of the time series dataset.
> - ARIMA is a generalization of Auto-Regressive, Moving Average, and integration terms.
> Those terms would be explained briefly in the following section.
> 1.	Auto-Regressive (AR): is a specific type of regression model, which means the current values are correlated with previous values in the time steps. To be more precise it’s partial Auto-correlation.
> <br><center>Y(t) = β1 + Ф1 Y(t-1) + Ф2 Y(t-2) +.. + Фp Y(t-p)</center>
> 	And, the (P) is the lagged order 
> 2.	Moving Average (MA): is analyzing the errors from the lagged observations and how they affect the current observation.
> <br><center>Y(t) = β2 + 𝟂1 𝞷(t-1) + 𝟂2 𝞷(t-2) +.. + 𝟂p 𝞷(t-p)</center>
> The 𝞷 terms are the errors observed, the 𝟂 is the weight of this error, and 𝟂 are calculated using a statistical correlation test.
> And, (q) represents the size of the moving average that has a significant impact on the current observation.<br>
> 3.	Integrated (I): the previous models can handle only the stationary time series dataset, which has a constant mean (μ), and variance (σ) without having seasonality. By taking the difference between consecutive timesteps, this transform will eliminate the trend and keep the mean constant. So, they defined (d) the order of differencing which means how many times we would apply the differencing process.
> <br><br>Till now, ARIMA can handle Non-stationary data with trends, but can’t handle the seasonality component, So, SARIMA (Seasonal Autoregressive Integrated Moving Average) was introduced as an extension adding more parameters to handle the seasonality.
> <br><center>![image](https://drive.google.com/uc?export=view&id=1K2-VRGdQZtigaXXw1AZXWMJJoUCJhanc)</center>
> <br >**SARIMA Parameters:**
> - p: non-seasonal or trend autoregressive (AR) order
> - d: non-seasonal or trend differencing<br>
> - q: non-seasonal or trend moving average (MA) order
> - P: seasonal AR order
> - D: seasonal differencing
> - Q: seasonal MA order
> - S: length of repeating seasonal pattern
> 
> To identify the values of AR and MA parameters we use ACF and PACF plots. Or by using grid search to go over all the combinations of parameters and chose the one that achieves the least loss according to a defined loss function such as Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) as both are penalized-likelihood information criteria.
And to identify the integrated parameters, they use a statistical test called the “Augmented Dickey-Fuller test”
> - Note: Due to the headache of identifying the model and its parameters, the python community has developed an automated library called “Auto Arima” that chooses the best model suitable for your dataset and identifies the best parameters to achieve the lowest loss.


### Why CNN at time series?
> **Convolutional Neural Networks (CNN):** <br> 
> The 1-d convolutional neural networks are used intensively in time series forecasting for a fixed window length, CNN has the ability to remove the noise and unrelated patterns along the time series data. Also, the convolutional and pooling layers both perform smoothing to the data instead of the traditional weighted average with require manual tuning.
> ![image](https://drive.google.com/uc?export=view&id=1uX57pjUzbgduD-9nRgpuquUh7uOKGTXC) <br>
> Convolution learn from data through its filter’s weights update. CNN layers depend on the learning of the filters, so these filters after training will be able to detect the features.
> ![image](https://drive.google.com/uc?export=view&id=1gDeLNYGX22hryuIUNbidhl5dadUUXXD0) <br>
> The pooling layer mainly smooths out and reduces the local noise by averaging the time series values.
> ![image](https://drive.google.com/uc?export=view&id=1SID4bHM2Xuu2q4KPqNxYbRBnnYPjKQI9) <br>
> The data change after each layer:
> ![image](https://drive.google.com/uc?export=view&id=1ySLDnYLasBa_rPSALGjutuRkmU6A2ZJl) <br>
> Figure 1: the change due to the CNN layer
> ![image](https://drive.google.com/uc?export=view&id=1t57Teb0kQK5lHSm_3ZXugbvjc4Ziw9Fq) <br>
> Figure 2: the change due to the pooling layer


### Why LSTM at time series?

> LSTM (Long Short-Term Memory) networks have the ability to learn long-term dependencies, especially, in a sequence prediction problem as time series, LSTM has outstanding performance on a large variety of data.
> ![image](https://drive.google.com/uc?export=view&id=1rX81D0b-WLr3MhiebNlPS9NIASQEHps1) <br>
> **Important Note:** The Bidirectional LSTM can learn in both direction forward and backward sequences, also the Bidirectional has complete information about all the points in the data.


## Methodology
### The Project Structure
![image](https://drive.google.com/uc?export=view&id=1dAWrpyQ9wc8ehxKFVR8NB2q45Zh468g0)
As Shown at the above figure, our project has nine stages:
- Stock Data Scraping
- Twitter Data Scraping
- Twitter Data Preprocessing and Sentiment analysis
- Modeling Using Two approaches
  - ARIMA as Baseline
  - Deep learning architecture using 1D-CNN and BiCudaLSTM layers as our new approach
- Compare and evaluate our approach vs ARIMA
- Evaluate our approach against new data in real-time

### Dataset Collection
- The Stock Data: The research team downloaded the stock data of Netflix in Spreadsheet from Yahoo Finance website from Jan 1st, 2018 to July 7th, 2022 
![image](https://drive.google.com/uc?export=view&id=1LNpK07hJg-gZNZVBaRy5In4BT8YNA4KR)

- The Twitter Data: The research team scrapped within the same period using the Snscrape tool. 
<br>**Note:** Twitter API could not be used in this project as it was limited to only one week
