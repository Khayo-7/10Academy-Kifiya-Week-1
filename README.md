# 10Academy-Kifiya-Week-1

# Final Report: Nova Financial Insights Challenge

## **Objective**

The primary objective of the Nova Financial Insights Challenge is to analyze financial news data to identify potential correlations between news sentiment and stock market movements. Using datasets that include financial news articles, sentiment scores, and historical stock data, the goal is to uncover patterns that could inform investment strategies.
Dataset Analysis

## 1. **Dataset Overview**

The dataset contains 1,407,328 rows of financial news data across five key columns:
  - headline: The title of the news article.
  - url: The source link for each article.
  - publisher: The name of the article's author or publishing entity.
  - date: The publication date and time.
  - stock: The stock ticker related to the news article.
  - 
### Data Stats and Highlights:

  - Unique Publishers: 1,034 distinct publishers; top contributors include Paul Quintaro (228,373 articles) and Lisa Levin (186,979 articles).
  - Stocks Covered: 6,204 unique stock tickers, with MRK being the most frequently mentioned stock (3,333 articles).
    
## 2. **Insights into Publishing Patterns**

### Temporal Distribution:

  - Years: Publications are distributed from 2009–2020, peaking in 2019 (150,380 articles).
  - Months: February 2020 (18,878) and March 2020 (24,995) had notable spikes, coinciding with the early stages of the COVID-19 pandemic.
  - Days and Hours:
    - Publications are primarily concentrated on weekdays, with minimal activity over the weekend (Sunday: 16,466 articles).
    - The majority of articles are timestamped at midnight, reflecting possible delays or data collection behaviors.
      
### Top Publishers:

Publishers with the highest contributions include:

  - Paul Quintaro (228,373 articles, 16% of total publications).
  - The majority use the domain benzinga.com, contributing 7937 verified publications.
    
## 3. **Sentiment Analysis**

### Sentiment Distribution:
  - Neutral: 934,928 articles (66%)
  - Positive: 341,161 articles (24%)
  - Negative: 131,239 articles (9%)
  - **Insight**: Neutral sentiment dominates, but peaks in negative sentiment during 2020 align with global economic downturns.
  - 
### Headline Length:
  - Average: 73 characters
  - Maximum: 512 characters
  - **Insight**: Shorter headlines dominate (~47–64 characters), focusing on clarity in financial reporting.
    
### Keyword and Phrase Usage:
  - Top Keywords: to, for, on, stocks, earnings, update.
  - Common Bigrams: Price Target (45,325 mentions), Earnings Scheduled (32,041 mentions).
  - **Insight**: Keywords and bigrams emphasize earnings updates, stock price targets, and market movements, which can inform key areas of stock market analysis.
    
## 4. **Stock Insights**

### Most Covered Stocks:
  - MRK: 3,333 articles
  - MS: 3,238 articles
  - NVDA: 3,146 articles
  - **Insight**: High-frequency mentions of specific tickers suggest stocks with significant market interest, offering potential candidates for sentiment-impact correlation.

## **Next Steps**
  - Integrate Stock Data: Merge financial news data with the historical stock price dataset to analyze correlations between publication patterns, sentiment, and stock performance (e.g., price changes).
  - Temporal Impact Analysis: Investigate the lag between news publication and stock market movement for different sentiment categories (positive, neutral, negative).
  - Keyword-Sentiment Relation: Assess if certain keywords or bigram phrases (e.g., "Price Target," "Earnings Scheduled") drive stock price movement more significantly.
  - Event-based Correlation: Examine publication spikes and sentiment trends around major events, like the 2020 market volatility, to identify actionable insights.

## **Conclusion**

This interim report highlights a well-rounded understanding of the dataset, including publishing behavior, sentiment distribution, and early stock insights. The foundation is now set for deeper integration with stock price data, enabling correlations that address the challenge's core objective.

