![Cinema](img/title_image.png)

# Data Analysis of AirBnB reviews

This document outlines the analysis performed on a DataFrame with reviews on Airbnb listings. It contains the following columns: `listing_id`, `id`, `date`, `reviewer_id`, `reviewer_name`, `comments`. The analyses include visualizations to help understand the data better.

## Motivation

The goal of this report is to create an understanding for the the reviews of Airbnb listings.

## View the Analysis

[View Analysis Report](analysis.md)

## Libraries used

* Pandas
* Matplotlib
* Seaborn
* Wordcloud

## Files

* analysis_files/
  * exported pictures of the diagrams
* data/
  * reviews.csv: The CSV-file containing the data
* img/
  * images
* analysis.jpynb: The Jupyter notebook containing the code
* anaysis.md: The markdown file exported from the jupyter notebook
* reviews.csv:


## Data Analysis

### 1. Distribution of Reviews by Listing

**Objective:** Visualize the number of reviews each listing has received.

**Diagram:**

![png](analysis_files/analysis_9_0.png)

**Findings:** This gives a good insight how many reviews there are per listing.

### 2. Distribution of Reviews over time

**Objective:** Visualize the number of reviews over time

**Diagram:**

![png](analysis_files/analysis_11_0.png)

**Finding:** The number of reviews has increased drastically over the last years and there is a peak of reviews towards the end of each year.

### 3. Review Counts by Reviewer

**Objective:** Determine the distribution of the number of reviews written by each of the top ten reviewers.

**Diagram:**

![png](analysis_files/analysis_13_0.png)

**Finding:** The top reviewer has more than double the reviews of the number ten reviewer.

### 4. Temporal Distribution of Reviews

**Objective:** Analyze the distribution of reviews over time.

**Diagram:**

![png](analysis_files/analysis_15_0.png)

**Finding:** The by far most listings have very few reviews, only a few listings have lots of reviews.

### 5. Common Words in Comments

**Objective:** Identify the most frequently occurring words in the comments.

**Diagram:**

![png](analysis_files/analysis_17_0.png)

**Findings:** The most often used words don't have anything in common except that they are related to travel, which is kind of expected. The most often visited city in this dataset seems to be Seattle.

## View the Analysis

[View Analysis Report](analysis.md)

## Conclusion

These analyses provide insights into the distribution and characteristics of the reviews dataset. Use the diagrams to gain a deeper understanding of review patterns and trends.
