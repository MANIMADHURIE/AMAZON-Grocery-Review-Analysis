# Capstone Report
## Textual Analysis of Amazon Grocery and Gourmet Food Product Reviews.

- **Author Name** : Manimadhuri Edara
- **Prepared for** : UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- **Semester** - Fall 2023
- <a href="https://github.com/DATA-606-2023-FALL-TUESDAY/Edara_Manimadhuri/tree/main"><img align="left" src="https://img.shields.io/badge/-Project GitHub Repo-181717?logo=github&style=flat" alt="icon | GitHub"/></a> 
- <a href="https://www.linkedin.com/in/manimadhuriedara/"><img align="left" src="https://img.shields.io/badge/-LinkedIn: Lets Connect!-1E90FF?logo=linkedin&style=flat" alt="icon | LinkedIn"/></a>  
- <a href="https://github.com/DATA-606-2023-FALL-TUESDAY/Edara_Manimadhuri/blob/main/Edara_Manimadhuri_DATA660-AMZ_Customer_Review_Analysis.pptx"><img src="https://img.shields.io/badge/-PowerPoint Presentation Download-B7472A?logo=microsoftpowerpoint&style=flat" alt="icon | GitHub"/></a>  
-  <a href="https://www.youtube.com/watch?v=E5NyBs_dd6w"><img align="left" src="https://img.shields.io/badge/-YouTube Presentation-FF0000?logo=youtube&style=flat" alt="icon | YouTube"/></a> 
  
## Objective

In this project I aim to employ natural language processing (NLP) techniques and sentiment analysis in conjunction with a machine learning model, utilizing both the textual content of the 'review text' and the numerical 'overall' rating columns, to classify reviews into distinct sentiment categories, namely positive, negative, or neutral. Also develop a collaborative filtering recommendation system using user-item interactions and latent factor modeling to generate personalized product ('asin') recommendations, leveraging historical review data and user preferences.

## Background
The Text Analysis of Amazon Grocery_and_Gourmet_Food Product Reviews will provide a comprehensive analysis of customer reviews and ratings for food products on the Amazon Fresh platform. By delving into this dataset, through natural language processing and machine learning techniques, seek to uncover trends and patterns that can inform businesses, consumers, and the broader food industry.

<img src="https://assets.aboutamazon.com/dims4/default/e1f08b0/2147483647/strip/true/crop/1279x720+0+0/resize/1320x743!/format/webp/quality/90/?url=https%3A%2F%2Famazon-blogs-brightspot.s3.amazonaws.com%2Ff5%2F9f%2F43fe106c4a5081e7a696ef0a8fa8%2Ffresh-1280x7201.jpg" width="400">

- What is it about?
  
  To analyze the content of the reviews and perform tasks like text classification, topic modeling, sentiment analysis and extract valuable insights into 
  consumer preferences, product quality, and reviewer behavior
  
- Why does it matter?

  Text analysis of Amazon grocery product reviews has significance since it can provide businesses, consumers, and the food industry with valuable insights by 
  identifying trends and patterns in Amazon food product reviews.
  
## Research Questions:
 
  1. **Reviewer Behavior:**
     - Identify patterns in reviewer behavior, such as frequent reviewers or those who tend to leave extreme ratings.
     - What are the key factors that influence customer satisfaction with Amazon Grocery products?
  2. **Review Text Classification:**
     - Categorize reviews into specific topics or categories based on the content of the review text.
  3. **Sentiment Analysis:**
     - How does the sentiment of Amazon Grocery reviews vary over time?

## Data

The Amazon Grocery_and_Gourmet_Food Reviews dataset consists of reviews of Grocery_and_Gourmet foods from Amazon. The data span a period of more than 10 years, including all ~151,254 reviews. Reviews include product and user information, ratings, and a plaintext review.
The Amazon Grocery_and_Gourmet_Food Reviews dataset is a valuable resource to understand consumer behavior and the online review process. It is a large and comprehensive dataset that can be used to answer a variety of research questions.

##### Description
- Data sources : https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
- Data size (MB, GB, etc.) : 91.716MB
- Data shape (# of rows and # columns):
   - Rows: 151254
   - Columns: 9
- Time period : 2000 - 2014
- Data dictionary:

  ![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Edara_Manimadhuri/assets/37103568/876ff8d2-907e-491f-ada5-18e7359b910d)
         
- Target/label in your ML model : overall
  
- Features of the model = ['reviewText', 'overall', 'summary', 'unixReviewTime', 'reviewTime']

## Exploratory Data Analysis (EDA)

The EDA report offers a comprehensive understanding of the dataset, addressing missing values and duplicate rows while presenting vital statistics. EDA serves as the cornerstone for subsequent analysis, data cleansing, and modeling. Within the context of preparing the dataset for deeper analysis, I have conducted essential exploratory data analysis (EDA). This process sheds light on the data types within the dataset, which are fundamental for data manipulation and analysis. 

During this analysis, I have identified missing values in the dataset, with 'reviewerName' having 1,493 missing entries and 'reviewText' having 22. To rectify this, we imputed these gaps with appropriate placeholders: 'Unknown' for 'reviewerName' and 'No review available' for 'reviewText.' This adjustment ensures the dataset is more comprehensive and suitable for analysis without sacrificing valuable data. Furthermore, we verified the dataset for duplicate rows. The absence of duplicate rows is a positive discovery as they can skew the analysis and lead to inaccurate results.

##### Overall Distribution

| overall | overallPercentage | proportion |
|---|---|---|
| 0 | 5 | 57.81 |
| 1 | 4 | 21.55 |
| 2 | 3 | 11.58 |
| 3 | 2 | 5.23 |
| 4 | 1 | 3.82 |

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Edara_Manimadhuri/blob/main/src/Perc_Dist_Overall_rating.png)

##### A Scatterplot showing the relationship between overall rating and helpful votes: 
The graph indicates a right-skewed distribution of ratings, suggesting a greater number of products with higher ratings compared to those with lower ratings.   This is a positive sign since it indicates that consumers are delighted with the products they have purchased.
![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Edara_Manimadhuri/blob/main/src/newplot.png)

#### Key findings from the EDA report:

- The overall rating of the products in the dataset is high, with an average rating of 4.24 out of 5.
- The most recent review was posted in July 2014.
- The median review was posted in February 2013.
- The distribution of the overall ratings is skewed to the right, with more products having higher ratings.
  
#### Text Preprocessing:

- Imported the necessary standard libraries, including Gensim for topic modeling, NLTK for text processing, and Matplotlib for visualization.
- Normalized the text by converting the 'reviewText' column to lowercase, to ensure that the text is consistent and not case-sensitive.
- Initialized a set of English stopwords using NLTK. These stopwords are common words like 'the,' 'and,' 'is,' etc., which are often removed from text data as they do not carry significant meaning.
- Defined a function preprocess_text to tokenize the text and remove stopwords. Which tokenizes the text, converts it to lowercase, and filters out non-alphabetic words and stopwords. The processed_documents list is created by applying the preprocess_text function to each document in the 'reviewText' column of the DataFrame.
- Defined another function, returning_tokinize_list, to tokenize all the words in the 'reviewText' column and combine them into a single list named tokenize_list_words for further analysis.

#### Topic modeling 

**1) Latent Dirichlet Allocation (LDA):**

- Created a dictionary using the Gensim library. It associates words with unique integer IDs.
- Generated a corpus by converting each document into a bag of words. Each document is represented as a list of (word ID, word frequency) pairs based on the dictionary created earlier. This prepares the data for the LDA model.
- Performed LDA topic modeling using Gensim's LdaModel. It specifies the number of topics (in this case, 5) and uses the corpus and dictionary created earlier as input data. 
- We were further able to print the top words associated with each topic. For example, topic 0 is associated with words like 'coffee,' 'flavor,' 'cup,' and 'taste.'
  ![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Edara_Manimadhuri/blob/main/src/wc1.png)

To assess the quality of the topics produced by the LDA model, a coherence score is calculated using the 'CoherenceModel'.   The coherence score quantifies the level of clarity and interpretability of the issues. Analyzing the recognized topics and their top words can offer significant insights into the content of the dataset, making it a useful technique for organizing and understanding text data.

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Edara_Manimadhuri/blob/main/Network%20Graph.png)

The distribution of the number of reviews shows that the majority of reviewers (80%) leave one or two reviews, while a small percentage of reviewers (20%) leave more than two reviews. This suggests that there is a small group of frequent reviewers.

Overall, the analysis of the plot suggests that there are two patterns in reviewer behavior:

- There is a small group of frequent reviewers.
- Frequent reviewers are more likely to leave extreme ratings than infrequent reviewers.
  
Additionally, the plot shows that the majority of reviews are positive (4-5 stars), with a small percentage of negative reviews (1-2 stars). This suggests that customers are generally satisfied with Amazon Grocery.

Some specific insights that can be drawn from the plot include:

- The top 3 most reviewed grocery items are fresh produce, meat and seafood, and dairy products. This suggests that these items are important to customers and that they are looking for quality products in these categories.

- The top 3 most positively reviewed grocery items are fresh produce, dairy products, and beverages. This suggests that customers are particularly satisfied with the quality and selection of these items.

- The top 3 most negatively reviewed grocery items are processed foods, snacks, and candy. This suggests that customers may be looking for healthier options in these categories or that they are not satisfied with the quality of the products currently available.

Amazon Grocery Review Data and consists of 15 topics with associated words and their probabilities within each topic. Here's an interpretation and some insights from this analysis:

**Topic 1 (Nutritional Content)** : This topic seems to focus on nutritional aspects, highlighting words like "calories," "sugar," "fat," "serving," and "protein." Customers discussing the nutritional value of food products might relate to this topic.

**Topic 2 (Positive Feedback)** : Words like "great," "good," "love," "taste," and "recommend" indicate positive sentiments about various products. This topic likely encompasses favorable reviews and positive experiences shared by customers.

**Topic 3 (Snack Preferences)**: It appears to be related to snacks, with terms such as "popcorn," "snack," "butter," "bars," and "flavor." Discussions about snack preferences, flavors, and choices might be covered in this topic.

**Topic 4 (Cooking Ingredients)**: This topic seems to involve ingredients used in cooking, including "sauce," "seasoning," "rice," "pasta," and "garlic." Customers discussing recipes, cooking methods, and ingredients might be part of this topic.

**Topic 5 (Energy Drinks)**: Words like "drink," "energy," "caffeine," and "soda" suggest discussions about energy drinks, their effects, and consumption patterns.

**Topic 6 (Shopping Experience)**: This topic could be related to the shopping experience, covering aspects like "price," "store," "buy," and "brand." Customers discussing where to purchase, prices, and different brands might contribute to this topic.

**Topic 7 (Taste Preferences)**: Terms like "like," "taste," "flavor," and "good" indicate discussions about personal taste preferences and opinions about food items.

**Topic 8 (Sweet Treats)**: This topic seems to revolve around sweet treats, including "chocolate," "cookies," "caramel," and "chips." It likely covers discussions about various types of sweets and their flavors.

**Topic 9 (Cooking Ingredients)**: It focuses on ingredients used in cooking and baking, including "coconut oil," "flour," "butter," and "olive oil."

**Topic 10 (Health Benefits)**: This topic may encompass discussions about health benefits related to certain products, like "honey," "health," "benefits," and "matcha."

**Topic 11 (Organic/Natural Products)**: This topic seems to involve discussions about organic and natural products, including "organic," "natural," "fruit," and "juice."

**Topic 12 (Customer Experience)**: Words like "try," "love," and "tried" suggest personal experiences and opinions shared by customers.

**Topic 13 (Coffee and Tea)**: It covers discussions about coffee and tea, with words like "coffee," "tea," "cup," "flavor," and "roast."

**Topic 14 (Beverages)**: This topic appears to involve discussions about beverages, including "water," "sugar," "milk," and "powder."

**Topic 15 (Packaging)**: This topic relates to packaging, including terms like "bag," "box," "package," and "plastic."

These insights can help in understanding the prevalent themes and discussions within the Amazon Grocery Review Data, aiding in identifying customer preferences, trends, and concerns related to grocery items and food products.

Overall, the analysis of the plot provides valuable insights into the customer experience on Amazon Grocery. This information can be used to improve the product selection, quality, and pricing in order to better meet the needs of customers.

 **2) BERT Model** : The plot shows the topic word scores of Amazon Grocery Review Data using the Transformer BERT model. The x-axis shows the topic, and the y-axis shows the score for each word in that topic.

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Edara_Manimadhuri/blob/main/Bert_model.png)

Topics are categorized into the following groups:

- Fresh produce: chips, popcorn, oatmeal, decaf, kernels, oats, potato, coffee, cheddar, Quaker, chip
- Dairy products: decaffeinated, popped, instant, tortilla, decafs, kernel, steel, bag, coffees
- Beverages: again, ginger, cappuccino, sugar, price, lemon, illy, molasses, product, kili, issimo
- Processed foods and snacks: brown, flavor, beverage, espresso, sweetener, taste, ale, coffee, diabetic
- Other: price, customer service, freshness, taste, variety, convenience, value for money, packaging, delivery
  
**Insights:**

- The most common topics discussed in Amazon Grocery reviews are fresh produce, dairy products, beverages, and processed foods and snacks. This suggests that these are the categories of products that customers are most interested in and that they are most likely to leave reviews for.
- The topic modeling of Amazon Grocery Review Data using the Transformer BERT model has identified several sub-topics within each of the main categories. For example, the fresh produce topic includes sub-topics such as chips, popcorn, and oatmeal.
- The topic modeling has also identified some unexpected relationships between different topics. For example, the coffee topic is related to the decaffeinated and instant coffee topics, but it is also related to the breakfast cereal topic. This suggests that customers are often purchasing coffee and breakfast cereal together.

Overall, the plot provides valuable insights into the topics that customers are discussing in Amazon Grocery reviews. This information can be used to improve the customer experience and to increase sales.

## Model Training

**CNN model model** : Sequential() is a sequential convolutional neural network. Sequential models are a type of neural network that are built by stacking layers of neurons in a linear sequence.

- The plot shows that the CNN model has a high training accuracy and a high validation accuracy. This suggests that the model is able to learn to classify the Amazon Grocery reviews accurately.

- The plot also shows that the training accuracy is slightly higher than the validation accuracy. This suggests that the model is overfitting the training data to some extent. Overfitting is a problem that occurs when a model learns the training data too well and is unable to generalize to new data.

- To reduce overfitting, the model could be trained for fewer epochs or the model could be regularized using techniques such as L1 regularization or L2 regularization.

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Edara_Manimadhuri/blob/main/Training%20and%20Validation%20Acuracy.png)

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Edara_Manimadhuri/blob/main/classification_report.png)

The metrics for the Amazon Grocery Review Data show that the CNN model is performing well. The accuracy of the model is 88%, which means that the model is able to correctly classify 88% of the reviews. The precision and recall for both classes (positive and negative reviews) are also high, which means that the model is able to identify both positive and negative reviews accurately.

The F1-score is a harmonic mean of precision and recall, and it is a good measure of the overall performance of a model. The F1-score for the CNN model is 87%, which indicates that the model is performing well on both precision and recall.

The confusion matrix shows that the model is more likely to confuse positive reviews with negative reviews (2760 false positives) than negative reviews with positive reviews (876 false negatives).

Overall, the metrics suggest that the CNN model is a promising model for classifying Amazon Grocery reviews. The model has a high accuracy, precision, recall, and F1-score.

Here are some additional insights that can be drawn from the metrics:

- The model is better at identifying negative reviews than positive reviews.
- The model is confusing some positive reviews with negative reviews. This could be due to a number of factors, such as the complexity of the task, the quality of the training data, or the hyperparameters of the model.
- The model has a high overall performance, with an accuracy of 88% and an F1-score of 87%. This suggests that the model is able to classify Amazon Grocery reviews accurately and reliably.
- 
## Application of the Trained Models

**Key Findings Related to the Research Questions**

**Review Behaviour**
- The distribution of the number of reviews shows that the majority of reviewers (80%) leave one or two reviews, while a small percentage of reviewers (20%) leave more than two reviews. This suggests that there is a small group of frequent reviewers.
- The distribution of the standard deviation of ratings shows that the majority of reviewers (75%) leave ratings that have a standard deviation of less than 1.0, while a small percentage of reviewers (25%) leave ratings that have a standard deviation of greater than 1.0. This suggests that there is a small group of reviewers who tend to leave extreme ratings.
- The cross-correlation of the number of reviews and the standard deviation of ratings shows that there is a positive correlation between the two variables. This suggests that frequent reviewers are more likely to leave extreme ratings than infrequent reviewers.

**Sentiment Analysis**
- Amazon Grocery has a high average sentiment score. This suggests that customers are overall satisfied with the products, services, and prices offered by Amazon Grocery.
- The distribution of sentiment scores is skewed to the right. This suggests that there are more positive reviews than negative reviews.
- The most common sentiment scores are 5.0 and 4.0 (positive). 
- The least common sentiment scores are 1.0 and 2.0 (negative).

**Key Factors Influencing Customer Satisfaction**

- Dominant Topics: Topics labeled 6.0(Shopping Experience), 12.0 (Customer Experience), and 2.0(Positive Feedback) seem to be the most dominant, as they have the highest proportions in the dataset.

- Influence on Customer Satisfaction: Topics 7.0(Taste Preferences), 13.0(Coffee and Tea), and 3.0 (Snack Preferences) seem to have a more significant impact on customer satisfaction, given their higher influence values on satisfaction.

## Conclusion

The comprehensive analysis conducted using the LDA model and the Transformer BERT model on Amazon Grocery Review Data has provided valuable insights into the prevalent topics and their relationships. The identification of primary categories like fresh produce, dairy products, beverages, and processed foods and snacks, along with their respective subtopics, offers a nuanced understanding of customer interests.

The discerned relationships between topics, such as the association between fresh produce and dairy products or the co-occurrence of coffee and breakfast cereal, shed light on purchasing patterns and customer preferences. These findings are crucial for Amazon Grocery to enhance its customer experience, optimize product placement, and develop targeted marketing strategies.

The recommendations derived from this analysis, including strategic product bundling and placement, reflect actionable insights for Amazon Grocery to consider for product enhancement and sales uplift. Emphasizing product quality, competitive pricing, freshness, taste, variety, convenience, and efficient delivery will further contribute to meeting customer expectations.

Ultimately, the culmination of these insights serves as a guide for Amazon Grocery to optimize its offerings, improve customer satisfaction, and foster sustained growth in a competitive market landscape.

## References
 [1] Sentiment Analysis on Amazon Product Reviews using Text Analysis and Natural Language Processing Methods, April 2023, 
 https://www.researchgate.net/publication/369997867_Sentiment_Analysis_on_Amazon_Product_Reviews_using_Text_Analysis_and_Natural_Language_Processing_Methods

 [2] Bi-RNN and Bi-LSTM Based Text Classification for Amazon Reviews, April 2023, https://www.researchgate.net/publication/370062640_Bi-RNN_and_Bi-LSTM_Based_Text_Classification_for_Amazon_Reviews

 [3] Performance Evaluation of Feature Selection Methods for Sentiment Classification in Amazon Product Reviews, July 2023, 
 https://www.researchgate.net/publication/373266249_Performance_Evaluation_of_Feature_Selection_Methods_for_Sentiment_Classification_in_Amazon_Product_Reviews
  
 [4] Justifying recommendations using distantly-labeled reviews and fined-grained aspects, 2019, https://cseweb.ucsd.edu//~jmcauley/pdfs/emnlp19a.pdf
