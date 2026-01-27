# ========== NLP Project 1: Text Classification (Sentiment Analysis) ==========
# Dataset: Amazon Fine Food Reviews subset (10,000 reviews)
# Source: Google Drive - https://drive.google.com/file/d/1MCA53sblp1uvvVcC_ucC1zeHNPazZKHK/view?usp=sharing
# Domain: Food product reviews (Positive/Negative sentiment in 'Score')

# Install & load required libraries
packages <- c("dplyr", "tidytext", "ggplot2", "text2vec", "wordcloud", "SnowballC", "caret", "e1071", "tm", "RColorBrewer")
missing <- setdiff(packages, rownames(installed.packages()))
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
invisible(lapply(packages, library, character.only = TRUE))

# A. Data Collection
dataset_url <- "https://drive.google.com/uc?export=download&id=1MCA53sblp1uvvVcC_ucC1zeHNPazZKHK"
if (!file.exists("dataset.csv")) download.file(dataset_url, "dataset.csv", mode = "wb")

# Load the dataset
library(data.table)
dt <- fread("dataset.csv", encoding = "UTF-8", na.strings = c("", "NA"))

# Auto-detect text column
cn <- tolower(names(dt))
text_col_idx <- which(cn %in% c("text", "reviewtext", "content"))
text_col <- names(dt)[text_col_idx[1]]
dt$sentiment <- factor(dt$Score)  # Positive/Negative

# B. Text Understanding & Exploration
cat("Total documents:", nrow(dt), "\n")
dt$text_len <- nchar(dt[[text_col]])
cat("Average text length:", mean(dt$text_len), "\n")

library(wordcloud)
library(ggplot2)

# Remove missing text data
dt <- dt[!is.na(dt[[text_col]]) & nchar(dt[[text_col]]) > 0, ]

# Create Corpus and Document-Term Matrix (DTM)
corp <- Corpus(VectorSource(dt[[text_col]]))

# Preprocess Corpus (remove punctuation, numbers, URLs, stopwords, etc.)
corp <- tm_map(corp, content_transformer(tolower))
corp <- tm_map(corp, removePunctuation)
corp <- tm_map(corp, removeNumbers)
corp <- tm_map(corp, content_transformer(function(x) gsub("http\\S+|www.\\S+", "", x)))  # Remove URLs
corp <- tm_map(corp, removeWords, stopwords("english"))
corp <- tm_map(corp, stripWhitespace)

# Check corpus length before and after preprocessing
cat("Corpus before preprocessing:", length(corp), "\n")
cat("Corpus after preprocessing:", length(corp), "\n")

# Create Document-Term Matrix (DTM) from Corpus
dtm <- DocumentTermMatrix(corp)

# Create TF-IDF representation
tfidf <- weightTfIdf(dtm, normalize = TRUE)

# Text Exploration - Most frequent words & Word Cloud
freq <- sort(colSums(as.matrix(dtm)), decreasing = TRUE)
cat("Vocabulary size (raw):", length(freq), "\n")

top_words <- head(freq, 20)
barplot(top_words, las = 2, main = "Top 20 Frequent Words")
wordcloud(names(freq), freq, max.words = 100, colors = brewer.pal(8, "Dark2"))

# Frequent n-grams (bigrams)
# Use text2vec for n-gram extraction (bigrams)
library(text2vec)
finder <- function(x) unlist(lapply(ngrams(words(x), 2), paste, collapse = " "), recursive = FALSE)

# Create a bigram tokenizer
BigramTokenizer <- function(x) {
  ngrams <- text2vec::word_tokenizer(x)
  bigrams <- unlist(lapply(ngrams, function(words) {
    if (length(words) >= 2) paste(words[1], words[2], sep = " ") else NULL
  }))
  return(bigrams)
}

# Apply DocumentTermMatrix using Bigram Tokenizer
dtm_bigram <- DocumentTermMatrix(corp, control = list(tokenize = BigramTokenizer))

# Frequency of bigrams
freq_bigram <- sort(colSums(as.matrix(dtm_bigram)), decreasing = TRUE)
top_bigrams <- head(freq_bigram, 20)
barplot(top_bigrams, las = 2, main = "Top 20 Bigrams")

ggplot(dt, aes(x = sentiment)) + geom_bar() + theme_minimal() + ggtitle("Sentiment Distribution")

# C. Text Preprocessing
library(SnowballC)

# Preprocessing function for corpus
preprocess_corpus <- function(c) {
  c <- tm_map(c, content_transformer(tolower))
  c <- tm_map(c, removePunctuation)
  c <- tm_map(c, removeNumbers)
  c <- tm_map(c, content_transformer(function(x) gsub("http\\S+|www.\\S+", "", x)))  # Remove URLs
  c <- tm_map(c, removeWords, stopwords("english"))
  c <- tm_map(c, stripWhitespace)
  c <- tm_map(c, stemDocument)
  c
}
corp_proc <- preprocess_corpus(corp)

# Create Document-Term Matrix (DTM) and TF-IDF representation
dtm <- DocumentTermMatrix(corp_proc)
tfidf <- weightTfIdf(dtm, normalize = TRUE)

# D. Modeling (Naive Bayes for binary classification)
library(caret)
library(e1071)

train_idx <- createDataPartition(dt$sentiment, p = 0.8, list = FALSE)
train_tfidf <- tfidf[train_idx, ]
test_tfidf <- tfidf[-train_idx, ]
train_label <- dt$sentiment[train_idx]
test_label <- dt$sentiment[-train_idx]

# Naive Bayes model
nb_model <- naiveBayes(as.matrix(train_tfidf), train_label)

# E. Evaluation & Interpretation
pred <- predict(nb_model, as.matrix(test_tfidf))
cm <- confusionMatrix(pred, test_label)
print(cm$table)

# Accuracy
cat("Accuracy:", cm$overall["Accuracy"], "\n")

# Manually extract confusion matrix values
TP_negative <- cm$table["Negative", "Negative"]  # True positives for Negative class
FP_negative <- cm$table["Positive", "Negative"]  # False positives for Negative class
FN_negative <- cm$table["Negative", "Positive"]  # False negatives for Negative class
TN_negative <- cm$table["Positive", "Positive"]  # True negatives for Negative class

TP_positive <- cm$table["Positive", "Positive"]  # True positives for Positive class
FP_positive <- cm$table["Negative", "Positive"]  # False positives for Positive class
FN_positive <- cm$table["Positive", "Negative"]  # False negatives for Positive class
TN_positive <- cm$table["Negative", "Negative"]  # True negatives for Positive class

# Calculate Precision, Recall, F1-Score for Negative class
precision_negative <- TP_negative / (TP_negative + FP_negative)
recall_negative <- TP_negative / (TP_negative + FN_negative)
f1_negative <- 2 * (precision_negative * recall_negative) / (precision_negative + recall_negative)

# Calculate Precision, Recall, F1-Score for Positive class
precision_positive <- TP_positive / (TP_positive + FP_positive)
recall_positive <- TP_positive / (TP_positive + FN_positive)
f1_positive <- 2 * (precision_positive * recall_positive) / (precision_positive + recall_positive)

# Print the F1-scores for each class
cat("F1-Score (Negative):", f1_negative, "\n")
cat("F1-Score (Positive):", f1_positive, "\n")

# Calculate the average F1-score
avg_f1_score <- mean(c(f1_negative, f1_positive))
cat("F1-Score (avg):", avg_f1_score, "\n")
