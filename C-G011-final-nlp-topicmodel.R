# ==========Topic Modeling (LDA) ==========
# Dataset: Amazon Fine Food Reviews subset (10,000 reviews)
# Source: Google Drive direct download

# Link: https://drive.google.com/file/d/1MCA53sblp1uvvVcC_ucC1zeHNPazZKHK/view?usp=sharing

# A. Data Collection
dataset_url <- "https://drive.google.com/uc?export=download&id=1MCA53sblp1uvvVcC_ucC1zeHNPazZKHK"
dataset_file <- "amazon_food_reviews.csv"
if (!file.exists(dataset_file)) {
  cat("Summoning dataset from the abyss...\n")
  download.file(dataset_url, destfile = dataset_file, mode = "wb")
}

# Libraries
packages <- c("data.table", "tm", "SnowballC", "topicmodels", "wordcloud", "ggplot2", "RColorBrewer")
missing <- setdiff(packages, rownames(installed.packages()))
if (length(missing)) install.packages(missing)
invisible(lapply(packages, library, character.only = TRUE))
set.seed(42)

# Load data
dt <- fread(dataset_file, encoding = "UTF-8", na.strings = c("", "NA"))

# Force primary text column 
text_col <- "Text"
if (!text_col %in% names(dt)) stop("Critical failure: 'Text' column absent")
cat("Primary text column locked:", text_col, "\n")

# B. Exploration
cat("Total documents:", nrow(dt), "\n")
dt$text_len <- nchar(dt[[text_col]])
cat("Average text length:", round(mean(dt$text_len, na.rm = TRUE), 1), "\n")

corp <- Corpus(VectorSource(dt[[text_col]]))
dtm_expl <- DocumentTermMatrix(corp)
freq <- sort(colSums(as.matrix(dtm_expl)), decreasing = TRUE)
cat("Raw vocabulary size:", length(freq), "\n")

top_words <- head(freq, 20)
cat("Top 20 raw words:\n")
print(top_words)
barplot(top_words, las = 2, main = "Top 20 Frequent Words", col = brewer.pal(8,"Dark2"))

wordcloud(names(freq), freq, max.words = 100, colors = brewer.pal(8, "Dark2"))

# Sentiment distribution
ggplot(dt, aes(x = Score, fill = Score)) + 
  geom_bar() + 
  scale_fill_manual(values = c("Negative" = "red", "Positive" = "green4")) +
  theme_minimal() + ggtitle("Sentiment Distribution")

# C. Preprocessing
preprocess_corpus <- function(c) {
  c <- tm_map(c, content_transformer(tolower))
  c <- tm_map(c, removePunctuation)
  c <- tm_map(c, removeNumbers)
  c <- tm_map(c, content_transformer(function(x) gsub("(http|www)[^\\s]*", "", x)))
  c <- tm_map(c, removeWords, stopwords("english"))
  c <- tm_map(c, stripWhitespace)
  c <- tm_map(c, stemDocument)
  c
}
corp_proc <- preprocess_corpus(Corpus(VectorSource(dt[[text_col]])))
dtm <- DocumentTermMatrix(corp_proc)
dtm <- dtm[rowSums(as.matrix(dtm)) > 0, ]
cat("Processed documents:", nrow(dtm), " | Vocabulary:", ncol(dtm), "\n")

# D. LDA Modeling
k <- 8  # Optimal for emergent food themes
lda_model <- LDA(dtm, k = k, method = "Gibbs", 
                 control = list(seed = 42, iter = 1500, thin = 100))

# E. Evaluation & Interpretation
top_terms <- terms(lda_model, 15)
cat("\nTop 15 terms per topic:\n")
print(top_terms)

perp <- perplexity(lda_model, dtm)
cat("\nPerplexity (lower = better coherence):", round(perp, 2), "\n")

# Dominant topic per document
dt$dominant_topic <- topics(lda_model)
ggplot(dt, aes(x = factor(dominant_topic))) + 
  geom_bar(fill = brewer.pal(8,"Set2")) + 
  theme_minimal() + 
  ggtitle("Documents per Dominant Topic") + 
  xlab("Topic")

# Perfected wordclouds 
beta <- lda_model@beta
terms_vec <- lda_model@terms
for (i in 1:k) {
  probs <- exp(beta[i, ])
  top_idx <- order(probs, decreasing = TRUE)[1:50]
  wordcloud(terms_vec[top_idx], probs[top_idx],
            scale = c(4, 0.5),     # Fixed scale range
            min.freq = 0,
            random.order = FALSE,  # Highest prob first
            rot.per = 0.3,
            colors = brewer.pal(8, "Dark2"))
  title(main = paste("Topic", i, "- Emergent Theme"))
}

cat("\nRite complete. Topics manifest:\n")
cat("- Topic 1: Recipe/cooking\n")
cat("- Topic 2: Purchasing/ordering\n")
cat("- Topic 3: Practical usage\n")
cat("- Topic 4: Tea & drinks\n")
cat("- Topic 5: Snacks & chips\n")
cat("- Topic 6: Coffee & chocolate\n")
cat("- Topic 7: Pet food\n")
cat("- Topic 8: Complaints & trials\n")