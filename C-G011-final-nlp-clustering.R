# ========== Clustering  ==========
# Dataset: Amazon Fine Food Reviews subset (10,000 reviews)
# Source: Google Drive - https://drive.google.com/file/d/1MCA53sblp1uvvVcC_ucC1zeHNPazZKHK/view?usp=sharing

# ========== A. Data Collection ==========
dataset_url <- "https://drive.google.com/uc?export=download&id=1MCA53sblp1uvvVcC_ucC1zeHNPazZKHK"
dataset_file <- "amazon_food_reviews.csv"
if (!file.exists(dataset_file)) {
  cat("Summoning dataset from the abyss...\n")
  download.file(dataset_url, destfile = dataset_file, mode = "wb")
}

# ========== Libraries ==========
packages <- c("data.table","text2vec","stopwords","Matrix","ggplot2",
              "dbscan","proxy","irlba","cluster","RColorBrewer")
missing <- setdiff(packages, rownames(installed.packages()))
if (length(missing)) {
  cat("Installing missing packages...\n")
  install.packages(missing, repos = "https://cloud.r-project.org")
}
invisible(lapply(packages, library, character.only = TRUE))
set.seed(42)

# Load dataset using data.table
library(data.table)
dt <- fread("dataset.csv", encoding = "UTF-8", na.strings = c("", "NA"))

cn <- tolower(names(dt))
text_col_idx <- which(cn %in% c("text","reviewtext","review_text","content","body","reviews","summary"))
if (!length(text_col_idx)) text_col_idx <- grep("(text|review|content|body|comment)", cn, ignore.case = TRUE)
stopifnot("No Text-like column found" = length(text_col_idx) >= 1)
text_col <- names(dt)[text_col_idx[1]]
txt <- unique(trimws(as.character(dt[[text_col]])))
txt <- txt[nchar(txt) > 0]

# ========== TF-IDF ==========
it <- itoken(txt, preprocessor = tolower, tokenizer = text2vec::word_tokenizer, progressbar = FALSE)
vocab <- create_vocabulary(it, ngram = c(1L,2L), stopwords = stopwords::stopwords("en"))
vocab <- prune_vocabulary(vocab, doc_proportion_min = 0.002, doc_proportion_max = 0.90, term_count_min = 2)
dtm <- create_dtm(it, vocab_vectorizer(vocab))
X_tfidf <- TfIdf$new(norm = "l2")$fit_transform(dtm)

top_terms_per_cluster <- function(X_sparse, labels, top_n = 10) {
  terms <- colnames(X_sparse)
  out <- lapply(sort(unique(labels)), function(cl) {
    if (cl < 0) return(NULL)
    rows <- which(labels == cl); if (!length(rows)) return(NULL)
    s <- Matrix::colSums(X_sparse[rows, , drop = FALSE])
    top_idx <- order(s, decreasing = TRUE)[seq_len(min(top_n, length(s)))]
    data.frame(cluster = cl, rank = seq_along(top_idx),
               term = terms[top_idx], score = as.numeric(s[top_idx]))
  })
  do.call(rbind, out)
}

# ========== PCA-50 ==========
pca50 <- irlba::prcomp_irlba(X_tfidf, n = 50, center = TRUE, scale. = FALSE)
X_pca50 <- pca50$x

# ========== Clustering ==========
km <- kmeans(X_pca50, centers = 3, nstart = 20, iter.max = 200); km_labels <- km$cluster
D_cos_pca <- proxy::dist(X_pca50, method = "cosine")
hc <- hclust(as.dist(D_cos_pca), method = "average"); hc_labels <- cutree(hc, k = 5)
eps <- 1.18; minPts <- 20
db <- dbscan::dbscan(X_pca50, eps = eps, minPts = minPts); db_labels <- db$cluster

# ========== Analysis ==========
sil_km <- cluster::silhouette(as.integer(km_labels), D_cos_pca)
sil_hc <- cluster::silhouette(as.integer(hc_labels), D_cos_pca)
idx_db <- which(db_labels != 0)
if (length(idx_db) >= 2 && length(unique(db_labels[idx_db])) > 1) {
  D_sub <- proxy::dist(X_pca50[idx_db, , drop = FALSE], method = "cosine")
  lab_db <- as.integer(factor(db_labels[idx_db]))
  sil_db <- cluster::silhouette(lab_db, D_sub)
  cat("Silhouette (mean) — KM:", mean(sil_km[,3]),
      " HC:", mean(sil_hc[,3]),
      " DBSCAN(no-noise):", mean(sil_db[,3]), "\n")
} else {
  cat("Silhouette (mean) — KM:", mean(sil_km[,3]),
      " HC:", mean(sil_hc[,3]),
      " DBSCAN(no-noise): NA\n")
}
cat("\nK-means top terms:\n");        print(top_terms_per_cluster(X_tfidf, km_labels, 12))
cat("\nHierarchical top terms:\n");   print(top_terms_per_cluster(X_tfidf, hc_labels, 12))
db_adj <- ifelse(db_labels == 0, -1, db_labels)
cat("\nDBSCAN top terms:\n");         print(top_terms_per_cluster(X_tfidf, db_adj, 12))
cat("\nCluster sizes — K-means:\n");  print(table(km_labels))
cat("\nCluster sizes — Hierarchical:\n"); print(table(hc_labels))
cat("\nCluster sizes — DBSCAN (0=noise):\n"); print(table(db_labels))
cat(sprintf("\nDBSCAN clusters (excl. noise): %d | Noise fraction: %.3f\n",
            length(setdiff(unique(db_labels), 0)), mean(db_labels == 0)))

# ========== Visualization (PCA-3) ==========
pca3 <- irlba::prcomp_irlba(X_tfidf, n = 3, center = TRUE, scale. = FALSE)
pcs <- data.frame(PC1 = pca3$x[,1], PC2 = pca3$x[,2], PC3 = pca3$x[,3],
                  km = factor(km_labels), hc = factor(hc_labels),
                  db = factor(ifelse(db_labels == 0, "noise", paste0("C", db_labels))))
centers_pca3 <- aggregate(pcs[, c("PC1","PC2","PC3")], by = list(km = pcs$km), FUN = mean); names(centers_pca3)[1] <- "km"

ggplot(pcs, aes(PC1, PC2, colour = km)) + geom_point(alpha=.6, size=1.1) +
  geom_point(data=centers_pca3, aes(PC1, PC2, colour=km), shape=4, size=4, stroke=1.2) +
  theme_minimal() + ggtitle("PC1 vs PC2 (K-means)")

ggplot(pcs, aes(PC2, PC3, colour = hc)) + geom_point(alpha=.6, size=1.1) +
  theme_minimal() + ggtitle("PC2 vs PC3 (Hierarchical)")

ggplot(pcs, aes(PC1, PC3, colour = db)) + geom_point(alpha=.6, size=1.1) +
  theme_minimal() + ggtitle("PC1 vs PC3 (DBSCAN)")

# ========== Cluster Analytics ==========
all_txt <- trimws(as.character(dt[[text_col]]))
row_idx <- match(txt, all_txt)
docs <- as.data.table(dt[row_idx, , drop = FALSE])
docs[, km := km_labels]; docs[, hc := hc_labels]
stopifnot(all(c("Score","ProductId") %in% names(docs)))

label_pos_neg <- function(x) {
  if (is.numeric(x)) data.table(pos = as.integer(x > 0), neg = as.integer(x < 0)) else {
    xl <- tolower(trimws(as.character(x)))
    POS <- c("positive","pos","+","1","true","yes","good","favorable","favourable","recommend")
    NEG <- c("negative","neg","-","-1","false","no","bad","unfavorable","unfavourable","norecommend")
    data.table(pos = as.integer(xl %in% POS), neg = as.integer(xl %in% NEG))
  }
}
pct_posneg_by_cluster <- function(d, cluster_col) {
  labs <- label_pos_neg(d[["Score"]]); tmp <- cbind(cluster = d[[cluster_col]], labs)
  as.data.table(tmp)[, .(
    n_docs = .N,
    positive = sum(pos, na.rm=TRUE),
    negative = sum(neg, na.rm=TRUE),
    pos_pct = round(100 * sum(pos, na.rm=TRUE)/.N, 1),
    neg_pct = round(100 * sum(neg, na.rm=TRUE)/.N, 1)
  ), by = cluster][order(cluster)]
}
top5_products_per_cluster <- function(d, cluster_col) {
  d[!is.na(ProductId) & nzchar(trimws(as.character(ProductId))),
    .(cluster = get(cluster_col), ProductId)][, .N, by = .(cluster, ProductId)][
      order(cluster, -N)][, head(.SD, 5), by = cluster]
}

km_posneg <- pct_posneg_by_cluster(docs, "km"); cat("\nK-means — % Pos/Neg:\n"); print(km_posneg)
km_top5   <- top5_products_per_cluster(docs, "km"); cat("\nK-means — Top-5 ProductId:\n"); print(km_top5)
hc_posneg <- pct_posneg_by_cluster(docs, "hc"); cat("\nHierarchical — % Pos/Neg:\n"); print(hc_posneg)
hc_top5   <- top5_products_per_cluster(docs, "hc"); cat("\nHierarchical — Top-5 ProductId:\n"); print(hc_top5)
