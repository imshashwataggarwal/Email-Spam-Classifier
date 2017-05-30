# Data source: http://spamassassin.apache.org/publiccorpus/

# Load libraries
library(tm);
library(ggplot2);

setwd("/home/suchet/Desktop/Machine Learning/SpamClassifier")

# Assign Paths :-

# Training Dataset
spam_train.path     <- "data/spam/"
easy_ham_train.path <- "data/easy_ham/"
hard_ham_train.path <- "data/hard_ham/"

# Testing Dataset
spam_test.path     <- "data/spam_2/"
easy_ham_test.path <- "data/easy_ham_2/"
hard_ham_test.path <- "data/hard_ham_2/"

# Extract Email Body ; Single Element Vector
get.msg <- function(path) {
    f <- file(path, open ="rt", encoding = "latin1")
    text <- readLines(f)
    msg  <- text[seq(which(text == "")[1] + 1, length(text), 1)]
    close(f)
    return(paste(msg, collapse ="\n"))
}

# Term Document Matrix
get.tdm <- function(doc.vec) {
    doc.corpus <- Corpus(VectorSource(doc.vec))
    control <- list(stopwords = TRUE, removePunctuation = TRUE, removeNumbers = TRUE, minDocFreq = 2)
    doc.tdm <- TermDocumentMatrix(doc.corpus, control)
    return(doc.tdm)
}

# Naive Bayes Classifier
classify.email <- function(path, train.df, prior = 0.5, q = 1e-6) {
    msg      <- get.msg(path)
    msg.tdm  <- get.tdm(msg)
    msg.freq <- rowSums(as.matrix(msg.tdm))
    msg.intersect <- intersect(names(msg.freq), train.df$term)
    if(length(msg.intersect) < 1){
        return(prior * q ^ (length(msg.freq)))
    }
    else{
        match.prob <- train.df$occurrence[match(msg.intersect, train.df$term)]
        return(prior * prod(match.prob) * q ^ (length(msg.freq)-length(msg.intersect)))
    }
}

# Training :-

# SPAM DATA SET

# Get all the SPAM Emails into a single character vector
spam.docs <- dir(spam_train.path)
spam.docs <- spam.docs[which(spam.docs != "cmds")]
all.spam  <- sapply(spam.docs, function(x) get.msg(file.path(spam_train.path,x)))

# DocumentTermMatrix from that vector
spam.tdm  <- get.tdm(all.spam)
spam.matrix <- as.matrix(spam.tdm)
spam.counts <- rowSums(spam.matrix)

# Data frame 
spam.df <- data.frame(cbind(names(spam.counts), as.numeric(spam.counts)), stringsAsFactors = FALSE)
names(spam.df) <- c("term", "frequency")
spam.df$frequency <- as.numeric(spam.df$frequency)
spam.occurrence <- sapply(1:nrow(spam.matrix),function(i){
                            length(which(spam.matrix[i, ] > 0)) / ncol(spam.matrix)
                          })
spam.density  <- spam.df$frequency / sum(spam.df$frequency)
spam.df <- transform(spam.df, density = spam.density, occurrence = spam.occurrence)


# HAM DATA SET

# Get all the HAM Emails into a single character vector
easy_ham.docs <- dir(easy_ham_train.path)
easy_ham.docs <- easy_ham.docs[which(easy_ham.docs != "cmds")]
all.easy_ham  <- sapply(easy_ham.docs, function(x) get.msg(file.path(easy_ham_train.path,x)))

# DocumentTermMatrix from that vector
easy_ham.tdm  <- get.tdm(all.easy_ham)
easy_ham.matrix <- as.matrix(easy_ham.tdm)
easy_ham.counts <- rowSums(easy_ham.matrix)

# Data frame 
easy_ham.df     <- data.frame(cbind(names(easy_ham.counts), as.numeric(easy_ham.counts)), stringsAsFactors = FALSE)
names(easy_ham.df)    <- c("term", "frequency")
easy_ham.df$frequency <- as.numeric(easy_ham.df$frequency)
easy_ham.occurrence   <- sapply(1:nrow(easy_ham.matrix), function(i){
                            length(which(easy_ham.matrix[i, ] > 0)) / ncol(easy_ham.matrix)
                          })
easy_ham.density <- easy_ham.df$frequency / sum(easy_ham.df$frequency)
easy_ham.df      <- transform(easy_ham.df, density = easy_ham.density, occurrence = easy_ham.occurrence)

# HARD HAM 

hard_ham.docs <- dir(hard_ham_train.path)
hard_ham.docs <- hard_ham.docs[which(hard_ham.docs != "cmds")]

hard_ham.spamtest <- sapply(hard_ham.docs, function(x) classify.email(file.path(hard_ham_train.path, x), train.df = spam.df))
hard_ham.hamtest  <- sapply(hard_ham.docs, function(x) classify.email(file.path(hard_ham_train.path, x), train.df = easy_ham.df))

hard_ham.res <- ifelse(hard_ham.spamtest > hard_ham.hamtest, TRUE, FALSE)
# Check
summary(hard_ham.res)

# Test DATA
spam.classifier <- function(path)
{
  pr.spam <- classify.email(path, spam.df)
  pr.ham  <- classify.email(path, easy_ham.df)
  return(c(pr.spam, pr.ham, ifelse(pr.spam > pr.ham, 1, 0)))
}

easy_ham2.docs <- dir(easy_ham_test.path)
easy_ham2.docs <- easy_ham2.docs[which(easy_ham2.docs != "cmds")]

hard_ham2.docs <- dir(hard_ham_test.path)
hard_ham2.docs <- hard_ham2.docs[which(hard_ham2.docs != "cmds")]

spam2.docs <- dir(spam_test.path)
spam2.docs <- spam2.docs[which(spam2.docs != "cmds")]

# Classify them all!
easy_ham2.class <- suppressWarnings(lapply(easy_ham2.docs,
                                   function(x)
                                   {
                                     spam.classifier(file.path(easy_ham_test.path, x))
                                   }))
hard_ham2.class <- suppressWarnings(lapply(hard_ham2.docs,
                                   function(x)
                                   {
                                     spam.classifier(file.path(hard_ham_test.path, x))
                                   }))
spam2.class <- suppressWarnings(lapply(spam2.docs,
                                function(x)
                                {
                                  spam.classifier(file.path(spam_test.path, x))
                                }))

easy_ham2.matrix <- do.call(rbind, easy_ham2.class)
easy_ham2.final  <- cbind(easy_ham2.matrix, "EASYHAM")

hard_ham2.matrix <- do.call(rbind, hard_ham2.class)
hard_ham2.final  <- cbind(hard_ham2.matrix, "HARDHAM")

spam2.matrix <- do.call(rbind, spam2.class)
spam2.final  <- cbind(spam2.matrix, "SPAM")

class.matrix <- rbind(easy_ham2.final, hard_ham2.final, spam2.final)
class.df <- data.frame(class.matrix, stringsAsFactors = FALSE)
names(class.df) <- c("Pr.SPAM" ,"Pr.HAM", "Class", "Type")
class.df$Pr.SPAM <- as.numeric(class.df$Pr.SPAM)
class.df$Pr.HAM  <- as.numeric(class.df$Pr.HAM)
class.df$Class <- as.logical(as.numeric(class.df$Class))
class.df$Type <- as.factor(class.df$Type)

# Create final plot of results
class.plot <-   ggplot(class.df, aes(x = log(Pr.HAM), log(Pr.SPAM))) +
                geom_point(aes(shape = Type, alpha = 0.5)) +
                scale_shape_manual(values = c( "EASYHAM" = 1, "HARDHAM" = 2, "SPAM" = 3), name = "Email Type") +
                scale_alpha(guide = "none") +
                xlab("log[P(HAM)]") +
                ylab("log[P(SPAM)]") +
                #theme_bw() +
                theme(axis.text.x = element_blank(), axis.text.y = element_blank())

                ggsave(plot = class.plot, filename = file.path("images", "result.png"), height = 10, width = 10)

get.results <- function(bool.vector)
{
  results <- c(length(bool.vector[which(bool.vector == FALSE)]) / length(bool.vector),
               length(bool.vector[which(bool.vector == TRUE)]) / length(bool.vector))
  return(results)
}

# Save results as a 2x3 table
EasyHam <- get.results(subset(class.df, Type == "EASYHAM")$Class)
HardHam <- get.results(subset(class.df, Type == "HARDHAM")$Class)
Spam    <- get.results(subset(class.df, Type == "SPAM")$Class)

class.res <- rbind(EasyHam, HardHam, Spam)
colnames(class.res) <- c("NOT SPAM", "SPAM")
print(class.res)

write.csv(spam.df, file.path("data", "spam_df.csv"), row.names = FALSE)
write.csv(easy_ham.df, file.path("data", "easyham_df.csv"), row.names = FALSE)

write.csv(class.df, file.path("Results", "result_df.csv"), row.names = FALSE)
write.table(class.res, file.path("Results", "classify.csv"), row.names=FALSE)

