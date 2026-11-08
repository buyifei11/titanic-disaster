# Titanic survival prediction with logistic regression (R version)
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

# Inside Docker DATA_DIR=/app/data; locally run from the repo root
data_dir <- Sys.getenv("DATA_DIR", unset = "src/data")
title_levels <- c("Mr", "Mrs", "Miss", "Master", "Rare")

section <- function(title) {
  cat("\n", strrep("=", 60), "\n", title, "\n", strrep("=", 60), "\n", sep = "")
}

load_csv <- function(name) {
  path <- file.path(data_dir, name)
  if (!file.exists(path)) {
    stop(paste0(path, " not found. Put the Kaggle file '", name, "' in src/data/ (see README)."))
  }
  df <- read_csv(path, show_col_types = FALSE)
  cat("Loaded", name, ":", nrow(df), "rows x", ncol(df), "columns\n")
  df
}

extract_title <- function(names) {
  t <- trimws(sub("^[^,]*,\\s*([^.]+)\\..*$", "\\1", names))
  t[t %in% c("Mlle", "Ms")] <- "Miss"
  t[t == "Mme"] <- "Mrs"
  t[!t %in% title_levels] <- "Rare"
  factor(t, levels = title_levels)
}

prepare <- function(df, fill, label) {
  cat("\n[", label, "] Missing values before cleaning:\n", sep = "")
  print(colSums(is.na(df[, c("Age", "Fare", "Embarked", "Cabin")])))

  n_age <- sum(is.na(df$Age))
  n_fare <- sum(is.na(df$Fare))
  n_emb <- sum(is.na(df$Embarked))

  df <- df %>%
    mutate(
      Age = ifelse(is.na(Age), fill$Age, Age),
      Fare = ifelse(is.na(Fare), fill$Fare, Fare),
      Embarked = ifelse(is.na(Embarked), fill$Embarked, Embarked)
    )
  cat("[", label, "] Filled ", n_age, " missing 'Age' values with ", fill$Age, "\n", sep = "")
  cat("[", label, "] Filled ", n_fare, " missing 'Fare' values with ", fill$Fare, "\n", sep = "")
  cat("[", label, "] Filled ", n_emb, " missing 'Embarked' values with ", fill$Embarked, "\n", sep = "")

  df <- df %>% select(-Cabin, -Ticket)
  cat("[", label, "] Dropping columns not used as features: Cabin, Ticket\n", sep = "")

  df <- df %>%
    mutate(
      Title = extract_title(Name),
      FamilySize = SibSp + Parch + 1,
      IsAlone = as.integer(FamilySize == 1),
      Sex = factor(Sex, levels = c("male", "female")),
      Pclass = factor(Pclass, levels = 1:3),
      Embarked = factor(Embarked, levels = c("S", "C", "Q"))
    )
  cat("[", label, "] New feature 'Title' extracted from Name:\n", sep = "")
  print(table(df$Title))
  cat("[", label, "] New features 'FamilySize' (SibSp + Parch + 1) and 'IsAlone'; ",
      sum(df$IsAlone), " passengers travelled alone\n", sep = "")
  cat("[", label, "] Converted Sex, Pclass and Embarked to factors\n", sep = "")
  df
}

# ---------------------------------------------------------------
section("STEP 1: Load training data")
train_raw <- load_csv("train.csv")
print(head(train_raw))
cat("\nOverall survival rate in train.csv:", round(mean(train_raw$Survived), 3), "\n")

section("STEP 2: Explore the data")
cat("Survival rate by Sex:\n")
print(train_raw %>% group_by(Sex) %>% summarise(rate = round(mean(Survived), 3)))
cat("\nSurvival rate by Pclass:\n")
print(train_raw %>% group_by(Pclass) %>% summarise(rate = round(mean(Survived), 3)))

section("STEP 3: Split train.csv into training (80%) and validation (20%)")
set.seed(42)
idx <- sample(seq_len(nrow(train_raw)), size = floor(0.8 * nrow(train_raw)))
tr_raw <- train_raw[idx, ]
va_raw <- train_raw[-idx, ]
cat("Training rows:", nrow(tr_raw), ", validation rows:", nrow(va_raw), "\n")

section("STEP 4: Clean and engineer features")
fill <- list(
  Age = median(tr_raw$Age, na.rm = TRUE),
  Fare = median(tr_raw$Fare, na.rm = TRUE),
  Embarked = names(sort(table(tr_raw$Embarked), decreasing = TRUE))[1]
)
cat("Fill values learned from training split: Age =", fill$Age,
    ", Fare =", fill$Fare, ", Embarked =", fill$Embarked, "\n")
tr <- prepare(tr_raw, fill, "train")
va <- prepare(va_raw, fill, "validation")

section("STEP 5: Fit logistic regression")
model_formula <- Survived ~ Pclass + Sex + Age + Fare + FamilySize + IsAlone + Title + Embarked
cat("Formula:", deparse(model_formula), "\n\n")
model <- glm(model_formula, data = tr, family = binomial)
print(round(coef(summary(model)), 3))

section("STEP 6: Accuracy")
predict_class <- function(m, df) as.integer(predict(m, newdata = df, type = "response") > 0.5)
train_acc <- mean(predict_class(model, tr) == tr$Survived)
val_acc <- mean(predict_class(model, va) == va$Survived)
cat("Training accuracy:  ", sprintf("%.4f", train_acc), "\n")
cat("Validation accuracy:", sprintf("%.4f", val_acc), "\n")

section("STEP 7: Predict on test.csv")
test_raw <- load_csv("test.csv")
te <- prepare(test_raw, fill, "test")
preds <- predict_class(model, te)
out <- data.frame(PassengerId = test_raw$PassengerId, Survived = preds)
out_path <- file.path(data_dir, "predictions_r.csv")
write_csv(out, out_path)
cat("Predicted survival rate on test set:", round(mean(preds), 3), "\n")
cat("Saved", nrow(out), "predictions to", out_path, "\n")

section("STEP 8: Test-set accuracy")
cat("test.csv has no 'Survived' column, so true test accuracy is only available\n")
cat("by submitting predictions_r.csv to Kaggle.\n")
gs_path <- file.path(data_dir, "gender_submission.csv")
if (file.exists(gs_path)) {
  gs <- read_csv(gs_path, show_col_types = FALSE)
  merged <- inner_join(out, gs, by = "PassengerId", suffix = c("_pred", "_ref"))
  agree <- mean(merged$Survived_pred == merged$Survived_ref)
  cat("Accuracy against gender_submission.csv (Kaggle's sample labels):", sprintf("%.4f", agree), "\n")
} else {
  cat("gender_submission.csv not found; skipping the comparison.\n")
}

cat("\nDone.\n")