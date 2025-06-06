# movielens_script.R
# Final model code to generate predictions and calculate RMSE

# Load libraries
library(tidyverse)
library(caret)

# Download and load data
zip_file <- "ml-10M100K.zip"
if (!file.exists(zip_file)) {
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", zip_file)
}
unzip(zip_file, exdir = ".")

ratings <- read_lines("ml-10M100K/ml-10M100K/ratings.dat")
movies <- read_lines("ml-10M100K/ml-10M100K/movies.dat")

# Prepare ratings
data_ratings <- as.data.frame(str_split(ratings, fixed("::"), simplify = TRUE))
colnames(data_ratings) <- c("userId", "movieId", "rating", "timestamp")
data_ratings <- data_ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

# Prepare movies
data_movies <- as.data.frame(str_split(movies, fixed("::"), simplify = TRUE))
colnames(data_movies) <- c("movieId", "title", "genres")
data_movies <- data_movies %>% mutate(movieId = as.integer(movieId))

# Merge
movielens <- left_join(data_ratings, data_movies, by = "movieId")

# Split into edx and final holdout test set
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

# Train final model
mu <- mean(edx$rating)

movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

user_avgs <- edx %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Predict and calculate RMSE
final_predictions <- final_holdout_test %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

final_rmse <- sqrt(mean((final_predictions - final_holdout_test$rating)^2))

# Output final RMSE
cat("Final RMSE on holdout set:", final_rmse, "\n")