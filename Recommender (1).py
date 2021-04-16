# Databricks notebook source
import numpy
import pandas
import random

ratings_data = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/ratings.csv')
ratings_data = ratings_data.toPandas()
print(ratings_data.head(10))

# COMMAND ----------

movie_names = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/movies.csv')
movie_names = movie_names.toPandas()
print(movie_names.head(10))

# COMMAND ----------

#Setting up all our data tables
#First goal: a merge table from which we can drop various columns later
movie_data = pandas.merge(ratings_data, movie_names, on='movieId')
print(movie_data.head(10))

# COMMAND ----------

#Goal 2: A title/genre table to show the user alongside their results (this will have a lot of dupliates so remember to drop.na)
movie_data2 = movie_data.drop(['rating','userId','movieId','timestamp'], axis=1)
print(movie_data2.head(10))

# COMMAND ----------

movie_data2 = movie_data2.drop_duplicates()
print(movie_data2.head(10))

# COMMAND ----------

#Now our movie genre table is filled and clean but in order to work with it later, we need to force reset the indexes (use the code below)
movie_genres = pandas.DataFrame.copy(movie_data2)
movie_genres = movie_genres.reset_index()
# fix the column names
movie_genres.columns = ['id','title','genres']
# Make the movie title the index
movie_genres = movie_genres.set_index('title')
movie_genres = movie_genres.drop(['id'], axis=1)
print(movie_genres.head(10))

# COMMAND ----------

#Now return to goal table 1
#Goal 3A: calculate an average rating for each title
movie_data.groupby('title')['rating'].mean()
#Goal 3B: Count how many ratings each movie has (a higher rating count indicates stronger confidence in the rating)
ratings_mean_count = pandas.DataFrame(movie_data.groupby('title')['rating'].mean())
#Goal 3C: Create a table showing title, average rating, and rating count
ratings_mean_count['rating_counts'] = pandas.DataFrame(movie_data.groupby('title')['rating'].count())
print(ratings_mean_count.head(10))

# COMMAND ----------

#Goal 4: Use a pivot table to create a matrix with a row for each user and a column for each movie with values for ratings (there will be a lot of null values because most users didn't rate many movies - this is OK)  We need a table like this to calculate correlation scores that tell us how alike each movie is based on the similarity of ratings given per user.
user_movie_rating = movie_data.pivot_table(index='userId', columns=['title'], values='rating')
print(user_movie_rating.head(10))

# COMMAND ----------

#Goal 5: Create a correlation matrix using the Pearson Correlation against the pivot table
matrix_corr = user_movie_rating.corr(method='pearson', min_periods=5)
print(matrix_corr.head(10))

# COMMAND ----------

print("Data Loaded.")
movies = movie_names.title.unique()
movies = random.sample(list(movies), 1000)
movie = movies[0]
print("Select a movie title from the dropdown above.")
dbutils.widgets.dropdown("movies", movie, [str(x) for x in movies], "Select a movie title")
#MyMovie = str(input("Enter a Movie title from Movies.csv:"))

# COMMAND ----------

MyMovie = dbutils.widgets.get("movies")
print("Selected Movie is", MyMovie)

# COMMAND ----------

#Goal 6: Query the correlation matrix for the user's movie to pull out all ratings relvant to their search

MyMovie_corr = matrix_corr[MyMovie]
MyMovie_corr.dropna(inplace=True)
print(MyMovie_corr.head(10))

# COMMAND ----------

# Again, the dataframe needs to have a forced index reset in order to prepare it for display (use the code below)
# Brute force index reset (creates a numeric index)
MyMovie_corr = MyMovie_corr.reset_index()
# fix the column names
MyMovie_corr.columns = ['title','correlation']
# Make the movie title the index
MyMovie_corr = MyMovie_corr.set_index('title')

# COMMAND ----------

# Now use join to add the averge rating and rating count from Goal 3 table
MyMovie_corr = MyMovie_corr.join(ratings_mean_count)
print(MyMovie_corr.head(10))

# COMMAND ----------

# Also use join to add the movie genre to your results table
corr_MyMovie = MyMovie_corr.join(movie_genres)
print(corr_MyMovie.head(10))

# COMMAND ----------

# And finally, sort your results table with highest correlation at the top, and within equal correlations, show highest rating count at the top.  Then print the first 10 rows
corr_MyMovie.sort_values(by=['correlation','rating_counts'],ascending=False,inplace=True)
print(corr_MyMovie.head(10))
