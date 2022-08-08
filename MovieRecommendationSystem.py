import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from sklearn import model_selection
# This is an interactive movie recommendation system using Collaborative Filtering

# Reading data using pandas
movies = pd.read_csv("movies.csv")
print(movies)

# Cleaning movie titles using Regex
def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]","",title)     #if not an alphabet/digit or space then, remove the character

# creating a column in a dataframe "movies" called clean_title,
# which applies clean_title() function on movie titles
movies["clean_title"] = movies["title"].apply(clean_title)
print(movies)

# Creating a TFIDF matrix (Time Frequency Matrix * Inverse Document Frequency Matrix)(Scalar Product)
# Converting title of the movies into set of numbers so that computer can search them effectively
vectorized = TfidfVectorizer(ngram_range=(1,2))
tfidf = vectorized.fit_transform(movies["clean_title"])

# Creating search function
def search(title):
    title = clean_title(title)
    query_vec = vectorized.transform([title])
    similarity = cosine_similarity(query_vec,tfidf).flatten()  #compare query_vec with the cleaned titles and score the extent of similarity
    indices = np.argpartition(similarity,-5)[-5:]
    results = movies.iloc[indices][::-1] #reversing the result as the most relevant result will be in  the last
    return results

# Reading in movie ratings data
ratings = pd.read_csv("ratings.csv")
# Finding users who liked the same movie by building a recommendation function
def find_similar_movies(movie_id):
    # Finding unique users who liked the same movie
    similar_user = ratings[(ratings["movieId"]==movie_id)] & (ratings["rating"] > 4)["userId"].unique()
    # Finding movieIds having similar rating rated by the same users
    similar_user_recs = ratings[(ratings["userId"].isin(similar_user)) & (ratings["rating"] > 4)]["movieId"]
    # We only need 10% of the movies for recommendation rather than all the movies
    similar_user_recs = similar_user_recs.value_counts()/len(similar_user)
    similar_user_recs = similar_user_recs[similar_user_recs>.1]

    # Finding how much all users like movies
    # all users who have watched all the movies that were recommended
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    #percentage of all users who recommended the movies
    all_users_rec = all_users["movieId"] / len(all_users["userId"].unique())

    # Creating a recommendation score
    rec_percentages = pd.concat([similar_user_recs, all_users_rec], axis=1)
    rec_percentages.columns = ["similar", "all"] #similar gives how much users similar to us like a movie
    # all gives how much an average user likes the same movie

    # getting ratio between similar and all in order to get the greatest difference between these two
    # by sorting the result later.
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("similar",ascending=False)

    # merging top 10 recommendation scores with movies dataframe using index of rec_percentage dataframe
    # merging that with movieId columns
    return rec_percentages.head(10).merge("movies",left_index=True,right_on="movieId")[["score","title","genres"]]

# Creating an interactive recommendation widget
# Creating an interactive search box
movie_input_name = widgets.Text(
    value="Toy Story",
    description = "Movie Title:",
    disabled = False
)
# Output widget
recommendation_list = widgets.Output()

# function is called whenever we type inside the textbox
def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"] #Input is a dictionary and "new" field will give new value which was entered in the input widget
        if len(title) > 5:
            results = search(title) #Display the relevant searched title
            movie_id = results.iloc[0]["movieId"]
            display(find_similar_movies(movie_id))
movie_input_name.observe(on_type,names="value")  # look for the value typed
display(movie_input_name, recommendation_list)











