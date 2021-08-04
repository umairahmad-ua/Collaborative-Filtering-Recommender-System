import numpy as np 
import pandas as pd
from collections import defaultdict
from scipy.io import loadmat
from scipy import spatial
from scipy.optimize import minimize
import pickle as pkl
import sys
J_history =[]
movieIDList = []


# Please Uncomment For Retraining 
'''
# Provide Path and Size
def file_read(path,size):
    df=pd.read_csv(path)
    df=df.head(size)
    return df
df=file_read("C:/Users/Umair/Desktop/ml-25m/ratings.csv",100000)
print("file reading done")

# Preparing Y and R Matrix @Provide DataFrame of user ratings

def Y_and_R_matrix(df):
    user_movie_rating = df.pivot_table(index='movieId', columns='userId', values='rating')
    user_movie_rating=user_movie_rating.fillna(0)
    R = user_movie_rating.apply(lambda x: [y if y <=0 else 1 for y in x])
    Y = user_movie_rating.to_numpy()
    R = R.to_numpy()
    return Y,R

Y,R=Y_and_R_matrix(df)
with open('Y.pkl','wb') as f:
    pkl.dump(Y, f)
print("Y and R Matrix Ready")

# Preparing Features @provide movies  path 
print("Feature Preparing")
def get_movie_features(path):
    movies=pd.read_csv(path)
    movies.genres = movies.genres.str.split('|')
    genre_columns = list(set([j for i in movies['genres'].tolist() for j in i]))
    for j in genre_columns:
        movies[j] = 0
    for i in range(movies.shape[0]):
        for j in genre_columns:
            if(j in movies['genres'].iloc[i]):
                movies.loc[i,j] = 1
    movies.drop('genres',axis=1,inplace=True)
    movieList = []
    movieList=movies["title"]
    movieIDList=movies["movieId"]
    features=movies
    features.drop('title',axis=1,inplace=True)
    features.drop('movieId',axis=1,inplace=True)
    features = features.to_numpy()
    features=features[:Y.shape[0],:]
    array = np.random.randint(10, size=(Y.shape[0] , 29))
    movie_features=np.append(features, array, axis=1)
    movie_features=features
    return movie_features,movieIDList
movie_features,movieIDList=get_movie_features("C:/Users/Umair/Desktop/ml-25m/movies.csv")
with open('movieIDList.pkl','wb') as f:
    pkl.dump(movieIDList, f)
print("Feature Prepared")

# Calculate the Cost by providing X,Theta in Params and Y,R matrix
def cost(params, Y, R, num_features, learning_rate):  
    Y = np.matrix(Y)  
    R = np.matrix(R) 
    num_movies = Y.shape[0]
    num_users = Y.shape[1]
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))
    Theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))
    X_grad = np.zeros(X.shape) 
    Theta_grad = np.zeros(Theta.shape) 
    J = 0
    temp=(X * Theta.T)
    error = np.multiply(temp - Y, R) 
    squared_error = np.power(error, 2)  
    J = (1. / 2) * np.sum(squared_error)
    #Regulirization Applied
    J = J + ((learning_rate / 2) * np.sum(np.power(Theta, 2)))
    J = J + ((learning_rate / 2) * np.sum(np.power(X, 2)))
    J_history.append(J)
    X_grad = (error * Theta) + (learning_rate * X)
    Theta_grad = (error.T * X) + (learning_rate * Theta)
    grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))
    return J, grad

print("Training Started")
# Training on movies features and Theta by providing R and Y Matrix 
def training():
    movies = Y.shape[0]  
    users = Y.shape[1]  
    features = 20  
    learning_rate = 0.01
    X = movie_features 
    Theta = np.random.random(size=(users, features)) 
    params = np.concatenate((np.ravel(X), np.ravel(Theta)))
    Ymean = np.zeros((movies, 1))  
    Ynorm = np.zeros((movies, users))
    for i in range(movies):  
        idx = np.where(R[i,:] == 1)[0]
        Ymean[i] = Y[i,idx].mean()
        Ynorm[i,idx] = Y[i,idx] - Ymean[i]
    fmin = minimize(fun=cost, x0=params, args=(Ynorm, R, features, learning_rate),  method='CG', jac=True, options={'maxiter': 100})
    X = np.matrix(np.reshape(fmin.x[:movies * features], (movies, features)))  
    Theta = np.matrix(np.reshape(fmin.x[movies * features:], (users, features)))
    return X,Theta,Ymean

X,Theta,Ymean=training()
Model = X * Theta.T
Model=Model+Ymean
Model[Model > 5] = 5
with open('Model.pkl','wb') as f:
    pkl.dump(Model, f)
print("Model Saved as Pickle")





'''
# Testing , We will only load the pretrained Model no retraining 

def flaten(array):
    pred=[]
    for x in array:
        pred.append(int(x))
    return pred
def test_user_predictions(path):
    with open('Model.pkl','rb') as f:
        predictions = pkl.load(f)
    with open('movieIDList.pkl','rb') as f:
        movieIDList = pkl.load(f)
    with open('Y.pkl','rb') as f:
        Y = pkl.load(f)
    movieIDList=movieIDList.values.flatten()
    test_user=pd.read_csv(path)
    test_user_movies=test_user["movieId"].to_numpy()
    test_real_rating=test_user["rating"].to_numpy()
    index_no =[]
    for x in test_user_movies:
        no=np.where(movieIDList == x)
        index_no.append(int(no[0]))
    results=predictions
    cosine_similarity_results=[]
    for x in results.T:
        temp=np.take(x, index_no)
        cosine_similarity = 1 - spatial.distance.cosine(temp, test_real_rating)
        cosine_similarity_results.append(cosine_similarity)
    similar = np.where(cosine_similarity_results == np.amax(cosine_similarity_results))
    Theta_for_prediction=results[:,int(similar[0])]
    df = pd.DataFrame({"MovieID" : movieIDList[:len(flaten(Theta_for_prediction))], "Predicted Ratings" : flaten(Theta_for_prediction)})
    df.to_csv("test_user_predictions.csv", index=False)
    five_similar = np.argpartition(np.array(cosine_similarity_results), -5)[-5:]
    df = pd.DataFrame({"Test User" : test_user_movies[np.argpartition(np.array(test_real_rating), -10)[-10:]],
                       "User1" : movieIDList [np.argpartition(np.array(Y[:,five_similar[0]]), -10)[-10:]],
                       "User2":  movieIDList [np.argpartition(np.array(Y[:,five_similar[1]]), -10)[-10:]],
                       "User3":  movieIDList [np.argpartition(np.array(Y[:,five_similar[2]]), -10)[-10:]],
                       "User4":  movieIDList [np.argpartition(np.array(Y[:,five_similar[3]]), -10)[-10:]],
                       "User5":  movieIDList [np.argpartition(np.array(Y[:,five_similar[4]]), -10)[-10:]],
                      })
    df.to_csv("five_similar_users.csv", index=False)

path = sys.argv[1]
path=path+".csv"
test_user_predictions(path)
print("Test User Ratings saved as CSV ")
print("Five similar Users saved as CSV ")