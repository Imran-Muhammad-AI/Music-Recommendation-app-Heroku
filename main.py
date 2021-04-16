import pandas as pd
import seaborn as sns
from scipy import sparse as sp
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import pairwise as pw
from flask import Flask, request, jsonify, render_template, url_for
import pickle


app = Flask(__name__)
 
#@app.route('/')
#def home():
#    return render_template('home.html')


##@app.route('/')
##def recommend():
    ##uid = request.args.get('uid',type=int)
    ## saved model
@app.route('/')
def home():
    import random
    uid= random.randint(0,10)

    hybridModel = pickle.load(open('model.pkl','rb'))
    data=pd.read_csv('data5000.csv')
    
    data=data.head(1000).copy()
    
    # Create a pivot table (interaction matrix) from the original dataset
    x = data.pivot_table(index='user_id', columns='song_id', values='listen_count')
    
    # Creating user dictionary based on their index and number in the interaction matrix using recsys library
    userDict = create_user_dict(interactions=x)
    
    # Creating a song dictionary based on their songID and artist name
    songDict = create_item_dict(df=data, id_col='song_id', name_col='title')
    
    recomendList = sample_recommendation_user(model=hybridModel, interactions=x, user_id=uid, 
                           user_dict=userDict, item_dict=songDict, threshold=5, nrec_items=10)
    recomendList= data[data['title'].isin(recomendList)]    
    return render_template('home.html',userid=uid,rlist=recomendList)

@app.route('/recommendBySong')
def recommendBySong():
    song = request.args.get('songid')
    ## saved model
    hybridModel = pickle.load(open('model.pkl','rb'))
    data=pd.read_csv('data5000.csv')
      
    # Create a pivot table (interaction matrix) from the original dataset
    x = data.pivot_table(index='user_id', columns='song_id', values='listen_count')

    
    # Creating user dictionary based on their index and number in the interaction matrix using recsys library
    userDict = create_user_dict(interactions=x)
    
    # Creating a song dictionary based on their songID and artist name
    songDict = create_item_dict(df=data, id_col='song_id', name_col='title')
    
    # Recommend songs similar to a given songID

    songID = get_key(song, songDict)
    songItemDist = create_item_emdedding_distance_matrix(model=hybridModel, interactions=x)
    itemrecomendList=item_item_recommendation(item_emdedding_distance_matrix=songItemDist, item_id=songID,item_dict=songDict, n_items=10)
    itemrecomendList= data[data['song_id'].isin(itemrecomendList)]  

    return render_template('songbysongresult.html',songtitle=song,rlist=itemrecomendList)
    

## Function to produce user recommendations
def sample_recommendation_user(model, interactions, user_id, user_dict,item_dict, threshold = 0, nrec_items = 10):
    n_users, n_items = interactions.shape
    user_x = user_dict[user_id]
    scores = pd.Series(model.predict(user_x, np.arange(n_items)))
    scores.index = interactions.columns
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    
    known_items = list(pd.Series(interactions.loc[user_id, :] \
                                 [interactions.loc[user_id, :] > threshold].index) \
								   .sort_values(ascending=False))
    
    scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0: nrec_items]
    known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
    
    return scores
    
    #print("Recommended songs for UserID:", user_id)
    #counter = 1

    #for i in scores:
    #    print(str(counter) + '- ' + i)
    #   counter+=1
        
# Function to create a user dictionary based on their index and number in interaction dataset
def create_user_dict(interactions):
    user_id = list(interactions.index)
    user_dict = {}
    counter = 0 

    for i in user_id:
        user_dict[i] = counter
        counter += 1

    new_dict = dict([(value, key) for key, value in user_dict.items()])
    print(new_dict)

    return new_dict

# Function to create an item dictionary based on their item_id and item name  
def create_item_dict(df, id_col, name_col):
    item_dict ={}

    for i in range(df.shape[0]):
        item_dict[(df.loc[i, id_col])] = df.loc[i, name_col]

    return item_dict

# Function to create item-item distance embedding matrix
def create_item_emdedding_distance_matrix(model, interactions):
    
    df_item_norm_sparse = sp.csr_matrix(model.item_embeddings)
    similarities = pw.cosine_similarity(df_item_norm_sparse)
    item_emdedding_distance_matrix = pd.DataFrame(similarities)
    item_emdedding_distance_matrix.columns = interactions.columns
    item_emdedding_distance_matrix.index = interactions.columns
    
    return item_emdedding_distance_matrix

# Function to create item-item recommendation
def item_item_recommendation(item_emdedding_distance_matrix, item_id, item_dict, n_items = 10):
    
    recommended_items = list(pd.Series(item_emdedding_distance_matrix.loc[item_id,:]. \
                                  sort_values(ascending = False).head(n_items+1). \
                                  index[1:n_items+1]))
    return recommended_items

# Function to return key for any value
def get_key(val, dictionary):
    for key, value in dictionary.items():
         if val == value:
                return key
            
            
if __name__ == '__main__':
    app.run(debug=True)
    
    