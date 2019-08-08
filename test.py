import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from scipy import sparse

# IMPORT AND DROP
# Restaurants
restaurants_df = pd.read_csv('./geoplaces2.csv')
restaurants_df = restaurants_df.set_index(['placeID']).drop(columns=['the_geom_meter','address','country','fax','zip','smoking_area','url','Rambience','franchise','area','other_services'])
res_accepted_pay = pd.read_csv('./chefmozaccepts.csv')
res_parking = pd.read_csv('./chefmozparking.csv')
# Users
user_df = pd.read_csv('./userprofile.csv')
user_df = user_df.set_index(['userID']).drop(columns=['smoker','ambience','hijos','marital_status','birth_year','interest','personality','religion','activity','color','weight','height'])
usr_payment = pd.read_csv('./userpayment.csv')
# Ratings
# {userID*, placeID*, rating} (* means key)
ratings_df = pd.read_csv('./rating_final.csv')
ratings_df = ratings_df.drop(columns=['food_rating', 'service_rating'])
ratings_df = ratings_df.set_index(['userID', 'placeID'])

# MAPPING AND ENCODING (OHE)

# User Payment
# {userID*, PAYMENT_cash  PAYMENT_credit_card  PAYMENT_debit_card}
map_user_payment = {
   'cash': 'cash',
   'bank_debit_cards': 'debit_card', 
   'MasterCard-Eurocard': 'credit_card', 
   'VISA': 'credit_card',
   'American_Express': 'credit_card',  
}
usr_payment['payment'] = usr_payment['Upayment'].apply(lambda x: map_user_payment[x])
usr_payment = usr_payment[['userID']].join(pd.get_dummies(usr_payment['payment']).add_prefix('PAYMENT_')).groupby('userID').max()

# Restaurants
map_restaurant_payment = {
   'cash': 'cash',
   'bank_debit_cards': 'debit_card', 
   'MasterCard-Eurocard': 'credit_card', 
   'VISA': 'credit_card',
   'Visa': 'credit_card',
   'American_Express': 'credit_card',  
   'Japan_Credit_Bureau': 'credit_card',  
   'Carte_Blanche': 'credit_card',
   'Diners_Club': 'credit_card', 
   'Discover': 'credit_card', 
   'gift_certificates': 'other',  
   'checks': 'other',  
}
res_accepted_pay['payment'] = res_accepted_pay['Rpayment'].apply(lambda x: map_restaurant_payment[x])
res_accepted_pay = res_accepted_pay[['placeID']].join(pd.get_dummies(res_accepted_pay['payment']).add_prefix('PAYMENT_')).groupby('placeID').max()

map_restaurant_parking = {
   'none': 'no',
   'public': 'yes', 
   'yes': 'yes', 
   'valet parking': 'yes',
   'fee': 'yes',
   'street': 'street',  
   'validated parking': 'yes',  
}
res_parking['parking'] = res_parking['parking_lot'].apply(lambda x: map_restaurant_parking[x])
res_parking = res_parking[['placeID']].join(pd.get_dummies(res_parking['parking']).add_prefix('PARKING_')).groupby('placeID').max()

# MERGE
# Users
# {userID*, 'latitude', 'longitude', 'drink_level', 'dress_preference', 'transport', 'budget', 'PAYMENT_cash', 'PAYMENT_credit_card', 'PAYMENT_debit_card'}
user_df = pd.merge(user_df, usr_payment, how='left', on=['userID'])
# Restaurants
# {placeID*,'latitude', 'longitude', 'name', 'city', 'state', 'alcohol', 'dress_code', 'accessibility', 'price', 'PAYMENT_cash', 'PAYMENT_credit_card', 'PAYMENT_debit_card', 'PAYMENT_other', 'PARKING_no', 'PARKING_street', 'PARKING_yes'}
restaurants_df = pd.merge(restaurants_df, res_accepted_pay, how='left', on=['placeID'])
restaurants_df = pd.merge(restaurants_df, res_parking, how='left', on=['placeID'])
# Map restaurant feature values
restaurants_df.alcohol = restaurants_df.alcohol.map({'No_Alcohol_Served':1,'Wine-Beer':2,'Full_Bar':3})
restaurants_df.dress_code = restaurants_df.dress_code.map({'informal':1,'casual':2,'formal':3})
restaurants_df.accessibility = restaurants_df.accessibility.map({'no_accessibility':1,'completely':2,'partially':3})
restaurants_df.price = restaurants_df.price.map({'low': 2, 'medium': 1, 'high': 3})

# At this point, ratings_df, user_df and restaurants_df are clean
ratings_df.head()
restaurants_df.head()
user_df.head()

ratings_df = ratings_df.reset_index()
restaurants_df = restaurants_df.reset_index()
user_df = user_df.reset_index()
#############################

# EVALUATION
ratings_train_df, ratings_test_df = train_test_split(ratings_df,
                                   stratify=ratings_df['userID'], 
                                   test_size=0.20,
                                   random_state=42)

print('# interactions on Train set: %d' % len(ratings_train_df))
print('# interactions on Test set: %d' % len(ratings_test_df))

#Indexing by userID to speed up the searches during evaluation
ratings_indexed_df = ratings_df.set_index('userID')
ratings_train_indexed_df = ratings_train_df.set_index('userID')
ratings_test_indexed_df = ratings_test_df.set_index('userID')

item_ids = restaurants_df['placeID'].tolist()
content_matrix = restaurants_df.set_index('placeID').fillna(0).drop(columns=['name','city','state'])

def get_person_items(person_id):
    items_per_person_list = list(ratings_df[ratings_df['userID']==person_id]['placeID'])
    item_list = restaurants_df.set_index('placeID').loc[items_per_person_list]
    item_list_cleaned = item_list.fillna(0).drop(columns=['name','city','state'])
    # TODO: Instead of a list, a profile with (1x18) shape should be returned as concensus of all user-rated items
    return item_list_cleaned.head(1)

class ContentBasedRecommender:
    
    MODEL_NAME = 'Content-Based'
    
    def __init__(self, items_df=None):
        self.item_ids = item_ids
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def _get_similar_items_to_user_profile(self, person_id, topn=1000):
        #Computes the cosine similarity between the user profile and all item profiles
        print(get_person_items(person_id))
        print(content_matrix)
        cosine_similarities = cosine_similarity(get_person_items(person_id), content_matrix)
        #Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        #Sort the similar items by similarity
        similar_items = sorted([(item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        #Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        # Commented for evaluation reasons
        # Remove user alrealy interacted items
        # user_interacted_items = get_person_items(user_id).index.values
        # similar_items_filtered = list(filter(lambda x: x[0] not in user_interacted_items, similar_items))
        similar_items_filtered = list(set(similar_items_filtered))
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['placeID', 'recStrength']).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'placeID', 
                                                          right_on = 'placeID')[['recStrength', 'placeID', 'latitude', 'longitude', 'name', 'city', 'state', 'alcohol', 'dress_code', 'accessibility', 'price', 'PAYMENT_cash', 'PAYMENT_credit_card', 'PAYMENT_debit_card', 'PAYMENT_other', 'PARKING_no', 'PARKING_street', 'PARKING_yes']]

        return recommendations_df
    
content_based_recommender_model = ContentBasedRecommender(restaurants_df)
print(content_based_recommender_model.recommend_items('U1100', verbose=True, topn=10)) 
