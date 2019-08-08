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
restaurants_df = restaurants_df.join(pd.get_dummies(restaurants_df['alcohol']).add_prefix('ALCOHOL_')).groupby('placeID').max().drop(columns=['alcohol'])
restaurants_df = restaurants_df.join(pd.get_dummies(restaurants_df['dress_code']).add_prefix('DRESS_CODE_')).groupby('placeID').max().drop(columns=['dress_code'])
restaurants_df = restaurants_df.join(pd.get_dummies(restaurants_df['accessibility']).add_prefix('ACCESSIBILITY_')).groupby('placeID').max().drop(columns=['accessibility'])
restaurants_df = restaurants_df.join(pd.get_dummies(restaurants_df['price']).add_prefix('PRICE_')).groupby('placeID').max().drop(columns=['price'])

# At this point, ratings_df, user_df and restaurants_df are clean
ratings_df.head()
print(list(restaurants_df.head()))
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

# Modified from original algorithm
def get_items_interacted(person_id, interactions_df):
    # Get the user's data and merge in the movie information.
    interacted_items = interactions_df.loc[person_id]['placeID']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

#Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

class ModelEvaluator:
    def get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
        interacted_items = get_items_interacted(person_id, ratings_indexed_df)
        all_items = set(restaurants_df['placeID'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):        
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index

    def evaluate_model_for_user(self, model, person_id):
        #Getting the items in test set
        interacted_values_testset = ratings_test_indexed_df.loc[person_id]
        if type(interacted_values_testset['placeID']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['placeID'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['placeID'])])  
        interacted_items_count_testset = len(person_interacted_items_testset) 

        #Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_items(person_id, 
                                               items_to_ignore=get_items_interacted(person_id, 
                                                                                    ratings_train_indexed_df), 
                                               topn=100)

        hits_at_5_count = 0
        hits_at_10_count = 0
        #For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            #Getting a random sample (100) items the user has not interacted 
            #(to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample = self.get_not_interacted_items_sample(person_id, 
                                                                          sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, 
                                                                          seed=item_id%(2**32))

            #Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            #Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['placeID'].isin(items_to_filter_recs)]                    
            valid_recs = valid_recs_df['placeID'].values
            #Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        #Recall is the rate of the interacted items that are ranked among the Top-N recommended items, 
        #when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        person_metrics = {'hits@5_count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count, 
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return person_metrics

    def evaluate_model(self, model):
        #print('Running evaluation for users')
        people_metrics = []
        for idx, person_id in enumerate(list(ratings_test_indexed_df.index.unique().values)):
            #if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, person_id)  
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics).sort_values('interacted_count', ascending=False)
        
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        
        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}    
        return global_metrics, detailed_results_df
    
model_evaluator = ModelEvaluator()   
############################

# POPULARITY

#Computes the most popular items
item_popularity_df = ratings_df.groupby('placeID')['rating'].sum().sort_values(ascending=False).reset_index()
item_popularity_df.head(10)

class PopularityRecommender:
    
    MODEL_NAME = 'Popularity'
    
    def __init__(self, popularity_df, items_df=None):
        self.popularity_df = popularity_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['placeID'].isin(items_to_ignore)].sort_values('rating', ascending = False).head(topn)

        if verbose: # To show item content (restaurants information)
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'placeID', 
                                                          right_on = 'placeID')[['rating', 'placeID', 'latitude', 'longitude', 'name', 'city', 'state', 'PAYMENT_cash', 'PAYMENT_credit_card', 'PAYMENT_debit_card', 'PAYMENT_other', 'PARKING_no', 'PARKING_street', 'PARKING_yes', 'ALCOHOL_Full_Bar', 'ALCOHOL_No_Alcohol_Served', 'ALCOHOL_Wine-Beer', 'DRESS_CODE_casual', 'DRESS_CODE_formal', 'DRESS_CODE_informal', 'ACCESSIBILITY_completely', 'ACCESSIBILITY_no_accessibility', 'ACCESSIBILITY_partially', 'PRICE_high', 'PRICE_low', 'PRICE_medium']]
        
        return recommendations_df
    
popularity_model = PopularityRecommender(item_popularity_df, restaurants_df)

print('Evaluating Popularity recommendation model...')
pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model)
print('\nGlobal metrics:\n%s' % pop_global_metrics)
pop_detailed_results_df.head(10)

print(popularity_model.recommend_items('U1100', verbose=True)) # userID is not used
# ###############################

# # CONTENT-BASED

item_ids = restaurants_df['placeID'].tolist()
content_matrix = restaurants_df.set_index('placeID').fillna(0).drop(columns=['name','city','state'])

def get_person_items(person_id):
    items_per_person_list = list(ratings_df[ratings_df['userID']==person_id]['placeID'])
    item_list = restaurants_df.set_index('placeID').loc[items_per_person_list]
    item_list['userID'] = person_id
    item_list_cleaned = item_list.fillna(0).drop(columns=['name','city','state'])
    # TODO: Instead of a list, a profile with (1x18) shape should be returned as concensus of all user-rated items
    item_list_cleaned = item_list_cleaned.groupby(['userID']).mean()

    return item_list_cleaned

class ContentBasedRecommender:
    
    MODEL_NAME = 'Content-Based'
    
    def __init__(self, items_df=None):
        self.item_ids = item_ids
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def _get_similar_items_to_user_profile(self, person_id, topn=1000):
        #Computes the cosine similarity between the user profile and all item profiles
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
        # similar_items_filtered = list(set(similar_items_filtered))
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['placeID', 'recStrength']).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'placeID', 
                                                          right_on = 'placeID')[['recStrength', 'placeID', 'latitude', 'longitude', 'name', 'city', 'state', 'PAYMENT_cash', 'PAYMENT_credit_card', 'PAYMENT_debit_card', 'PAYMENT_other', 'PARKING_no', 'PARKING_street', 'PARKING_yes', 'ALCOHOL_Full_Bar', 'ALCOHOL_No_Alcohol_Served', 'ALCOHOL_Wine-Beer', 'DRESS_CODE_casual', 'DRESS_CODE_formal', 'DRESS_CODE_informal', 'ACCESSIBILITY_completely', 'ACCESSIBILITY_no_accessibility', 'ACCESSIBILITY_partially', 'PRICE_high', 'PRICE_low', 'PRICE_medium']]

        return recommendations_df
    
content_based_recommender_model = ContentBasedRecommender(restaurants_df)
print(content_based_recommender_model.recommend_items('U1100', verbose=True, topn=10)) 

print('Evaluating Content-Based Filtering model...')
cb_global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(content_based_recommender_model)
print('\nGlobal metrics:\n%s' % cb_global_metrics)
cb_detailed_results_df.head(10)
#################################

# COLLABORATIVE FILTERING

#Creating a sparse pivot table with users in rows and items in columns
users_items_pivot_matrix_df = ratings_df.pivot(index='userID', 
                                                          columns='placeID', 
                                                          values='rating').fillna(0)

# users_items_pivot_matrix_df.head(10)

users_items_pivot_matrix = users_items_pivot_matrix_df.as_matrix()
# users_items_pivot_matrix[:10]

users_ids = list(users_items_pivot_matrix_df.index)
# users_ids[:10]

#The number of factors to factor the user-item matrix.
NUMBER_OF_FACTORS_MF = 15
#Performs matrix factorization of the original user item matrix
U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)

# U.shape

# Vt.shape

sigma = np.diag(sigma)
# sigma.shape

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
# all_user_predicted_ratings

#Converting the reconstructed matrix back to a Pandas dataframe
cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
# cf_preds_df.head(10)

len(cf_preds_df.columns)

class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False).reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['placeID'].isin(items_to_ignore)].sort_values('recStrength', ascending = False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'placeID', 
                                                          right_on = 'placeID')[['recStrength', 'placeID', 'latitude', 'longitude', 'name', 'city', 'state', 'PAYMENT_cash', 'PAYMENT_credit_card', 'PAYMENT_debit_card', 'PAYMENT_other', 'PARKING_no', 'PARKING_street', 'PARKING_yes', 'ALCOHOL_Full_Bar', 'ALCOHOL_No_Alcohol_Served', 'ALCOHOL_Wine-Beer', 'DRESS_CODE_casual', 'DRESS_CODE_formal', 'DRESS_CODE_informal', 'ACCESSIBILITY_completely', 'ACCESSIBILITY_no_accessibility', 'ACCESSIBILITY_partially', 'PRICE_high', 'PRICE_low', 'PRICE_medium']]

        return recommendations_df
    
cf_recommender_model = CFRecommender(cf_preds_df, restaurants_df)

print(cf_recommender_model.recommend_items('U1100', verbose=True, topn=10)) 

print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model)
print('\nGlobal metrics:\n%s' % cf_global_metrics)
cf_detailed_results_df.head(10)
###############################

# HYBRID

class HybridRecommender:
    
    MODEL_NAME = 'Hybrid'
    
    def __init__(self, cb_rec_model, cf_rec_model, items_df):
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        #Getting the top-1000 Content-based filtering recommendations
        cb_recs_df = self.cb_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose,
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCB'})
        
        #Getting the top-1000 Collaborative filtering recommendations
        cf_recs_df = self.cf_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose, 
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCF'})
        
        #Combining the results by contentId
        recs_df = cb_recs_df.merge(cf_recs_df,
                                   how = 'inner', 
                                   left_on = 'placeID', 
                                   right_on = 'placeID')
        
        #Computing a hybrid recommendation score based on CF and CB scores
        recs_df['recStrengthHybrid'] = recs_df['recStrengthCB'] * recs_df['recStrengthCF']
        
        #Sorting recommendations by hybrid score
        recommendations_df = recs_df.sort_values('recStrengthHybrid', ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'placeID', 
                                                          right_on = 'placeID')[['recStrengthHybrid', 'placeID', 'latitude', 'longitude', 'name', 'city', 'state', 'PAYMENT_cash', 'PAYMENT_credit_card', 'PAYMENT_debit_card', 'PAYMENT_other', 'PARKING_no', 'PARKING_street', 'PARKING_yes', 'ALCOHOL_Full_Bar', 'ALCOHOL_No_Alcohol_Served', 'ALCOHOL_Wine-Beer', 'DRESS_CODE_casual', 'DRESS_CODE_formal', 'DRESS_CODE_informal', 'ACCESSIBILITY_completely', 'ACCESSIBILITY_no_accessibility', 'ACCESSIBILITY_partially', 'PRICE_high', 'PRICE_low', 'PRICE_medium']]

        return recommendations_df
    
hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, restaurants_df)

print(hybrid_recommender_model.recommend_items('U1100', verbose=True, topn=10)) 

print('Evaluating Hybrid model...')
hybrid_global_metrics, hybrid_detailed_results_df = model_evaluator.evaluate_model(hybrid_recommender_model)
print('\nGlobal metrics:\n%s' % hybrid_global_metrics)
hybrid_detailed_results_df.head(10)
##################################

# COMPARISON

global_metrics_df = pd.DataFrame([pop_global_metrics, cf_global_metrics, cb_global_metrics, hybrid_global_metrics]).set_index('modelName')

ax = global_metrics_df.transpose().plot(kind='bar', figsize=(10,4))
for p in ax.patches:
    ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show(block=True)