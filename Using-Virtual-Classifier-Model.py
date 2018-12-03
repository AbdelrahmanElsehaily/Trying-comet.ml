
# coding: utf-8

# # Introducation

# This Notebook shows how to use the implemented model in this paper for author type classification (user, entity) using the following steps (all this steps should be applied after detecting gender using the existing gender enging):  
# * check if the first name in the new virtual lexicon if true then entity.
# * for the author bio model:
#     * clean the bio using the provided function.
#     * calculate the bio features.
#     * use the trained model to generate the probability Logistic Regression.
# 
# * for the author name model:
#     * clean the name using the provided function.
#     * calculate the name features.
#     * use the trained model to generate the probability using Logistic Regression.
# * after calculating the probabilities of the two models calculate the probability of the trained combined model
#    if the probability is greater than 0.635 then virtual else not covered.
#    
# ** The data used for training the different models is the interaction of different dayes of different month starting from Jan 2018 and test data is a stratified random sample from Sep 2017.**

# In[1]:

from comet_ml import Experiment

import pandas as pd
import numpy as np
import json
import cleaning_text
import re
import virtual_classifier_features 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support, recall_score, accuracy_score, confusion_matrix

experiment = Experiment(api_key="OlBgoYho0VQ74SQkV4R6rmOWh",
                        project_name="general", workspace="abdelrahmanelsehaily")
# In[2]:


#cleaned virtual lexicon
with open('Virtual-model-files/virtual_lexicon.txt', encoding='utf-8') as f:
    virtual_lexicon = set([x.strip('\n') for x in f.readlines()])

# cleaned stop words
with open('Virtual-model-files/stop_words.txt', encoding='utf-8') as f:
    stop_words = set([x.strip('\n') for x in f.readlines()])

## bio files
#the most frequent words in the bio for the entity class
with open('Virtual-model-files/Bio-Model/bio_entity_vocab.txt', encoding='utf-8') as f:
    bio_entity_vocab = set([x.strip('\n') for x in f.readlines()])
    
#the bio emojis with highest bayes probability of the entity class
with open('Virtual-model-files/Bio-Model/bio_entity_emojis.txt', encoding='utf-8') as f:
    bio_entity_emojis = set([x.strip('\n') for x in f.readlines()])    

#the bio emojis with highest bayes probability of the user class
with open('Virtual-model-files/Bio-Model/bio_user_emojis.txt', encoding='utf-8') as f:
    bio_user_emojis = set([x.strip('\n') for x in f.readlines()])        

#all the words in the author bio train data clusterd by bayes probability
with open('Virtual-model-files/Bio-Model/bio_entity_dictionary.json', encoding='utf-8') as f:
    bio_entity_dict = json.load(f)
    for key in bio_entity_dict:
        bio_entity_dict[key] = set(bio_entity_dict[key])
    

##name files
#the name emojis with highest bayes probability of the entity class 
with open('Virtual-model-files/Name-Model/name_entity_emojis.txt', encoding='utf-8') as f:
    name_entity_emojis = set([x.strip('\n') for x in f.readlines()])    

#the name emojis with highest bayes probability of the user class
with open('Virtual-model-files/Name-Model/name_user_emojis.txt', encoding='utf-8') as f:
    name_user_emojis = set([x.strip('\n') for x in f.readlines()])

#all the words in the author name train data clusterd by bayes probability
with open('Virtual-model-files/Name-Model/name_entity_dictionary.json', encoding='utf-8') as f:
    name_entity_dict = json.load(f)    


# # Test Data

# In[3]:


test_data = pd.read_excel('sep_pred_data_tested.xlsx')
#removin unannotated data
test_data = test_data[test_data['True_Gender'].isna()==False]
test_data.reset_index(drop=True,  inplace=True)

#adding new column represents the type instead of the gender
test_data['True_Type'] = 'user'
test_data.loc[test_data['True_Gender']=='entity', 'True_Type'] = 'entity'

test_data.loc[test_data['author.type']=='entity', 'author.gender'] = 'entity'
test_data['author.gender'].fillna('not-covered', inplace=True)
test_data['author.type'].fillna('not-covered', inplace=True)

#this column is the final prediction of the differnet stages
test_data['final_gender_pred'] = test_data['author.gender']

print('Accuracy of the engine: ', 
      accuracy_score(test_data[test_data['author.gender']!='not-covered']['True_Gender'],
               test_data[test_data['author.gender']!='not-covered']['author.gender']))

print('Coverage of the engine: ', 
      len(test_data[test_data['author.gender']==test_data['True_Gender']])/len(test_data)
     )

print('Precision and recall of entity: ', 
      precision_recall_fscore_support(test_data['True_Gender'],
               test_data['author.gender'], labels=['entity'])
     )

print('Precision and recall of male: ', 
      precision_recall_fscore_support(test_data['True_Gender'],
               test_data['author.gender'], labels=['male'])
     )

print('Precision and recall of female: ', 
      precision_recall_fscore_support(test_data['True_Gender'],
               test_data['author.gender'], labels=['female'])
     )


# # Data Cleaning

# In[4]:


#cleaning name
def add_clean_bio_and_name(data):
    data['author.name'].fillna('', inplace=True)
    _, data['cleaned_author_name_with_suffix'] = zip(*data['author.name'].map(
        lambda x: cleaning_text.clean_str(x, unify_mentions=False)))

    #cleaning bio
    data['author.bio'].fillna('', inplace=True)
    data['cleaned_author_bio'], data['cleaned_author_bio_with_suffix'] = zip(*data['author.bio'].map(cleaning_text.clean_str))
    
add_clean_bio_and_name(test_data)


# # MALE and FEMALE Detection

# In[5]:


#annotated male lexicon
with open('Virtual-model-files/males_names.txt', encoding='utf-8') as f:
    males_names = set([x.strip('\n') for x in f.readlines()])

#annotated female lexicon
with open('Virtual-model-files/females_names.txt', encoding='utf-8') as f:
    females_names = set([x.strip('\n') for x in f.readlines()])


# In[6]:


def detect_male(name):
    if name:
        first_name = name.split()[0]
        if re.match(r"\bعبد\S*|\bابو\S*|\bبو\S*|\babd\S*|\bابن\S*", first_name):
            return 1
        elif first_name in males_names:
            return 1
    return 0 

def detect_female(name):
    if name:
        if 'بنت' in name.split():
            return 1
        first_name = name.split()[0]
        if first_name =='ام':
            return 1
        elif first_name in females_names:
            return 1
    return 0


# In[7]:


test_data.loc[(test_data['cleaned_author_name_with_suffix'].apply(detect_male)&(
    test_data['author.gender']=='not-covered')), 'final_gender_pred'] = 'male'

test_data.loc[(test_data['cleaned_author_name_with_suffix'].apply(detect_female)&(
    test_data['author.gender']=='not-covered')), 'final_gender_pred'] = 'female'


# In[8]:


precision_recall_fscore_support(test_data['True_Gender'], test_data['final_gender_pred'], labels=['male', 'female'])


# # Virtual Detection

# ## Name Model

# In[9]:


def name_featurs(data):
    data['name_first_virtual'] = data['cleaned_author_name_with_suffix'].apply(
        lambda x: 1 if (x != '') and (x.split()[0] in virtual_lexicon) else 0)

    # language of the bio if contains any arabic alphapet -> arabic else other
    data['name_arabic_language'], _ = zip( 
        *data['author.bio'].map(virtual_classifier_features.detect_language))

    # counts of words in the cleaned name
    data['name_word_count'] = data['cleaned_author_name_with_suffix'].apply(
        lambda x: len(x.split())).astype('uint8')


    # count of characters in the cleaned name
    data['name_char_count'] = data['cleaned_author_name_with_suffix'].apply(
        lambda x: len(x)).astype('uint8')

    # emojis in the raw name
    data['emojis_in_name'] = data['author.name'].apply(virtual_classifier_features.extract_emojis)
    data['name_emojis_count'] = data['emojis_in_name'].apply(
        lambda x: len(x)).astype('uint16')

    # count of the virtual words in the bio using a virtual lexicon
    data['name_virtual_count'] = data['cleaned_author_name_with_suffix'].apply(
        lambda x: virtual_classifier_features.count_virtual_words(x, virtual_lexicon)).astype('uint8')

    # count of plural words using suffixes and the plural stemmer
    data['name_plural_count'] = data['cleaned_author_name_with_suffix'].apply(
        virtual_classifier_features.count_plural_words).astype('uint8')

    # count of stop words in the raw name
    data['name_stop_word_count'] = data['author.name'].apply(
        lambda x: len([word for word in x.split() if word in stop_words]))

    # count of words starts with crtain prefixes
    data['name_prefix_count'] = data['author.name'].apply(
        lambda x: len([word for word in x.split() if len(word) > 4 and
            re.match(r"\bبال\S*|\bكال\S*|\bبال\S*|\bلل\S*|\bولل\S*", word)]))

    #count of the emojis with highest log bayes probability given entity class
    data['name_entity_emojis_count'] = data['emojis_in_name'].apply(
        lambda x: len([emoji for emoji in x if emoji in name_entity_emojis] )).astype('uint8')

    #count of the emojis with highest log bayes probability given  user class
    data['name_user_emojis_count'] = data['emojis_in_name'].apply(
        lambda x: len([emoji for emoji in x if emoji in name_user_emojis] )).astype('uint8')
    
    for key in bio_entity_dict:
        data['name_entity'+key] = data['cleaned_author_name_with_suffix'].apply(
            lambda x: 1 if len([word for word in x.split() if word in name_entity_dict[key]])>0 else 0)
    return data

test_data = name_featurs(test_data)


# In[10]:


name_model_coef_divisor = pd.read_excel('Virtual-model-files/Name-Model/name_model_coefficients_divisor.xlsx')
name_model_coef_divisor.set_index('feature', inplace=True)
test_data['intercept'] = 1
test_data['name_pred_prob'] = 1-(1/(1+np.exp(-1*np.sum(
    (test_data[list(name_model_coef_divisor.index)].values
    /name_model_coef_divisor['divisor'].values)*name_model_coef_divisor['coef'].values, axis=1))))
test_data['pred_name'] = ['entity' if x>=0.5 else 'user' for x in test_data['name_pred_prob'].values]
precision_recall_fscore_support(test_data['True_Type'], test_data['pred_name'])


# ## classifying using First Name

# In[11]:


test_data.loc[(test_data['author.type']!='user')&(
    (test_data['name_first_virtual']==1)|(
        test_data['author.type']=='entity')), 'final_gender_pred'] = 'entity'

test_data['stemmed_virtual_pred'] = 'user'
test_data.loc[
    (test_data['name_first_virtual']==1), 'stemmed_virtual_pred'] = 'entity'

test_data[(test_data['author.gender']=='not-covered')&(test_data['final_gender_pred']!='not-covered')].shape


# ## Bio Model

# In[14]:


def bio_features(data):
    #language of the bio if contains any arabic alphapet then 'arabic' else then 'other'
    data['bio_arabic_language'], _ = zip(*data['author.bio'].map(
        virtual_classifier_features.detect_language))

    # word count of the raw bio split on space
    data['bio_word_count'] = data['cleaned_author_bio'].apply(lambda x: len(x.split())).astype('uint16')

    #extracting emojis in the raw bio
    data['emojis_in_bio'] = data['author.bio'].apply(virtual_classifier_features.extract_emojis)
    data['bio_emojis_count'] = data['emojis_in_bio'].apply(
        lambda x: len(x)).astype('uint16')

    #check if the author name in the bio
    data['bio_contains_username'] = data.apply(virtual_classifier_features.check_username_in_bio, axis=1).astype('uint8')

    # count of the virtual words in the bio using a virtual lexicon
    data['bio_virtual_count'] = data['cleaned_author_bio_with_suffix'].apply(lambda x:
        virtual_classifier_features.count_virtual_words(x,virtual_lexicon)).astype('uint8')

    # count of plural words using suffixes and the plural stemmer
    data['bio_plural_count'] = data['cleaned_author_bio_with_suffix'].apply(
        virtual_classifier_features.count_plural_words).astype('uint8')

    #check if the the first word of the bio is in the virtual lexicon
    data['bio_starts_with_virtual'] = data['cleaned_author_bio_with_suffix'].apply(
        lambda x: 1 if x and x.split()[0] in virtual_lexicon else 0)

    #count of the emojis with highest log bayes probability given entity class
    data['bio_entity_emojis_count'] = data['emojis_in_bio'].apply(
        lambda x: len([emoji for emoji in x if emoji in bio_entity_emojis])).astype('uint8')

    #count of the emojis with highest log bayes probability given user class
    data['bio_user_emojis_count'] = data['emojis_in_bio'].apply(
        lambda x: len([emoji for emoji in x if emoji in bio_user_emojis])).astype('uint8')
    
    for key in bio_entity_dict:
        data['bio_entity'+key] = data['cleaned_author_bio'].apply(
            lambda x: len([word for word in x.split() if word in bio_entity_dict[key]]))
        
    #flag feautre for every word in the most frequent words in the bio
    bio_vec = CountVectorizer(ngram_range=[1,1], binary=True, vocabulary=bio_entity_vocab).fit(
        data['cleaned_author_bio'])
    count_vect_df = pd.DataFrame(bio_vec.transform(
        data['cleaned_author_bio']).todense(), columns=bio_vec.get_feature_names())
    data = pd.concat([data, count_vect_df], axis=1)
    return data

test_data = bio_features(test_data)


# In[15]:


bio_model_coef_divisor = pd.read_excel('Virtual-model-files/Bio-Model/bio_model_coefficients_divisor.xlsx')
bio_model_coef_divisor.set_index('feature', inplace=True)
test_data['intercept'] = 1
test_data['bio_pred_prob'] = 1-(1/(1+np.exp(-1*np.sum(
    (test_data[list(bio_model_coef_divisor.index)].values
    /bio_model_coef_divisor['divisor'].values)*bio_model_coef_divisor['coef'].values, axis=1))))
test_data['pred_bio'] = ['entity' if x>=0.5 else 'user' for x in test_data['bio_pred_prob'].values]
precision_recall_fscore_support(test_data['True_Type'], test_data['pred_bio'])


# ## Combined Model

# In[16]:


#the trained model on the name and bio models probabilities
with open('Virtual-model-files/combined_model.json', encoding='utf-8') as f:
    combined_model = json.load(f)    


# In[17]:


test_data['combined_pred_prob'] = 1-(1/(1+np.exp(-1*(
            combined_model['intercept']
            +(test_data['name_pred_prob']*combined_model['name_coef'])
            +(test_data['bio_pred_prob']*combined_model['bio_coef'])
        )
    )
))

test_data['combined_pred'] = ['entity' if x>=0.4 else 'user' for x in test_data['combined_pred_prob']]


# # Final Results

# In[18]:


test_data.loc[
    (test_data['combined_pred_prob']>=0.635)&(test_data['final_gender_pred']=='not-covered'), 'final_gender_pred'] = 'entity'


# In[19]:
metrics = {
"precision": accuracy_score(test_data[test_data['final_gender_pred']!='not-covered']['True_Gender'],
                                      test_data[test_data['final_gender_pred']!='not-covered']['final_gender_pred'])
#'recall':accuracy_score(test_data[test_data['final_gender_pred']!='not-covered']['True_Gender'],
 #                                     test_data[test_data['final_gender_pred']!='not-covered']['final_gender_pred'])
}
experiment.log_multiple_metrics(metrics)


print('Accuracy after aplying all stages', 
      accuracy_score(test_data[test_data['final_gender_pred']!='not-covered']['True_Gender'],
                                      test_data[test_data['final_gender_pred']!='not-covered']['final_gender_pred']))

print('Coverage after applying all stages: ', 
      len(test_data[test_data['final_gender_pred']==test_data['True_Gender']])/len(test_data)
     )

