import os, json, random
import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split

np.random.seed(19680801)

def getPoliticalOrientation(newspaper):
    if ("The Australian" in newspaper) or ("The Times of India" in newspaper) or ("The Times" in newspaper):
        return "right-center"
    elif ("Sydney Morning Herald" in newspaper) or ("The Age" in newspaper) or ("The Hindu" in newspaper) or ("Mail & Guardian" in newspaper) or ("The Washington Post" in newspaper) or ("New York Times" in newspaper):
        return "left-center"
    else:
        # randomly pick political orientation if it is unknown
        #   does not occur in this dataset
        #   pick accordingly to the distribution in the data
        #   9148 / 23474 = 0.3897077
        #   results in 9106 (right-center) and 23516 (left-center)
        if random.random() < 0.3897077:
            return "right-center"
        else:
            return "left-center"
        
def parsePoliticalOrientation(articles):
    result = np.array([])
    for article in articles:
        newArticle = article
        newArticle['political_orientation'] = getPoliticalOrientation(article['newspaper'])
        result = np.append(result, newArticle)
    return result

def read_data():
    files = Path("data").glob("**/*.json")

    data = np.array([])

    for file in files:
        with open(file) as json_file:
          
            rawJson = json.load(json_file)
            data = np.append(data,[{
                "metadata": {
                    'cop_edition': rawJson['cop_edition'],
                    'collection_start' : rawJson['collection_start'],
                    'collection_end' : rawJson['collection_end']
                },
                "articles": parsePoliticalOrientation(rawJson['articles'])
            }])
    
    return data    

def read_articles(train_percentage=0.8, test_percentage=0.1, dev_percentage=0.1, randomise=True):
    if train_percentage + test_percentage + dev_percentage != 1.0:
        print('Split does not add to 1')
        sys.exit(-1)
    raw_data = read_data()

    articles = []
    for cop_edition in raw_data:
        articles = np.append(articles, cop_edition['articles'])

    # split train/test
    train, test = train_test_split(articles, train_size=train_percentage, test_size=1-train_percentage, shuffle=randomise, random_state=19680801)

    # split test into dev/test
    dev, test = train_test_split(articles, train_size=test_percentage, test_size=dev_percentage, shuffle=randomise, random_state=19680801)

    return [[ article['headline'] for article in train ], 
    [ article['political_orientation'] for article in train ],
    [ article['headline'] for article in dev ], 
    [ article['political_orientation'] for article in dev ],
    [ article['headline'] for article in test ], 
    [ article['political_orientation'] for article in test ]]
