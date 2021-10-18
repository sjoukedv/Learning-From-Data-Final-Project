import os, json, random
import numpy as np
from pathlib import Path

np.random.seed(19680801)

def getPoliticalOrientation(newspaper):
    if ("The Australian" in newspaper) or ("The Times of India" in newspaper) or ("The Times" in newspaper):
        return "right-center"
    elif ("Sydney Morning Herald" in newspaper) or ("The Age" in newspaper) or ("The Hindu" in newspaper) or ("Mail and Guardian" in newspaper) or ("The Washington Post" in newspaper) or ("New York Times" in newspaper):
        return "left-center"
    else:
        # randomly pick political orientation if it is unknown
        #   pick accordingly to the distribution in the data
        #   9148 / 23474 = 0.3897077
        #   results in 9106 (right-center) and 23516 (left-center)
        if random.random() < 0.3897077:
            return "right-center"
        else:
            return "left-center"
        
def parsePoliticalOrientation(articles):
    orientations = np.array([])
    for article in articles:
        newspaper = article['newspaper']
        orientations = np.append(orientations, getPoliticalOrientation(newspaper))
    
    return orientations

def read():
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
                "articles": rawJson['articles'],
                "orientations": parsePoliticalOrientation(rawJson['articles'])
            }])
    
    return data    

def mergeCopEditions(data):
    articles = []
    orientations = []
    for elem in data:
        articles = articles + elem['articles']
        orientations = np.append(orientations, elem['orientations'])

    return articles, orientations
