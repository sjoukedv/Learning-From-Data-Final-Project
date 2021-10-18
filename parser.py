import os, json
import numpy as np
from pathlib import Path

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
                "aricles": rawJson['articles']
            }])
    
    return data    
