from dataParser import read, mergeCopEditions
import matplotlib.pyplot as plt
import numpy as np
import sys

    
def histogramPoliticalOrientation(orientations):
    fig, ax = plt.subplots()
    ax.set_ylabel('Number of articles')   
    ax.set_title('Articles by political orientation')
    plt.hist(orientations)
    fig.savefig('figures/hist_political_orientation.png')
    
def barChartNewspaper(articles):
    # 'Mail & Guardian' 'Sydney Morning Herald (Australia)'
    #  'The Age (Melbourne, Australia)' 'The Australian' 'The Hindu'
    #  'The New York Times' 'The Times (South Africa)'
    #  'The Times of India (TOI)' 'The Washington Post'

    tuples = [ (article['newspaper'], article['political_orientation']) for article in articles]
    sorted_newspapers = np.unique(tuples, axis=0, return_counts=True)

    labels_left = [ newspaper[0] for (newspaper, count) in zip(sorted_newspapers[0], sorted_newspapers[1]) if newspaper[1] == 'left-center']
    values_left = [ count for (newspaper, count) in zip(sorted_newspapers[0], sorted_newspapers[1]) if newspaper[1] == 'left-center']
    labels_right = [ newspaper[0] for (newspaper, count) in zip(sorted_newspapers[0], sorted_newspapers[1]) if newspaper[1] == 'right-center']
    values_right = [ count for (newspaper, count) in zip(sorted_newspapers[0], sorted_newspapers[1]) if newspaper[1] == 'right-center']

    fig, ax = plt.subplots()
    plt.xticks(rotation=30, ha='right')
    ax.bar(labels_left, values_left, 0.35, label='left-center')
    ax.bar(labels_right, values_right, 0.35, label='right-center')

    ax.set_ylabel('Number of articles')
    ax.set_title('Articles by political orientation per newspaper')
    ax.legend()
    fig.savefig('figures/newspaper_bar_chart', bbox_inches="tight")

# debug to run only this file
if __name__ == "__main__":
    data = read()
    articles = mergeCopEditions(data)

    histogramPoliticalOrientation([article['political_orientation'] for article in articles])
    barChartNewspaper(articles)
