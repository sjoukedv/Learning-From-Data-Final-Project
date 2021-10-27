from dataParser import read_articles
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import sys

    
def histogramPoliticalOrientation(orientations):
    fig, ax = plt.subplots()
    ax.set_ylabel('Number of articles')   
    ax.set_title('Articles by political orientation')
    plt.hist(orientations)
    fig.savefig('figures/hist_political_orientation.png')

def barChartSet(Y_train, Y_dev, Y_test):
    train_cnt = np.unique(Y_train, return_counts=True)
    dev_cnt = np.unique(Y_dev, return_counts=True)
    test_cnt = np.unique(Y_test, return_counts=True)
    
    left = [train_cnt[1][0], dev_cnt[1][0], test_cnt[1][0]]
    right = [train_cnt[1][1], dev_cnt[1][1], test_cnt[1][1]]


    X = np.arange(3)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(X + 0.00, left, color = 'b', width = 0.25, label='left-center')
    ax.bar(X + 0.25,right, color = 'g', width = 0.25, label='right-center')
    ax.set_ylabel('Number of articles')
    ax.set_title('Distribution per test set')
    ax.legend()
    fig.savefig('figures/sets_bar_chart', bbox_inches="tight")

def mostOccuringWords(headlines):
    # most k words
    k = 20

    # Pass the split_it list to instance of Counter class.
    counter = Counter(headlines)
  
    # most_common() produces k frequently encountered
    # input values and their respective counts.
    x = [ tag[:25] if len(tag) > 25 else tag for tag,count in counter.most_common(k)] 
    y = [ count for tag,count in counter.most_common(k)] 

    fig, ax = plt.subplots()
    ax.bar(x, y, color='crimson')
    plt.xticks(rotation=90)
    plt.xlim(-0.6, len(x)-0.4) # optionally set tighter x lims
    ax.set_ylabel('Number of occurences')
    ax.set_title('Word occurence')
    fig.savefig('figures/most_occuring_words', bbox_inches="tight")

def barChartNewspaper(articles):
    # 'Mail & Guardian' 'Sydney Morning Herald (Australia)'
    #  'The Age (Melbourne, Australia)' 'The Australian' 'The Hindu'
    #  'The New York Times' 'The Times (South Africa)'
    #  'The Times of India (TOI)' 'The Washington Post'

    tuples = [ (article[0], article['political_orientation']) for article in articles]
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
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = read_articles() 

    histogramPoliticalOrientation(Y_train + Y_dev + Y_test)
    barChartSet(Y_train, Y_dev, Y_test)
    mostOccuringWords(X_train + X_dev + X_test)
    # barChartNewspaper(zip(X_train + X_dev + X_test, Y_train + Y_dev + Y_test))
