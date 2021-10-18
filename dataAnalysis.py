from dataParser import read, mergeCopEditions
import matplotlib.pyplot as plt

# debug to run only this file
if __name__ == "__main__":
    data = read()

    # print(data[0]['metadata'])
    # print('-------------------------')
    # print(data[0]['articles'][0]['newspaper'])
    # print(data[0]['articles'][0]['headline'])
    # print(data[0]['articles'][0]['classification'])
    # print('-------------------------')
    # print(data[0]['orientations'][0])

    articles, orientations = mergeCopEditions(data)
    # for i in range(1,30):
    #     print(articles[i]['newspaper'])
    #     print(orientations[i])

    plt.hist(orientations)
    plt.show()
