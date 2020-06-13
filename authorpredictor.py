import numpy as np
#import sklearn
import nltk
import math
#import sklearn.feature_extraction.text
from nltk.stem import WordNetLemmatizer

'''
Author Predictor

Bayes Multinomial With Bag Of Words for Multi Class
Using One vs Rest
Bigrams and Trigrams
Lemmetization w/ nltk library
'''

TRIGRAM = 300  # weight of trigram
BIGRAM = 100  # weight of bigram


def bayesTrain(training_data, training_labels, fNumPosts):
    '''
    Training bag of words w/ bigrams and trigrams
    '''
    featDictArr = []
    priorProbDict = {}
    numPosWordsDic = {}
    for i in range(len(training_labels)):
        featDict = {}
        numPosWords = 0
        featDict[training_labels[i]] = 0
        wlist = training_data[i].split()
        len2 = len(wlist)
        for j in range(len2):
            if wlist[j] in featDict:  # UniGram
                featDict[wlist[j]] += 1
                numPosWords += 1
            else:
                featDict[wlist[j]] = 1
                numPosWords += 1
            if(j+1 < len2):  # BiGram
                w = wlist[j]+wlist[j+1]
                if w in featDict:
                    featDict[w] += BIGRAM
                    numPosWords += 1
                else:
                    featDict[w] = BIGRAM
                    numPosWords += 1
            if(j+2 < len2):  # TriGram
                w = wlist[j]+wlist[j+1]+wlist[j+2]
                if w in featDict:
                    featDict[w] += TRIGRAM
                    numPosWords += 1
                else:
                    featDict[w] = TRIGRAM
                    numPosWords += 1
        numPosWordsDic[training_labels[i]] = numPosWords
        featDictArr.append(featDict)
    totalPosts = 0
    for key in training_labels:
        totalPosts += fNumPosts[key]
        priorProbDict[key] = math.log2(fNumPosts[key]/totalPosts)
    return featDictArr, priorProbDict, numPosWordsDic, totalPosts


def bayesTest(testing_data, featDictArr, priorProbDict, numPosWordsDic, totalPosts, featNameArr):
    '''
    tests dataset and returns array of predicted names
    '''
    solArray = []
    for i in range(len(testing_data)):
        words = testing_data[i].split()
        wdlen = len(words)
        for i in range(len(words)):
            if (i+1 < wdlen):  # adding bigram
                words.append(words[i]+words[i+1])
            if (i+2 < wdlen):  # adding trigram
                words.append(words[i]+words[i+1]+words[i+2])
        highestProb = -600000
        bestFeature = ""
        for feat in range(len(featDictArr)):
            probPosGivenPos = 0
            for word in words:
                if word in featDictArr[feat]:
                    probPosGivenPos += math.log2((featDictArr[feat][word]+1)/(
                        numPosWordsDic[featNameArr[feat]]+len(featDictArr[feat])))
                else:
                    probPosGivenPos += math.log2(
                        1/(numPosWordsDic[featNameArr[feat]]+len(featDictArr[feat])))
            probPosGivenPos += priorProbDict[featNameArr[feat]]
            if(probPosGivenPos > highestProb):
                highestProb = probPosGivenPos
                bestFeature = featNameArr[feat]
        solArray.append(bestFeature)

    return solArray


def run_train_test(training_data, training_labels, testing_data):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: List[string]
        training_label: List[string]
        testing_data: List[string]
    Output:
        testing_prediction: List[string]
    Example output:
    return ['NickLouth']*len(testing_data)
    """
    # step 1 format the data
    lmtzr = WordNetLemmatizer()
    stopWords = []
    #stopWords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until","while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]  # testing
    len1 = len(training_data)
    # formatting training data
    for i in range(len1):
        temp_string = ""
        training_data[i] = training_data[i].lower()
        training_data[i] = training_data[i].replace(",", " , ").replace(
            ".", " ").replace(";", " ; ").replace(":", " : ").replace("\n", " ")
        for word in training_data[i].split(" "):
            if(word not in stopWords):
                if(word != "," or word != "." or word != ":"):
                    temp_string += word+" "
                    temp_string += lmtzr.lemmatize(word)+" "
                else:
                    temp_string += word+" "
        training_data[i] = temp_string
    # formatting testing data
    for i in range(len(testing_data)):
        temp_string = ""
        testing_data[i] = testing_data[i].lower()
        testing_data[i] = testing_data[i].replace(",", " , ").replace(
            ".", " ").replace(";", " ; ").replace(":", " : ").replace("\n", " ")
        for word in testing_data[i].split(" "):
            if(word not in stopWords):
                if(word != "," or word != "." or word != ":"):
                    temp_string += word+" "
                    temp_string += lmtzr.lemmatize(word)+" "
                else:
                    temp_string += word+" "
        testing_data[i] = temp_string

    # grouping together all training data sentences with same writer
    # more time intensive way to do this but did so because wasn't sure whether I was going SVM or Bayes route initially
    fDic = {}
    fNumPost = {}
    for i in range(len(training_labels)):
        if training_labels[i] in fDic:
            fDic[training_labels[i]] += training_data[i]+" "
            fNumPost[training_labels[i]] += 1
        else:
            fDic[training_labels[i]] = training_data[i]+" "
            fNumPost[training_labels[i]] = 1
    # making into new more palatable array to place into my bayes model to train
    training_data_new = []
    training_labels_new = []
    for key, value in fDic.items(): 
        traindatatmp = value
        trainlabeltmp = key
        training_data_new.append(traindatatmp)
        training_labels_new.append(trainlabeltmp)
    # train the model
    featDictArr, priorProbDict, numPosWordsDic, totalPosts = bayesTrain(
        training_data_new, training_labels_new, fNumPost)
    # test the model on the training dataset and return solution array
    sol = bayesTest(testing_data, featDictArr, priorProbDict,
                    numPosWordsDic, totalPosts, training_labels_new)
    # print(sol)
    return sol
