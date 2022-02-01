# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import nltk
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
# list of stopwords(str)
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
 'by', 'for','before', 'after', 'to',
 'from', 'up', 'down', 'in', 'out', 'on', 'off',
 'again', 'further', 'then', 'once', 'here', 'there', 'when',
 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
 't', 'can', 'will', 'just', 'don', "don't", 'should',
 "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',
 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
 "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
 "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
 'wouldn', "wouldn't"]

def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels



# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")

"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.26, pos_prior=0.8,silently=False):
    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)


    #print(dev_set[0])
    dict_pos_words = {}   # dictionary of words in a positive training set with the # of occurences
    dict_neg_words = {}   # dictionary of words in a negative training set with the # of occurences
    dict_all_words = {}   # dict to count all words, both from positive and negative data sets
    index = 0

    #iterate through each review in the train_set
    for review in train_set:
        if train_labels[index] == 1: # for positive reviews
            for word in review:
                if word not in stopwords:
                    word = word.lower()

                    if word not in dict_pos_words:
                        dict_pos_words[word] = 1     # add new word to dict of pos words 
                        dict_all_words[word] = 1     # add word to dict of all UNIQUE words in training set
                    else:
                        dict_pos_words[word] += 1    # increment count of existing words    
                else:
                    continue  # if the word is a stopword, skip current word
        else: # for negative reviews
            for word in review:
                if word not in stopwords:
                    word = word.lower()

                    if word not in dict_neg_words:
                        dict_neg_words[word] = 1
                        dict_all_words[word] = 1
                    else:
                        dict_neg_words[word] += 1     
                else:
                    continue
        index += 1

    total_pos_words = sum(dict_pos_words.values())   # sum of all the non-stopword, non-unique, words from positive reviews
    total_neg_words = sum(dict_neg_words.values())   # sum of all the non-stopword, non-unique, words from negative reviews
    n = total_neg_words + total_pos_words # number of words in training data set (both pos and neg)
    V_pos = len(dict_pos_words.keys())  #number of word types or unique words seen in positive training set
    V_neg = len(dict_neg_words.keys())  #number of word types or unique words seen in negative training set

    pos_prob = {}   # P(Word|Positive)
    neg_prob = {}   # P(Word|Negative)

    prob_unknown_pos = laplace/(total_pos_words + laplace*(V_pos+1))
    prob_unknown_neg = laplace/(total_neg_words + laplace*(V_neg+1))    

    pos_words_list = [dict_pos_words.keys()]  # list of all unique words from positive reviews
    pos_item_list = [dict_pos_words.items()]  # list of number of occurences of each word from pos reviews
    neg_words_list = [dict_neg_words.keys()]  # list of all unique words from negative reviews
    neg_item_list = [dict_neg_words.items()]  # list of number of occurences of each word from neg reviews

    #get the probability of a word in a positive review:
    for key in dict_pos_words:
    # P(Word|Positive)   count(Word) in Positive    alpha         n             alpha*(V+1)
        pos_prob[key] = (dict_pos_words.get(key) + laplace)/(total_pos_words + laplace*(V_pos+1))  #P(Word|Positive)

    for key in dict_neg_words:
    # P(Word|Negative)   count(Word) in Negative    alpha         n             alpha*(V+1)
        neg_prob[key] = (dict_neg_words.get(key) + laplace)/(total_neg_words + laplace*(V_neg+1)) # P(Word|Negative)

    #print(dev_set[0])
    total_pos_prob = sum(pos_prob.values())   # sum of all probabilities of each P(Word|Positive)
    total_neg_prob = sum(neg_prob.values())   # sum of all probabilities of each P(Word|Negative) 

    yhats = []

    #print(pos_prob.get('spolier'))
    for review in dev_set:
        sum_word_pos = []  #sum of log10(P(Word|Positive))
        sum_word_neg = []  #sum of log10(P(Word|Negative))        
        #calculate the prob of a review being positive given the words in it
        for word in review:   # access each word token in every review 
            if word not in stopwords:
                word = word.lower()
                if pos_prob.get(word) is None:  #if word is not in training set
                    sum_word_pos.append(np.log10(prob_unknown_pos)) # add prob_unknown to list of P(Word|Type=Positive)
                else:
                    sum_word_pos.append(np.log10(pos_prob[word])) # add P(Word|Type=Positive) to list of P(Word|Type=Positive)

                if neg_prob.get(word) is None:  #if word is not in training set
                    sum_word_neg.append(np.log10(prob_unknown_neg))
                else:
                    sum_word_neg.append(np.log10(neg_prob[word]))
            else:
                continue
    # P(Positive|Words)   log10(P(Positive))  sum of log10(P(Word|Positive)) 
        prob_type_pos = np.log10(pos_prior) + sum(sum_word_pos)
    # P(Negative|Words)   log10(P(Negative))  sum of log10(P(Word|Negative)) 
        prob_type_neg = np.log10(1-pos_prior) + sum(sum_word_neg)


        #print(sum(sum_word_pos))
        if (prob_type_pos) > (prob_type_neg):  # compare which one has greater prob
            yhats.append(1)
            continue
        else:
            yhats.append(0)
            continue

    #for doc in tqdm(dev_set,disable=silently):
    #    yhats.append(-1)
    return yhats


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")

# main function for the bigrammixture model
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.004, bigram_laplace=0.007, bigram_lambda=0.4,pos_prior=0.5, silently=False):
    #unigram_laplace=0.5
    #bigram_laplace=0.15
    #bigram_lambda=0.15
    #pos_prior=0.5

    print_paramter_vals_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior)
    # UNIGRAM PART
    #print(train_labels[5990:6010])
    dict_pos_words_uni = {}   # dictionary of words in a positive training set with the # of occurences
    dict_neg_words_uni = {}   # dictionary of words in a negative training set with the # of occurences
    dict_all_words_uni = {}   # dict to count all words, both from positive and negative data sets
    index = 0

    #iterate through each review in the train_set
    for review in train_set:
        if train_labels[index] == 1: # for positive reviews
            for word in review:
                if word not in stopwords:
                    word = word.lower()

                    if word not in dict_pos_words_uni:
                        dict_pos_words_uni[word] = 1     # add new word to dict of pos words 
                        dict_all_words_uni[word] = 1     # add word to dict of all UNIQUE words in training set
                    else:
                        dict_pos_words_uni[word] += 1    # increment count of existing words    
                else:
                    continue  # if the word is a stopword, skip current word
        else: # for negative reviews
            for word in review:
                if word not in stopwords:
                    word = word.lower()

                    if word not in dict_neg_words_uni:
                        dict_neg_words_uni[word] = 1
                        dict_all_words_uni[word] = 1
                    else:
                        dict_neg_words_uni[word] += 1     
                else:
                    continue
        index += 1

    total_pos_words_uni = sum(dict_pos_words_uni.values())   # sum of all the non-stopword, non-unique, words from positive reviews
    total_neg_words_uni = sum(dict_neg_words_uni.values())   # sum of all the non-stopword, non-unique, words from negative reviews
    n_uni = total_neg_words_uni + total_pos_words_uni # number of words in training data set (both pos and neg)
    V_uni_pos = len(dict_pos_words_uni.keys())  #number of word types or unique words seen in positive training set
    V_uni_neg = len(dict_neg_words_uni.keys())  #number of word types or unique words seen in negative training set



    pos_prob_uni = {}   # P(Word|Positive)
    neg_prob_uni = {}   # P(Word|Negative)

    prob_unknown_uni_pos = unigram_laplace/(total_pos_words_uni + (unigram_laplace*(V_uni_pos+1)))
    prob_unknown_uni_neg = unigram_laplace/(total_neg_words_uni + (unigram_laplace*(V_uni_neg+1)))

    #get the probability of a word in a positive review:
    for key in dict_pos_words_uni:
        # P(Word|Positive)   count(Word) in Positive             alpha                n                 alpha*(V+1)
        pos_prob_uni[key] = (dict_pos_words_uni.get(key) + unigram_laplace)/(total_pos_words_uni + unigram_laplace*(V_uni_pos+1))  #P(Word|Positive)

    for key in dict_neg_words_uni:
        # P(Word|Negative)      count(Word) in Negative          alpha                n                 alpha*(V+1)
        neg_prob_uni[key] = (dict_neg_words_uni.get(key) + unigram_laplace)/(total_neg_words_uni + unigram_laplace*(V_uni_neg+1)) # P(Word|Negative)

    total_pos_prob = sum(pos_prob_uni.values())   # sum of all probabilities of each P(Word|Positive)
    total_neg_prob = sum(neg_prob_uni.values())   # sum of all probabilities of each P(Word|Negative) 


    # BIGRAM PART
    #print(dev_set[0])
    dict_pos_words_bi = {}   # dictionary of pair of words in a positive training set with the # of occurences
    dict_neg_words_bi = {}   # dictionary of pair words in a negative training set with the # of occurences
    dict_all_words_bi = {}   # dict to count all pairs of words, both from positive and negative data sets
    index1 = 0

    #iterate through each review in the train_set
    for review in train_set:
        if train_labels[index1] == 1: # for positive reviews
            reviewset_length = len(review)-1
            for index in range(0, reviewset_length):
                next = index+1
                pair = (review[index], review[next])  # ('word1', 'word2')
               # print(pair)   

                if pair not in dict_pos_words_bi:
                    dict_pos_words_bi[pair] = 1     # add new pair of word to dict of pairs of pos words 
                    dict_all_words_bi[pair] = 1     # add pair of word to dict of all UNIQUE pairs of words in training set
                else:
                    dict_pos_words_bi[pair] += 1    # increment count of existing pos pairs of words    
                
                index += 1
        else: # for negative reviews
            reviewset_length = len(review)-1
            for index in range(0,reviewset_length):
                next = index+1
                pair = (review[index], review[next])

                if pair not in dict_neg_words_bi:
                    dict_neg_words_bi[pair] = 1     # add new pair of word to dict of pairs of neg words 
                    dict_all_words_bi[pair] = 1     # add pair of word to dict of all UNIQUE pairs of words in training set
                else:
                    dict_neg_words_bi[pair] += 1    # increment count of existing neg pairs of words    
                
                index += 1   
        index1 += 1

    #print(len(dict_neg_words_bi), len(dict_pos_words_bi))

    total_pos_words_bi = sum(dict_pos_words_bi.values())   # sum of all the non-stopword, non-unique, words from positive reviews
    total_neg_words_bi = sum(dict_neg_words_bi.values())   # sum of all the non-stopword, non-unique, words from negative reviews
    n_bi = total_neg_words_bi + total_pos_words_bi # number of words in training data set (both pos and neg)
    V_bi_pos = sum(dict_all_words_bi.values())  #number of word types or unique words seen in positive training set
    V_bi_neg = sum(dict_all_words_bi.values())  #number of word types or unique words seen in negative training set

    pos_prob_bi = {}   # P(Word|Positive)
    neg_prob_bi = {}   # P(Word|Negative)

    prob_unknown_bi_pos = bigram_laplace/(total_pos_words_bi + (bigram_laplace*(V_bi_pos+1)))
    prob_unknown_bi_neg = bigram_laplace/(total_neg_words_bi + (bigram_laplace*(V_bi_neg+1)))

    #get the probability of a pair of words in a positive review:
    for key in dict_pos_words_bi:
    # P(Word|Positive)   count(Word) in Positive    alpha         n             alpha*(V+1)
        pos_prob_bi[key] = (dict_pos_words_bi.get(key) + bigram_laplace)/(total_pos_words_bi + bigram_laplace*(V_bi_pos+1))  #P(Word Pair|Positive)

    for key in dict_neg_words_bi:
    # P(Word|Negative)   count(Word) in Negative    alpha         n             alpha*(V+1)
        neg_prob_bi[key] = (dict_neg_words_bi.get(key) + bigram_laplace)/(total_neg_words_bi + bigram_laplace*(V_bi_neg+1)) # P(Word Pair|Negative)

    #print(dev_set[0])
    total_pos_prob_bi = sum(pos_prob_bi.values())   # sum of all probabilities of each P(Word Pair|Positive)
    total_neg_prob_bi = sum(neg_prob_bi.values())   # sum of all probabilities of each P(Word Pair|Negative) 

    yhats = []

    #print(pos_prob.get('spolier'))
    for review in dev_set:

        # UNIGRAM PART OF DEV SET
        sum_word_pos_uni = []  #sum of log10(P(Word|Positive))
        sum_word_neg_uni = []  #sum of log10(P(Word|Negative))        
        #calculate the prob of a review being positive given the words in it
        for word in review:   # access each word token in every review 
            if word not in stopwords:
                word = word.lower()
                if pos_prob_uni.get(word) is None:  #if word is not in training set
                    sum_word_pos_uni.append(np.log10(prob_unknown_uni_pos)) # add prob_unknown to list of P(Word|Type=Positive)
                else:
                    sum_word_pos_uni.append(np.log10(pos_prob_uni[word])) # add P(Word|Type=Positive) to list of P(Word|Type=Positive)

                if neg_prob_uni.get(word) is None:  #if word is not in training set
                    sum_word_neg_uni.append(np.log10(prob_unknown_uni_neg))
                else:
                    sum_word_neg_uni.append(np.log10(neg_prob_uni[word]))
            else:
                continue 
        prob_type_pos_uni = np.log10(pos_prior) + sum(sum_word_pos_uni)
        prob_type_neg_uni = np.log10(1-pos_prior) + sum(sum_word_neg_uni)  

        # BIGRAM PART OF DEV SET  
        sum_word_pos_bi = []  #sum of log10(P(Word|Positive))
        sum_word_neg_bi = []  #sum of log10(P(Word|Negative))        
        #calculate the prob of a review being positive given the words in it
        review_set_length_bi = len(review)-1
        
        for index2 in range(0, review_set_length_bi):
            next = index2 + 1
            pair = (review[index2], review[next])

            if pos_prob_bi.get(pair) is None:  #if pair of word is not in training set
                sum_word_pos_bi.append(np.log10(prob_unknown_bi_pos)) # add prob_unknown to list of P(Word|Type=Positive)
            else:
                sum_word_pos_bi.append(np.log10(pos_prob_bi[pair])) # add P(Word|Type=Positive) to list of P(Word|Type=Positive)

            if neg_prob_bi.get(pair) is None:  #if pair of word is not in training set
                sum_word_neg_bi.append(np.log10(prob_unknown_bi_neg))
            else:
                sum_word_neg_bi.append(np.log10(neg_prob_bi[pair]))

            index2 += 1
    # P(Positive|Words)   log10(P(Positive))  sum of log10(P(Word|Positive)) 
        prob_type_pos_bi = np.log10(pos_prior) + sum(sum_word_pos_bi)
    # P(Negative|Words)   log10(P(Negative))  sum of log10(P(Word|Negative)) 
        prob_type_neg_bi = np.log10(1-pos_prior) + sum(sum_word_neg_bi)
        #print(prob_type_pos_bi, prob_type_neg_bi)
        # P(Positive|Words)    (1-lambda)*log(P(Y)*PI(P(Wi|Y)))       (lambda)*log(P(Y)*PI(P(Bi|Y)))
        combined_type_pos = ((1-bigram_lambda)*prob_type_pos_uni) + ((bigram_lambda)*prob_type_pos_bi)       
        combined_type_neg = ((1-bigram_lambda)*prob_type_neg_uni) + ((bigram_lambda)*prob_type_neg_bi) 

        #print(bigram_lambda)
        #print(sum(sum_word_pos))
        if (combined_type_pos) > (combined_type_neg):  # compare which one has greater prob
            yhats.append(1)
            continue
        else:
            yhats.append(0)
            continue

    #for doc in tqdm(dev_set,disable=silently):
    #    yhats.append(-1)
    return yhats

