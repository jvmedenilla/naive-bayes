# naive-bayes

Machine Problem 3 of ECE 448/CS 440 - Fall 2021 at University of Illinois Urbana-Champaign

Using Naive Bayes, I labeled movie reviews as positive or negative using both unigram and bigram models.

Following Bayes Theorem: P(Type=Positive | Words) = P(Type=Positive)/P(Words) * P(Word|Type = Positive)
                         P(Type=Negative | Words) = P(Type=Negative)/P(Words) * P(Word|Type =Negative)
                         
Part 1 of this code implements Unigram model. A Unigram model tallies the likelihood probability, P(Word|Type = Positive), per each word occurence.
Meanwhile, Part 2 demonstrates a Bigram Mixture Model where I combined Unigram with Bigram. Bigram model, in contrast with Unigram, tallies the likelihood probability
of word pairs, i.e. "is good", each time a word pair occurs. In this project, a word pair was defined as any two consecutive words in a sentence. 

To make the model more sensible, I removed stopwords so I could focus on words that inherently have positive or negative connotations, or atleast most oftenly used in such manner.
