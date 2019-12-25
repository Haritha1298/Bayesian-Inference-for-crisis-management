# Bayesian-Inference-for-crisis-management
Large volumes of data is generated through social media tweets during and after crisis. Analysis and classification of tweets is essential for situational awareness and alerting appropriate personnel.

Packages Required: 1. pybbn 2. dbn.tensorflow
Data Source: CrisisLexT26

### Methods Used
##### Nested Naive Bayes Classifier
<b>Our hypothesis:</b> A tweet belongs to the informativeness (I) class “related and informative”, and that it fits into information type (T).  The “evidence” would be the words (W) occurring in the tweet.
i.e., we are interested in P( I | W) and P(T | I, W)<br>
In our Naïve Bayes Classifier, we assumed a bag-of-words (i.i.d) representation.  So we computed P(I | W₁, W₂ …Wn) and P(T | I, W₁, W₂ …Wn), where W is broken down to a word vector.<br>
Obtained the word vector using CountVectorizer() from sklearn<br>
Stripped out nonwords, numbers, articles, and other things that are not useful for predicting informativeness (I) and information type (T) from our counts (W).<br>
We used a default 75-25 training/test data split and trained a Multinomial Naïve Bayes Classifier with Leave One Out Cross Validation.<br>
In the first stage of our nested classifier, we predicted the informativeness P(I | W).<br>
If we predicted that a tweet is “related and informative”, in the next stage, we now predict the information type P(T | I, W).<br>


##### Deep Belief Network without Latent Variable using dbn.tensorflow library
Start with a random configuration<br>
Binary state of each variable is chosen from a Bernoulli Distribution determined by active parents on the level above<br>
We can then sample from the posterior of the hidden layer on to the visible layer.<br>
Use the transpose of our weight matrix to infer distribution over the hidden layer<br>
refer "G. E. Hinton, S. Osindero, Y. W. Teh. “A fast learning algorithm for deep belief nets, ” in Neural Comput, 2006;18(7):1527–1554." for more detailed explanation.

##### Bayesian Belief Network with latent Variable using pybbn library
![Repo List](bnet.png)
Random Variables: Information Source(A), Informativeness(B), Information Type(C)
From the bayesian network, the joint probability distribution can be written as
P(A,B,C) = P(A)* P(B/C) * P(C/B)

