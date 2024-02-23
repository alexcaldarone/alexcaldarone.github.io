---
layout: post
title: Lessons Learned From My First NLP Competition
date: 2023-09-19
tags: Programming
---

Earlier this month, together with two other students from the University of Padua, I took part in the [ITADATAhack 2023](https://journal.opendataplayground.com/ITADATAhack-2023/) competition. This proved to be an incredible opportunity to push my capabilities in data analysis and modelling. 

The competition consisted of developing a classification algorithm that, given the raw text of a legal document as input, would be able to classify the document in the correct category. In order to do this, we were given access to a dataset containing the legal documents found in the [EURLex database](https://eur-lex.europa.eu/collection/eu-law/legislation/recent.html). The competition spanned over 3 days and was organized in increasing levels of difficulty. 

On the first day we had to solve a 20 label classification problem, then a 96 label classification on the second day and then on the last day we had to deal with a binary multi-label classification problem (where each label had 89 dimensions). This competition was open to all universities in Italy and there were a total of 31 participating teams.

### 1. How do you embed text?
After analysing and cleaning the raw text, one of the most important steps is _embedding_ it. Thats is, to convert it from words to numbers. For this step we tried two different approaces: a simple one (TF-IDF) and a more complicated one (transformer model embeddings).

**_TF-IDF_**

The [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) is a combination of two measures to assign a score to every word in a document. The first measure _tf_ (term frequency) considers the frequency of a word inside of a document, while _idf_ (inverse document frequency) is a measure of the importance of how much information a word provides (i.e. if it's very common or rare among the documents in the corpus). The final score is the multiplication of these two measures. 

To our surprise the TF-IDF approach worked incredibly well, giving a classification accuracy of over 90% (on the dataset with balanced labels) when combined with a model such as logistic regression or KNN classification. Although this method has an interpretability advantage over the transformer model embeddings, it does come at the cost of having to perfrom computiationally intensive text cleaning (such as stopword removal, lemmatization of text and further custom cleaning procedures specifically for the type of text we were handling) in order to perform well. Overall, even if transformer model embeddings allowed us to achieve slightly better results, the TF-IDF approach proved to be very valid and how a simple solution can perform very well even on a complicated task.

**_Transformer model embeddings_**

All of our final solutions to the challenge utilized transformer model embeddings as the bedrock on which we built our models. Initially we debated whether to use a fine-tuned version of BERT ([LEGAL-BERT](https://huggingface.co/nlpaueb/legal-bert-base-uncased)) or another model we had found on HuggingFace, which was [e5-small](https://huggingface.co/intfloat/e5-small-v2). After some tests we ended up going with the latter. This was because this model's embedding dimension is 384, compared to 768 for the fine-tuned version of BERT. After some testing we established that even though LEGAL-BERT was finetuned on legal documents (including the EURLex database) it did not provide an increase in the accuracy of the models that were trained utilizing the word embeddings. 

Embedding long documents proved to be an interesting challenge. This was becuase the transformer model we chose had a maximum input size of 512 tokens, but we were dealing with documents that had varying length, with some being several thousand words long. We solved this problem by tokenizing and embedding the text using a "sliding window approach", that is we only considered _n_ words at a time, tokenized them and added them to an array where we stored the current law's tokens. Then we iterated over this array and passed 512 tokens at a time to the transformer model in order to obtain their embeddings. Finally we concatenateed the embeddings of the various sequences into an array. This meant that we had embedding vectors of varying length (shorter laws had shorter embedding vectors, while longer laws had longer embedding vectors). 

This solution posed another challenge. As we know, when training neural networks the inputs all need to have the same dimension. We decided to solve this problem by averaging the elements in the embedding vector in order to obtain vectors that all had 384 dimensions (see [here](https://datascience.stackexchange.com/questions/107462/why-does-averaging-word-embedding-vectors-exctracted-from-the-nn-embedding-laye) why it makes sense to average embedding vectors).

Embedding vectors also had the advantage of allowing us to augment the dataset in order to create "fake data" to train models that would have otherwise struggled to classify the laws correctly because of unbalanced labels. 

Overall these embeddings proved to be an extremely powerful and versatile instrument that allowed us to train a variety of models efficiently, although this came at the cost of losing the interpretability we had when using the TF-IDF method and a higher computational cost (we had to use GPUs in order to obtain them in a time-efficient way).

### 2. Challenges of unbalanced datasets

On the first day of the competition we had a 20 label classification problem and a balanced dataset. On the second day, we had more labels to classify correctly but we also had an _unbalanced_ dataset. This proved to an interesting challenge to confront because for some categories we only had a handful or one example. This obviously meant that the models struggled to learn the features of the laws in these categories and failed to classify them correctly. 

**_The solution_**

In order to solve this problem we devised a particular resampling technique. The script we ended up writing essentialy resampled the embedding vectors to create new instances of the law embeddings belonging to a certain category. This solution was based on the intuition that the embeddings of the laws amongst the different categories would have different properties and, that by resampling them, we would be able to create new embedding vectors that had properties similar to the ones in the category from which the orignal one was selected. Essentially this was a way of applying data augmentation to text embeddings.

This method only made sense with the transformer model embeddings as these encode the similarity of the vaious tokens within each other, while the tf-idf representation of the corpus is based on the frequency of the words inside the documents and in how many documents these words appear. This means that we obviously couldn't have resampled these frequencies in order to create new "fake data".

### 3. Features > Models

In the first day of the competition the TF-IDF representation of the corpus combined with simple statistical models such as logistic regression, K-nearest neighbors classification provided very good results. Although these results were very good, they were slightly outperformed by the accuracy obtained using a simple feed-forward neural network. The best performing model though was obtained using the transformer model embeddings combined with additional features designed to capture another important features present in the data, which was the citations of each law (or, the other laws that a particular law referenced). One of these features was the label probability distribution for a particular law depending on its citations. We constructed this by considering the citations of a particular document and considering what was the number of documents in each category that referenced that shared at least on citation.

For the second day the best performing model was a neural network which took as input the transformer mdoel embeddings, the citation features and some other features. In the second and third day the dataset rebalancing proved to be incredibly useful in increasing the model's performance. The same architecture, with some small modifications to account for the different type of output, proved to be efficient for the third day as well. 

The scores our team obtained were: 
- Day 1: 0.9236/1
- Day 2: 0.8681/1
- Day 3: 0.8542/1 

Final score (weighted average of the three): 0.8704/1

The score for the first two days is the F-1 score, while the score utilized for the third day was the Jaccard score.

Overall, while I am pleased with the result our team achieved I think there is one thing that penalized us: _not spending enough time on feature engineering_. After having obtained good baseline features such as the text embeddings, the features corresponding to the citations and some other, we shifted out attention to the models we were using. I think this penalized us and that, instead of spending too much time tweaking the model's architecture, we should have spent most of our time exploring the raw data and trying to find other features which could have been more useful that the ones we had already found. 

This will be an incrediby important lesson that I will not forget when working on future projects. Even though it has been said many times, the best way to learn this lesson is to commit the mistake of focusing too much on the model instead of the data (_"garbage in, garbage out"_ as they saying goes). **Finding the correct features is more important than stressing over the model you are using.**

---

This competition was an incredible learning experience, where I managed to compress tons of learning and experimentation in just three days. While I do think that we commited some mistakes and could have done things differently, I learned a lot of useful lessons that I will carry with me in future projects. Our team placed 13th out of 31 participating teams, which I consider to be a good result given this was my first experience in this type of competition!