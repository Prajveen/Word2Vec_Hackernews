In this project, we will try to predict the upvote score of posts on the Hacker News website https://news.ycombinator.com/ using just their titles.

Import SentencePiece to use for tokenizing our data

Prepare the dataset of Hacker News titles and upvote scores
Obtain the data from the database.
Use psql or some other tool to connect.
Tokenise the titles using SentencePiece
Implement and train an architecture to obtain word embeddings in the style of the word2vec paper
https://arxiv.org/pdf/1301.3781.pdf
using either the *continuous bag of words (CBOW) or Skip-gram model (or both).
Implement a regression model to predict a Hacker News upvote score from the pooled average of the word embeddings in each title.
Extension: train your word embeddings on a different dataset, such as
More Hacker News content, such as comments
A completely different corpus of text, like (some of) Wikipedia
