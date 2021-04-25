from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier

import re, string, random


class SentimentAnalyzerService:
    def __init__(self):
        self.positive_tweets = twitter_samples.strings('positive_tweets.json')
        self.negative_tweets = twitter_samples.strings('negative_tweets.json')
        self.text = twitter_samples.strings('tweets.20150430-223406.json')
        self.tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

        self.stop_words = stopwords.words('english')

        self.positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
        self.negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

        self.positive_cleaned_tokens_list = []
        self.negative_cleaned_tokens_list = []

    def remove_noise(self, tweet_tokens, stop_words=()):

        cleaned_tokens = []

        for token, tag in pos_tag(tweet_tokens):
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                           '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
            token = re.sub("(@[A-Za-z0-9_]+)", "", token)

            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)

            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                cleaned_tokens.append(token.lower())
        return cleaned_tokens

    def get_all_words(self, cleaned_tokens_list):
        for tokens in cleaned_tokens_list:
            for token in tokens:
                yield token

    def get_tweets_for_model(self, cleaned_tokens_list):
        for tweet_tokens in cleaned_tokens_list:
            yield dict([token, True] for token in tweet_tokens)

    def run(self):

        for tokens in self.positive_tweet_tokens:
            self.positive_cleaned_tokens_list.append(self.remove_noise(tokens, self.stop_words))

        for tokens in self.negative_tweet_tokens:
            self.negative_cleaned_tokens_list.append(self.remove_noise(tokens, self.stop_words))

        all_pos_words = self.get_all_words(self.positive_cleaned_tokens_list)

        freq_dist_pos = FreqDist(all_pos_words)
        print(freq_dist_pos.most_common(10))

        positive_tokens_for_model = self.get_tweets_for_model(self.positive_cleaned_tokens_list)
        negative_tokens_for_model = self.get_tweets_for_model(self.negative_cleaned_tokens_list)

        positive_dataset = [(tweet_dict, "Positive")
                            for tweet_dict in positive_tokens_for_model]

        negative_dataset = [(tweet_dict, "Negative")
                            for tweet_dict in negative_tokens_for_model]

        dataset = positive_dataset + negative_dataset

        random.shuffle(dataset)

        train_data = dataset[:7000]
        test_data = dataset[7000:]

        classifier = NaiveBayesClassifier.train(train_data)

        print("Accuracy is:", classify.accuracy(classifier, test_data))

        print(classifier.show_most_informative_features(10))

        custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."

        custom_tokens = self.remove_noise(word_tokenize(custom_tweet))

        print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))


if __name__ == "__main__":
    service = SentimentAnalyzerService()
    service.run()

