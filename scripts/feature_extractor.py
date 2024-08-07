import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
class FeatureExtractor:

    def __init__(self):
        self.sentences_pairs = []
        self.labels = []
        self.tfidf_vectorizer = TfidfVectorizer()

    def load_data(self, data, extract_labels=True):
        # Extract sentences and labels
        for item in data:
            judge, orig_sent, cand_sent, trend_id = item
            self.sentences_pairs.append((orig_sent, cand_sent))
            if extract_labels:
                self.labels.append(int(judge))
    
    def preprocess(self,text):
        # Tokenize the text and remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        return ' '.join(filtered_tokens)
    

    
    def extract_features(self):
        # Preprocess sentences and join them for vectorization
    
        all_sentences = [self.preprocess(sent) for pair in self.sentences_pairs for sent in pair]
        
        # fit a TF-IDF vectorizer to extract the frequency of words in the sentences
        self.tfidf_vectorizer.fit(all_sentences)
        
        # Initialize lists for features
        cosine_similarities = []
        jaccard_similarities = []
        length_diffs = []
        common_words = []
        pos_similarities = []
        
        for orig_sent, cand_sent in self.sentences_pairs:
            # Preprocess sentences
            orig_sent_processed = self.preprocess(orig_sent)
            cand_sent_processed = self.preprocess(cand_sent)
            
            # Feature 1: Cosine Similarity using TF-IDF
            tfidf_orig = self.tfidf_vectorizer.transform([orig_sent_processed])
            tfidf_cand = self.tfidf_vectorizer.transform([cand_sent_processed])
            cosine_sim = cosine_similarity(tfidf_orig, tfidf_cand)[0][0]
            cosine_similarities.append(cosine_sim)
            
            # Feature 2: Jaccard Similarity
            set_orig = set(orig_sent_processed.split())
            set_cand = set(cand_sent_processed.split())
            intersection = set_orig.intersection(set_cand)
            union = set_orig.union(set_cand)
            jaccard_sim = len(intersection) / len(union) if len(union) > 0 else 0
            jaccard_similarities.append(jaccard_sim)
            
            # Feature 3: Sentence Length Difference
            length_diff = abs(len(orig_sent) - len(cand_sent))
            length_diffs.append(length_diff)

            # Feature 4: Number of common words
            common_word = len(intersection)
            common_words.append(common_word)

            # Feature 5: POS tags similarity
            pos_orig = pos_tag(orig_sent_processed.split())
            pos_cand = pos_tag(cand_sent_processed.split())
            pos_common = set(pos_orig).intersection(set(pos_cand))
            pos_union = set(pos_orig).union(set(pos_cand))
            pos_sim = len(pos_common) / len(pos_union) if len(pos_union) > 0 else 0
            pos_similarities.append(pos_sim)


        
        # Combine all features into a single feature matrix
        X = np.array(list(zip(cosine_similarities, jaccard_similarities, length_diffs, common_words, pos_similarities)))
        y = np.array(self.labels)
        
        return X, y