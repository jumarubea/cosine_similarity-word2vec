import nltk
import numpy as np
from model import get_country
from gensim.models import KeyedVectors


def extract_dataset():
    '''
    Load pre-trained word embeddings and extract a dataset of relevant key features.

    Steps:
    1. Load Google News pre-trained Word2Vec embeddings.
    2. Read and tokenize words from 'capitals.txt'.
    3. Create a set of unique words, including manually selected keywords.

    Returns:
        tuple: 
            - embeddings (KeyedVectors): Pre-trained word embeddings.
            - set_words (set): Unique set of words including those from 'capitals.txt' and predefined keywords.
    '''
    # Load pre-trained Word2Vec embeddings
    embeddings = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

    # Read and tokenize the 'capitals.txt' file
    f = open('capitals.txt', 'r').read()
    set_words = set(nltk.word_tokenize(f))

    # Predefine a list of additional relevant keywords
    select_words = ['king', 'queen', 'oil', 'gas', 'happy', 'sad', 
                    'city', 'town', 'village', 'country', 'continent', 
                    'petroleum', 'joyful']
    
    # Add the predefined keywords to the set of words
    for w in select_words:
        set_words.add(w)
    
    return embeddings, set_words

def get_word_embeddings():
    '''
    Extract word embeddings for a specific set of words.

    Steps:
    1. Use `extract_dataset()` to load embeddings and key word sets.
    2. Filter the embeddings to include only words in the selected set.

    Returns:
        dict: A dictionary containing filtered word embeddings for the selected words.
    '''
    # Extract embeddings and selected words using `extract_dataset`
    embeddings, set_words = extract_dataset()
    
    # Initialize dictionary to hold filtered word embeddings
    word_embeddings = {}

    # Iterate through all words in the embeddings and add relevant ones to the dictionary
    for word in embeddings.key_to_index:
        if word in set_words:
            word_embeddings[word] = embeddings[word]
    
    return word_embeddings


def euclidean(A, B):
    """
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        d: numerical number representing the Euclidean distance between A and B.
    """

    # euclidean distance

    d = np.linalg.norm(A-B)

    return d


def cosine_similarity(A, B):
    '''
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    '''
    
    dot = np.dot(A,B)
    norma = np.sqrt(np.dot(A,A))
    normb = np.sqrt(np.dot(B,B))
    cos = dot / (norma*normb)

    return cos


def accuracy(word_embeddings, data):
    '''
    Input:
        word_embeddings: a dictionary where the key is a word and the value is its embedding
        data: a pandas dataframe containing all the country and capital city pairs
    
    Output:
        accuracy: the accuracy of the model
    '''

    # initialize num correct to zero
    num_correct = 0

    # loop through the rows of the dataframe
    for i, row in data.iterrows():

        # get city1
        city1 = row['city1']

        # get country1
        country1 = row['country1']

        # get city2
        city2 =  row['city2']

        # get country2
        country2 = row['country2']

        # use get_country to find the predicted country2
        predicted_country2, _ = get_country(city1,country1,city2,word_embeddings)

        # if the predicted country2 is the same as the actual country2...
        if predicted_country2 == country2:
            # increment the number of correct by 1
            num_correct += 1

    # get the number of rows in the data dataframe (length of dataframe)
    m = len(data)

    # calculate the accuracy by dividing the number correct by m
    accuracy = num_correct/m

    return accuracy


def get_vectors(embeddings, words):
    """
    Input:
        embeddings: a word 
        fr_embeddings:
        words: a list of words
    Output: 
        X: a matrix where the rows are the embeddings corresponding to the rows on the list
        
    """
    m = len(words)
    X = np.zeros((1, 300))
    for word in words:
        english = word
        eng_emb = embeddings[english]
        X = np.row_stack((X, eng_emb))
    X = X[1:,:]
    return X
