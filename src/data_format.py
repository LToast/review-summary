from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))
CONTRACTION_MAPPING = {
    "ain't": '', "aren't": '', "can't": 'cannot', "'cause": '', "could've": 'could', "couldn't":
    'could', "didn't": '', "doesn't": '', "don't": '', "hadn't": '', "hasn't": '', "haven't": '', "he'd": 'would',
    "he'll": '', "he's": '', "how'd": '', "how'd'y": '', "how'll": '', "how's": '', "I'd": 'would',
    "I'd've": 'would', "I'll": '', "I'll've": '', "I'm": '', "I've": '', "i'd": 'would', "i'd've":
    'would', "i'll": '', "i'll've": '', "i'm": '', "i've": '', "isn't": '', "it'd": 'would',
    "it'd've": 'would', "it'll": '', "it'll've": '', "it's": '', "let's": 'letus', "ma'am": 'madam',
    "mayn't": 'may', "might've": 'might', "mightn't": 'might', "mightn't've": 'might',
    "must've": 'must', "mustn't": 'must', "mustn't've": 'must', "needn't": 'need',
    "needn't've": 'need', "o'clock": 'clock', "oughtn't": 'ought', "oughtn't've": 'ought',
    "shan't": 'shall', "sha'n't": 'shall', "shan't've": 'shall', "she'd": 'would',
    "she'd've": 'would', "she'll": '', "she'll've": '', "she's": '', "should've": '',
    "shouldn't": '', "shouldn't've": '', "so've": '', "so's": '', "this's": '', "that'd": 'would',
    "that'd've": 'would', "that's": '', "there'd": 'would', "there'd've": 'would', "there's": '',
    "here's": '', "they'd": 'would', "they'd've": 'would', "they'll": '', "they'll've": '',
    "they're": '', "they've": '', "to've": '', "wasn't": '', "we'd": 'would', "we'd've": 'would',
    "we'll": '', "we'll've": '', "we're": '', "we've": '', "weren't": '', "what'll": '',
    "what'll've": '', "what're": '', "what's": '', "what've": '', "when's": '', "when've": '',
    "where'd": '', "where's": '', "where've": '', "who'll": '', "who'll've": '', "who's": '',
    "who've": '', "why's": '', "why've": '', "will've": '', "won't": '', "won't've": '',
    "would've": 'would', "wouldn't": 'would', "wouldn't've": 'would', "y'all": '',
    "y'all'd": 'would', "y'all'd've": 'would', "y'all're": '', "y'all've": '', "you'd": 'would',
    "you'd've": 'would', "you'll": '', "you'll've": '', "you're": '', "you've": ''}


def stopwords_filter(sentence):
    return " ".join(CONTRACTION_MAPPING.get(word.lower(), word.lower())
                    for word in sentence.split() if word.lower() not in STOP_WORDS)


def data_processing(df):
    df["reviewText"] = df["reviewText"].apply(lambda x: stopwords_filter(x))
    df["summary"] = df["summary"].apply(lambda x: stopwords_filter(x))
    return df
