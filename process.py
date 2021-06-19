import tensorflow as tf
from tensorflow import keras
import pickle
from keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model('model/model.h5')

tokenizer = pickle.load(open('model/tokenizer.pkl', 'rb'))

decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
SENTIMENT_THRESHOLDS = (0.5, 0.6)
def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = "NEUTRAL"
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = "NEGATIVE"
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = "POSITIVE"

        return label
    else:
        return 'NEGATIVE' if score < 0.5 else 'POSITIVE'
    

def predict(text, include_neutral=True):
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=300)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score)

    return {"label": label, "score": float(score)} 

