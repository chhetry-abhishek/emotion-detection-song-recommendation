# Import libraries
import pandas as pd
from nrclex import NRCLex


class Songs_Recommendation:

    def __init__(self, emotion):
        # Read csv file
        df = pd.read_csv('tcc_ceds_music.csv')
        # Data cleaning I
        df = df[['artist_name', 'track_name', 'release_date', 'genre', 'lyrics', 'len']]
        # Creating dataset
        df['emotions'] = df['lyrics'].apply(lambda x: NRCLex(x).affect_frequencies)
        df = pd.concat([df, df['emotions'].apply(pd.Series)], axis=1)
        # Data cleaning II
        df = df.drop(['anticip', 'anticipation', 'trust'], axis=1)
        df = df.rename(columns={'anger': 'Angry', 'disgust': 'Disgust', 'fear': 'Fear', 'joy': 'Happy', 'sadness': 'Sad', 'surprise': 'Surprise'}, inplace=False)
        # Initiating dataframe
        self.df = df
        self.emotion = emotion

    def get_songs_recommended(self):
        try:
          emotion_related_best_songs = self.df.sort_values(self.emotion, ascending=False)
          songs_recommended = emotion_related_best_songs.head(10).set_index('track_name').to_dict('index')
          n = 0
          for key, value in songs_recommended.items():
                n += 1
                print(f"{n}. {key.upper()} by {value['artist_name'].capitalize()}")
          return "Above are the songs recommended"  
        except Exception as e:
            return "Neutral songs are not available, try other emotions"