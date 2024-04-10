import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import * 
import  matplotlib.pyplot as plt

data = pd.read_csv("tar_ap24.csv")
print(data)

print(data.isnull().sum())

sia = SentimentIntensityAnalyzer()
def gs(txt):
    ps = sia.polarity_scores(txt)
    if ps["compound"] >= 0.05 :
        return "pos"
    elif ps["compound"] <= -0.05 :
        return "neg"
    else:
        return "neu"
    
data["sentiments"] = data["Review"].apply(gs)
print(data)

pos_data = data[data.sentiments == "pos"]
# print(pos_data)
pos_reviews = " ".join(pos_data["Review"])
# print(pos_reviews)

wc1 = WordCloud(max_words=50,width=800,height=400,background_color="white",colormap="Set2").generate(pos_reviews)
plt.figure(figsize=(12,5))
plt.axis("off")
plt.imshow(wc1)
plt.savefig("tarpos.png")
