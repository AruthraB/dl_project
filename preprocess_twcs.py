import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# ---------------- LOAD DATA ----------------
data = pd.read_csv("twcs/twcs.csv", low_memory=False)

# ---------------- FILTER ----------------
data = data[data['inbound'] == True]
data = data[['text']]
data = data.dropna()

# ---------------- CLEAN TEXT ----------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

data['text'] = data['text'].apply(clean_text)

# ---------------- CATEGORY ----------------
def assign_category(text):
    if any(w in text for w in ['refund','payment','charge','billing','invoice','money']):
        return 'Billing'
    elif any(w in text for w in ['password','login','account','reset','locked']):
        return 'Account'
    elif any(w in text for w in ['error','issue','bug','crash','fail','problem']):
        return 'Technical'
    elif any(w in text for w in ['delivery','shipping','delay','late','tracking']):
        return 'Delivery'
    else:
        return 'General'

data['category'] = data['text'].apply(assign_category)

# ---------------- URGENCY (IMPROVED) ----------------
def assign_urgency(text):

    # HIGH (strong signals)
    if any(w in text for w in [
        'urgent','asap','immediately','critical','emergency',
        'worst','very bad','serious','right now'
    ]):
        return 'High'

    # MEDIUM (common problems)
    elif any(w in text for w in [
        'error','failed','issue','problem','not working',
        'unable','delay','late','crash'
    ]):
        return 'Medium'

    else:
        return 'Low'

data['urgency'] = data['text'].apply(assign_urgency)

# ---------------- BALANCE CATEGORY ----------------
max_cat = 300
cat_balanced = []

for c in data['category'].unique():
    subset = data[data['category'] == c]
    subset = subset.sample(min(len(subset), max_cat), random_state=42)
    cat_balanced.append(subset)

data = pd.concat(cat_balanced)

# ---------------- BALANCE URGENCY ----------------
max_urg = 200
urg_balanced = []

for u in data['urgency'].unique():
    subset = data[data['urgency'] == u]

    if len(subset) < max_urg:
        subset = subset.sample(max_urg, replace=True, random_state=42)
    else:
        subset = subset.sample(max_urg, random_state=42)

    urg_balanced.append(subset)

data = pd.concat(urg_balanced)

# ---------------- SHUFFLE ----------------
data = data.sample(frac=1, random_state=42)

# ---------------- OUTPUT ----------------
print("\nCategory Count:\n", data['category'].value_counts())
print("\nUrgency Count:\n", data['urgency'].value_counts())

print("\nSample:\n", data.head())

# ---------------- SAVE ----------------
data.to_csv("final_data.csv", index=False)

print("\n✅ CLEAN BALANCED DATA READY")