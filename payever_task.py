#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Embedding,
    Flatten
)
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
# In[ ]:


# get_ipython().system('pip install Faker')


# In[ ]:


import random
from faker import Faker

# Initialize Faker
fake = Faker()

# Define product categories
categories = [
    "Electronics",
    "Personal Care",
    "Sports & Outdoors",
    "Home Appliances",
    "Furniture"
    ]

# Define sample product names and descriptions by category
product_samples = {
    "Electronics": [
        {
            "name": "Laptop",
            "description": "High-performance laptop with {}-inch display"
        },
        {
            "name": "Smartphone",
            "description": "Latest model smartphone with {} battery"
        },
        {
            "name": "Headphones",
            "description": "Noise-cancelling {} headphones"
        },
        {
            "name": "Tablet",
            "description": "Portable tablet with {}-inch screen"
        },
        {
            "name": "Smartwatch",
            "description": "Wearable smartwatch with {} features"
        }
    ],
    "Personal Care": [
        {
            "name": "Toothbrush",
            "description": "Rechargeable electric toothbrush with {} modes"
        },
        {
            "name": "Hair Dryer",
            "description": "Professional hair dryer with {} settings"
        },
        {
            "name": "Electric Shaver",
            "description": "Rechargeable electric shaver with {} blades"
        },
        {
            "name": "Facial Cleanser",
            "description": "Electric facial cleanser with {} speed settings"
        },
        {
            "name": "Hair Straightener",
            "description": "Ceramic hair straightener with"
            " {} temperature settings"
        }
    ],
    "Sports & Outdoors": [
        {
            "name": "Yoga Mat",
            "description": "Eco-friendly yoga mat with {} surface"
        },
        {
            "name": "Running Shoes",
            "description": "Lightweight running shoes with {} material"
        },
        {
            "name": "Tent",
            "description": "Waterproof tent for {} people"
        },
        {
            "name": "Fitness Tracker",
            "description": "Wearable fitness tracker with {} features"
        },
        {
            "name": "Bicycle",
            "description": "Mountain bicycle with {} gears"
        }
    ],
    "Home Appliances": [
        {
            "name": "Blender",
            "description": "High-speed blender with {} settings"
        },
        {
            "name": "Coffee Maker",
            "description": "Programmable coffee maker with {} capacity"
        },
        {
            "name": "Vacuum Cleaner",
            "description": "Bagless vacuum cleaner with {} suction power"
        },
        {
            "name": "Microwave Oven",
            "description": "Countertop microwave oven with {} presets"
        },
        {
            "name": "Air Purifier",
            "description": "HEPA air purifier with {} speed settings"
        }
    ],
    "Furniture": [
        {
            "name": "Desk Chair",
            "description": "Ergonomic desk chair with {} support"
        },
        {
            "name": "Dining Table",
            "description": "Modern dining table with seating for {}"
        },
        {
            "name": "Sofa",
            "description": "Comfortable sofa with {} cushions"
        },
        {
            "name": "Bookshelf",
            "description": "Wooden bookshelf with {} shelves"
        },
        {
            "name": "Bed Frame",
            "description": "King-size bed frame with {} design"
        }
    ],
}

# Function to generate a random product


def generate_random_product():
    category = random.choice(categories)
    product_sample = random.choice(product_samples[category])
    name = f"{product_sample['name']} {fake.word().capitalize()}"
    description = product_sample["description"].format(fake.word())
    price = round(random.uniform(10.0, 2000.0), 2)
    return {
        "name": name,
        "description": description,
        "price": price,
        "category": category
    }


# Generate 10,000 random products
products = [generate_random_product() for _ in range(10000)]

# Convert to DataFrame
df = pd.DataFrame(products)

# Display the DataFrame
print(df.head())

# Save to CSV for further use
df.to_csv("dummy_products.csv", index=False)


# In[ ]:


df.sample(5)


# In[ ]:


# converting to lowercase
# removing special characters, punctuation


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


df['name'] = df['name'].apply(clean_text)
df['description'] = df['description'].apply(clean_text)


# In[ ]:


# get_ipython().system('pip install nltk')


# In[ ]:


# tokenizing data
nltk.download('punkt')

df['name'] = df['name'].apply(word_tokenize)
df['description'] = df['description'].apply(word_tokenize)


# In[8]:


df.head()


# In[9]:


# removing stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def remove_stop_words(tokens):
    return [word for word in tokens if word not in stop_words]


df['name'] = df['name'].apply(remove_stop_words)
df['description'] = df['description'].apply(remove_stop_words)


# In[10]:


df.head()


# In[11]:


# lemmatization - converting words to their root word or base form
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()


def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]


df['name'] = df['name'].apply(lemmatize_tokens)
df['description'] = df['description'].apply(lemmatize_tokens)


# In[12]:


df.head()


# In[15]:
tokenizer = Tokenizer()

df['name'] = df['name'].astype(str)
df['description'] = df['description'].astype(str)

# Concatenate text data from both columns
all_text = (df['name'].apply(lambda x: ' '.join(x)) + ' ' +
            df['description'].apply(lambda x: ' '.join(x)))
tokenizer.fit_on_texts(all_text)

# Convert text data in each column to sequences of integers
sequences_col1 = tokenizer.texts_to_sequences(
    df['name'].apply(lambda x: ' '.join(x))
)
sequences_col2 = tokenizer.texts_to_sequences(
    df['description'].apply(lambda x: ' '.join(x))
)

# Pad sequences to ensure uniform length
max_length = max(
    max(len(seq) for seq in sequences_col1),
    max(len(seq) for seq in sequences_col2)
)
padded_sequences_col1 = pad_sequences(
    sequences_col1, maxlen=max_length, padding='post'
)
padded_sequences_col2 = pad_sequences(
    sequences_col2, maxlen=max_length, padding='post'
)

# Replace original text data in the columns with padded sequences
df['name'] = padded_sequences_col1.tolist()
df['description'] = padded_sequences_col2.tolist()


# In[16]:


df.head()


# In[17]:
label_encoder = LabelEncoder()
df['category'] = label_encoder.fit_transform(df['category'])
y = df['category']


# In[18]:
scaler = MinMaxScaler(feature_range=(1, 10))
df['price'] = scaler.fit_transform(df[['price']])
df['combined'] = df.apply(
    lambda row: row['name'] +
    row['description'] +
    [int(row['price'])], axis=1
    )

max_seq_length = max(df['combined'].apply(len))
X = pad_sequences(df['combined'], maxlen=max_seq_length, padding='post')

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)


# In[8]:


df.head()


# In[19]:
model = Sequential([
    Embedding(input_dim=100, output_dim=50, input_length=max_seq_length),
    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Use 'categorical_crossentropy' for multiclass
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[20]:


history = model.fit(X_train, y_train,
                    epochs=10,  # Adjust the number of epochs
                    batch_size=128,  # Adjust the batch size
                    validation_split=0.2)


# In[26]:
train_loss, train_accuracy = model.evaluate(X_train, y_train)
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")
print(f"Testing Loss: {test_loss}, Testing Accuracy: {test_accuracy}")
# In[27]:


# Predict on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate evaluation metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

print("\nClassification Report:")
target_names = list(map(str, label_encoder.classes_))
print("Target names:", target_names)  # Debugging line to verify target names

print("Encoded Label -> Actual Category")
for encoded_label in range(len(label_encoder.classes_)):
    actual_category = label_encoder.inverse_transform([encoded_label])
    print(f"{encoded_label} -> {actual_category[0]}")
print(classification_report(y_test, y_pred_classes, target_names=target_names))


# In[1]:


# get_ipython().system('pip install nbconvert flake8')


# In[2]:


# get_ipython().system('jupyter nbconvert --to script payever_task.ipynb')


# In[54]:


# get_ipython().system('flake8 payever_task.py')


# In[1]:


# get_ipython().system('pip install pipreqs')


# In[7]:


# get_ipython().system('pipreqs . --use-local')
