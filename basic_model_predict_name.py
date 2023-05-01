# In[15]:


import pandas as pd

#modele (algorytmy)
from sklearn.dummy import DummyClassifier           # <== Najprostszy możliwy model 
from sklearn.linear_model import LogisticRegression # <== Regresja logistyczna (liniowa)

#metryka sukcesu
from sklearn.metrics import accuracy_score

# In[16]:


df = pd.read_csv("../input/polish_names.csv")
df.head()


# In[18]:


df.sample(10)

# In[19]:

df['gender'].value_counts()

# In[20]:


def transform_string_into_number(string):
    return string
    
df['gender'].head().map( transform_string_into_number )


# In[21]:


def transform_string_into_number(string):
    return int(string == 'm')
    
df['gender'].head().map( transform_string_into_number )

# In[22]:


df['target'] = df['gender'].map( lambda x: int(x == 'm') )
df.head(10)


# ## Zadanie 1.3.1
# Twoim zadaniem jest stworzyć nową cechę (kolumnę), która będzie zawierać długość imienia (możesz stworzyć więcej niż jedną cechę, o ile masz na to pomysły).
# 
# 

# In[23]:


import pandas as pd

df = pd.read_csv("../input/polish_names.csv")
df.head()

df['len_name'] = df['name'].map( lambda x:len(x))

df.head(10)


# <details>
#     <summary style="background: #e6eaeb; padding: 4px 0; text-align: center; font-size: 20px; font-weight: 900;"> 👉 Kliknij tutaj (1 klik), aby zobaczyć podpowiedź 👈 </summary>
# <p>
# Długość w Python mierzy się przy pomocy funkcji len, np. len("Abc").
# <details>
#     <summary style="background: #e6eaeb; padding: 4px 0; text-align: center; font-size: 20px; font-weight: 900;"> 👉 Kliknij tutaj (1 klik), aby zobaczyć odpowied 👈 </summary>
# <p>
# 
# ```python
# df['len_name'] = df['name'].map(lambda x: len(x))
# ```
# 
# </p>
# </details>
# </p>
# </details> 


# In[24]:


#na wszelki wypadek odpowiedź na zadanie 1.3.1
df['len_name'] = df['name'].map(lambda x: len(x))


# Teraz przygotujmy `X`, `y` i wytrenujmy pierwszy model.

# In[25]:


X = df[ ['len_name'] ].values
y = df['target'].values

model = DummyClassifier(strategy = 'stratified')
model.fit(X, y)
y_pred = model.predict(X)

# In[26]:


df['gender_pred'] = y_pred
df['gender_pred'].value_counts()

# In[27]:

df[ df.target != y_pred ].shape # błędna odpowiedź

# In[28]:

accuracy_score(y, y_pred)

# In[29]:


model = DummyClassifier(strategy = 'stratified', random_state=0)
model.fit(X, y)
y_pred = model.predict(X)
accuracy_score(y, y_pred)

# In[30]:


model = LogisticRegression(solver='lbfgs')
model.fit(X, y)
y_pred = model.predict(X)
accuracy_score(y, y_pred)

# In[31]:


df['gender_pred'] = y_pred
df['gender_pred'].value_counts()

# In[32]:


y_pred = [1]*X.shape[0] #ilość jedynek powinna zgadzać się z ilością wierszy w macierzy X
accuracy_score(y, y_pred)


# In[33]:


def train_and_predict_model(X, y, model, success_metric=accuracy_score):
    model.fit(X, y)
    y_pred = model.predict(X)
    
    print("Distribution:")
    print( pd.Series(y_pred).value_counts() )
    
    return success_metric(y, y_pred)


# In[34]:


vowels = ['a', 'ą', 'e', 'ę', 'i', 'o', 'u', 'y']

def how_many_vowels(name):
    count = sum( map(lambda x: int(x in vowels), name.lower()) )
    
    return count

#how_many_vowels('Jana')

df['count_vowels'] = df['name'].map(how_many_vowels)
train_and_predict_model(df[['len_name', 'count_vowels'] ], y, LogisticRegression(solver='lbfgs'))


# In[35]:


def first_is_vowel(name):
    return name.lower()[0] in vowels

#first_is_vowel('Ada')

df['first_is_vowel'] = df['name'].map(first_is_vowel)

train_and_predict_model(df[['len_name', 'first_is_vowel'] ], y, LogisticRegression(solver='lbfgs'))


# In[36]:


X = df[['len_name', 'count_vowels', 'first_is_vowel'] ]
train_and_predict_model(X, y, LogisticRegression(solver='lbfgs'))


# In[37]:


pd.factorize(['blue', 'green', 'yellow', 'blue'])


# In[38]:


pd.factorize(['blue', 'green', 'yellow', 'blue'])[0]


# ## Zadanie 1.3.2
# 
# Napisz podobny kod jak wyżej, tylko wyciągnij ostatnią literę jako cechę (zamiast pierwszej).

# In[47]:


df['last_letter'] = df['name'].map(lambda x: x.lower()[-1])
df['last_letter_cnt'] = df['last_letter'].factorize()[0]

X = df[['len_name','count_vowels','first_is_vowel','last_letter_cnt']]

train_and_predict_model(X, y, LogisticRegression(solver='lbfgs'))


# <details>
#     <summary style="background: #e6eaeb; padding: 4px 0; text-align: center; font-size: 20px; font-weight: 900;"> 👉 Kliknij tutaj (1 klik), aby zobaczyć podpowiedź 👈 </summary>
# <p>
# Musisz utworzyć nowy atrybut last_letter, a następnie last_letter_cnt
# <details>
#     <summary style="background: #e6eaeb; padding: 4px 0; text-align: center; font-size: 20px; font-weight: 900;"> 👉 Kliknij tutaj (1 klik), aby zobaczyć odpowiedź 👈 </summary>
# <p>
# 
# ```python
# df['last_letter'] = df['name'].map(lambda x: x.lower()[-1])
# df['last_letter_cnt'] = df['last_letter'].factorize()[0]
# 
# X = df[['len_name', 'count_vowels', 'first_is_vowel', 'last_letter_cnt'] ]
# train_and_predict_model(X, y, LogisticRegression(solver='lbfgs'))
# ```
#  
# </p>
# </details>
# </p>
# </details> 

# ## Kolejne pomysły na nowe cechy
# 1. Pobierzmy wszystkie samogłoski (ang. *vowels*), zakładając, że to może mieć wpływ. Na przykład **Sławomir** ma trzy samogłoski w tej kolejności: **aoi**, natomiast **Patrycja** również ma trzy samogłoski, ale inna kombinacja: **aya**. Dla każdej kombinacji pojawi się unikalny ID.
# 2. Zróbmy podobnie, tylko tym razem spółgłoski (ang. *consonants*).

# In[40]:


def get_all_vowels(name):
    all_vowels = [letter for letter in name.lower() if letter in vowels]
    
    return ''.join(all_vowels)
    
#get_all_vowels('Sławomir')

df['all_vowels'] = df['name'].map(get_all_vowels)
df['all_vowels_cnt'] = pd.factorize(df['all_vowels'])[0]


X = df[['len_name', 'count_vowels', 'first_is_vowel', 'first_letter_cnt', 'all_vowels_cnt'] ]
train_and_predict_model(X, y, LogisticRegression(solver='lbfgs'))


# In[41]:


def get_all_consonants(name):
    all_consonants = [letter for letter in name.lower() if letter not in vowels]
    
    return ''.join(all_consonants)
    
#get_all_consonants('Sławomir')

df['all_consonants'] = df['name'].map(get_all_consonants)
df['all_consonants_cnt'] = pd.factorize(df['all_consonants'])[0]

X = df[['len_name', 'count_vowels', 'first_is_vowel', 'first_letter_cnt', 'all_consonants_cnt'] ]
train_and_predict_model(X, y, LogisticRegression(solver='lbfgs', max_iter=200))


def last_is_vowel(name):
    return name.lower()[-1] in vowels

#last_is_vowel('Ada')

df['last_is_vowel'] = df['name'].map(last_is_vowel)

X = df[['last_is_vowel'] ]
train_and_predict_model(X, y, LogisticRegression(solver='lbfgs', max_iter=200))


# In[43]:


feats = ['last_is_vowel', 'len_name', 'count_vowels', 'first_is_vowel', 'all_vowels_cnt', 'all_consonants_cnt']
X = df[ feats ]
train_and_predict_model(X, y, LogisticRegression(solver='lbfgs', max_iter=200))

# In[44]:


df.columns


# In[45]:


df['lst_letter_a'] = df.name.map(lambda x: x[-1] == 'a')

df[ (df.gender == 'm') & df.lst_letter_a ]


# In[46]:


df[ (df.gender == 'f') & (~df.lst_letter_a) ]
