#!/usr/bin/env python
# coding: utf-8

# # Predykcja płci po imieniu
# 
# ### Celem jest zrobienie prostej, ale już wartościowej predykcji.
# 
# * [pandas](https://bit.ly/3sy04Jw) - biblioteka do wczytania i manipulacji danymi
# * [numpy](https://bit.ly/2Pe9A65) - biblioteka do pracy z wektorami/macierzami, pandas wewnątrz również używa `numpy`
# * [sklearn](https://bit.ly/3fzXLlF) - biblioteka, która zawiera konkretne implementacje algorytmów uczenia maszynowego (wymawia się *[saɪ-kit-lə:n]*, to jest skrócona wersja od `"science-kit-learn"`)

# ### Krok po kroku 
# 
# Jeśli wolisz najpierw słuchać i oglądać, to obejrzyj nagranie poniżej, które omawia tę lekcję. 

# In[1]:


get_ipython().run_cell_magic('html', '', '<iframe style="height:500px;width:100%" src="https://bit.ly/3swlfLZ" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>')


# In[15]:


import pandas as pd

#modele (algorytmy)
from sklearn.dummy import DummyClassifier           # <== Najprostszy możliwy model 
from sklearn.linear_model import LogisticRegression # <== Regresja logistyczna (liniowa)

#metryka sukcesu
from sklearn.metrics import accuracy_score


# **Uwaga!** nazwa modułu (*LogisticRegression*) wskazuje, że jest to regresja logistyczna, natomiast to jest podklasa regresji liniowej (czyli pod spodem jest zwykła regresja liniowa + dodatkowa funkcja na końcu).

# ## Wczytujemy dane
# 
# Dane są w formacie `.csv`, `pandas` umożliwia w jednym wierszu wczytanie danych w formacie `.csv`: `.read_csv()`. Po uruchomieniu tej linii `df` będzie zawierać dane wczytane z pliku w postaci tabelarycznej (czyli wiersze i kolumny).

# In[16]:


df = pd.read_csv("../input/polish_names.csv")
df.head()


# ## Sprawdzamy dane
# 
# Na początek chcemy wiedzieć bardzo proste rzeczy:
# 1. Ile jest wierszy (wszystkich obiektów)?
# 2. Ile jest kolumn (cech obiektów)?
# 3. Która zmienna jest zmienną docelową (ang. *target variable*)?
# 4. Jaki problem jest do rozwiązania (klasyfikacja czy regresja)?
# 5. W przypadku klasyfikacji, ile (dwie czy więcej) i jakie unikalne wartości ma zmienna docelowa?
# 6. Jak wygląda rozkład unikalnych wartości zmiennej docelowej (czy jest mniej więcej po równo, czy jednak są bardzo popularne/rzadkie klasy)?
# 7. Czy są brakujące dane?

# In[17]:


df.info()


# - Druga linia "mówi": `1705 entries`, to jest ilość wierszy (obiektów).
# - Trzecia linia "mówi": `total 2 columns`, co oznacza, że mamy 2 kolumny (cechy).
# - Następnie mamy informację o każdej kolumnie i liczbę wartości (`non-null`). 
# - Jeśli kolumna X ma mniej `non-null` wierszy niż całość, to oznacza, że dla tej cechy mamy brakujące wartości (ang. *missing data*), z którymi trzeba będzie sobie "jakoś" poradzić.
# - W naszym przypadku (na początku) wszystko jest bardzo proste. Mamy wszystkie wartości i tylko jedną cechę - **imię**. A druga kolumna to jest zmienna docelowa (eng. *target variable*), czyli czy imię jest **męskie** czy **żeńskie** (tylko dwie wartości, więc klasyfikacja binarna). 
# - Ostatnia linia `memory usage` mówi, ile pamięci RAM zużywa, w tym przypadku bardzo mało (jedynie 26.7 KB).

# ## Jak wyglądają dane?
# Zobacz 10 losowych wierszy.

# In[18]:


df.sample(10)


# - Kolumna `name` zawiera imię i czasem są dość ciekawe :).
# - Kolumna `gender` zawiera płeć, gdzie **`m`** oznacza imię męskie a **`f`** - imię żeńskie
# 
# Sprawdźmy, jaki jest rozkład imion **męskich** i **żeńskich**.

# In[19]:


df['gender'].value_counts()


# - Męskich imion jest prawie 2 razy więcej (**1033** do **672**).
# - Dalej będzie widać, czy jest to dla nas jakiś problem (np. przez to, że imion żeńskich jest mniej, jakość modelu jest gorsza. Jeśli tak, to będziemy później myśleć, co z tym zrobić).
# 
# Pamiętasz, że model oczekuje na reprezentację liczbową zamiast słowną? Teraz mamy transformować: `m => 1, f => 0`.
# 
# Pomoże nam w tym funkcja [`map`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.map.html). Żeby lepiej zrozumieć, jak działa funkcja .map(), zróbmy to w kilku krokach.
# 
# Funkcja `transform_string_into_number` zwraca to samo, co dostała, to tak zwana funkcja [funkcja tożsamościowa](https://pl.wikipedia.org/wiki/Funkcja_to%C5%BCsamo%C5%9Bciowa). Robimy to po to, żeby poznać składnię.

# In[20]:


def transform_string_into_number(string):
    return string
    
df['gender'].head().map( transform_string_into_number )


# Teraz dodajmy logikę do funkcji `transform_string_into_number`

# In[21]:


def transform_string_into_number(string):
    return int(string == 'm')
    
df['gender'].head().map( transform_string_into_number )


# Użyjmy teraz anonimowej funkcji (*lambda*), żeby zmniejszyć ilość kodu. Wynik mapowania przypisujemy do nowej kolumny o nazwie `target`.
# 
# *Zwróć uwagę*, że *lambda* nie ma słowa kluczowego `return`, bo to z definicji ma być jednowierszowa logika (wynik, który zostanie zwrócony).

# In[22]:


df['target'] = df['gender'].map( lambda x: int(x == 'm') )
df.head(10)


# ## Feature engineering
# Dodajmy pierwszą cechę, np. długość imienia. Załóżmy, że ilość literek może wpłynąć na predykcję, czy imię jest męskie czy żeńskie.
# 
# Dlaczego akurat tak? Od czegoś musimy zacząć i to jest jedna z prostszych cech, którą można wnioskować na podstawie słowa. Czy jest skuteczna? Właśnie to chcemy sprawdzić.
# 
# ![](../images/len_fi.png)

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

# ### 🤝🗣️ Współpraca 💪 i komunikacja 💬
# 
# - 👉 [#pml_module1](https://practicalmlcourse.slack.com/archives/C045CNLNH89) - to jest miejsce, gdzie można szukać pomocy i dzielić się doświadczeniem - także pomagać innym 🥰. 
# 
# Jeśli masz pytanie, to staraj się jak najdokładniej je sprecyzować, najlepiej wrzuć screen z twoim kodem i błędem, który się pojawił ✔️
# 
# - 👉 [#pml_module1_done](https://practicalmlcourse.slack.com/archives/C045CP89KND) - to miejsce, gdzie możesz dzielić się swoimi przerobionymi zadaniami, wystarczy, że wrzucisz screen z #done i numerem lekcji np. *#1.2.1_done*, śmiało dodaj komentarz, jeśli czujesz taką potrzebę, a także rozmawiaj z innymi o ich rozwiązaniach 😊 
# 
# - 👉 [#pml_module1_ideas](https://practicalmlcourse.slack.com/archives/C044TFZLF1U)- tutaj możesz dzielić się swoimi pomysłami
# 

# ## Pierwszy model
# - Zróbmy nasz pierwszy model (eng. *basic model*), który będzie dość prosty, wręcz "głupi", zresztą ma on taką nazwę `DummyClassifier`.
# - Bardzo polecam zaczynać od czegoś bardzo prostego, to pomoże zrozumieć, gdzie jesteś teraz i mieć pierwszy wynik, z którym możemy się porównywać (np. jeśli później spędzimy kilka tygodni robiąc coś bardziej zaawansowanego, a być może warto było zastosować tylko najprostszy model?)
# - Idea polega na tym, że model patrzy tylko na zmienną docelową, jak często występują te czy inne klasy (w naszym przypadku imię męskie lub żeńskie).
# 
# 
# ## Przygotujmy dane
# Metoda odpowiedzialna do trenowania modelu ma nazwę: **`fit`** i oczekuje ona 2 argumentów:
# - Pierwszy argument to jest **macierz/tablica** cech (**Uwaga**: cecha może być jedna, ale to nadal ma być tablica, nie wektor!)
# - Drugi argument to **wektor** zmiennej docelowej (eng. *target variable*)
# 
# 
# **Podpowiedź**:
# - `[1, 2, 3, 4, 5]` => to jest wektor
# - `[[1], [2], [3], [4], [5]]` => to jest wektor wektorów, czyli macierz/tablica (w tym przypadku tylko z jedną cechę dla każdego obiektu)
# - `[[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]` => to jest wektor wektorów, czyli macierz (w tym przypadku dwie cechy dla każdego obiektu)

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


# Jak już masz tablicę **X** (cechy dla naszych obiektów) i wektor **y** (odpowiedzi dla obiektów lub zmienną docelową [eng. *target variable*]), to już możemy zacząć budować model.
# 
# Ten proces składa się z prostych trzech kroków:
# 1. Wybór modelu (algorytmu) i utworzenie instancji
# 2. Trenowanie modelu (podając X i y) => **`fit(X_train, y_train)`**
# 3. Predykcja modelu (w tym przypadku podajemy tylko cechy, bo odpowiedź zwraca model) => **`predict(X_test)`**
# 
# *Swoją drogą*, zwykle odpowiedź z modelu jest przypisywana do zmiennej `y_pred` (oczywiście możesz tą zmienną nazwać, jak tylko chcesz), ale polecam trzymać się tej konwencji.
# 
# Teraz możemy przypisać `y_pred` do nowej kolumny i zobaczyć, ile przydzielił imion męskich, a ile żeńskich.

# In[26]:


df['gender_pred'] = y_pred
df['gender_pred'].value_counts()


# Zobaczmy teraz, w ilu przypadkach model podał inną odpowiedź, niż była w rzeczywistości.

# In[27]:


df[ df.target != y_pred ].shape # błędna odpowiedź


# Pamiętaj, że `1` oznacza imię męskie oraz `0` oznacza imię żeńskie.
# 
# Zwróć uwagę, w ilu przypadkach (`df[ df.target != y_pred ].shape`) z 1705 model pomylił się. Model był o tyle "mądry", że jedynie uwzględnił ówczesny rozkład (przypomnę, że było 1033 vs 672) i to dlatego uznał, że imię męskie ma występować częściej. Oczywiście takie podejście jest błędne... ale już można wyciągnąć ciekawe wnioski o tym, jak łatwo zniekształcić rzeczywistość modelu podając pewne dane częściej lub rzadziej. 
# 
# *Swoją drogą* ciekawy [artykuł](https://medium.com/@angebassa/data-alone-isnt-ground-truth-9e733079dfd4) o tym, że dane nie są prawdą absolutną.
# 
# Kolejnym krokiem jest zmierzenie jakości. Dla uproszczenia będziemy patrzeć na `accuracy`, czyli dokładność naszego modelu (w tej chwili opuścimy inne możliwe metryki, żeby uprościć początek).

# In[28]:


accuracy_score(y, y_pred)


# Mamy ok. 50%, wynik jest bardzo bliski do losowego (50% zawsze możemy osiągnąć, po prostu podrzucając monetę).
# 
# **Uwaga!** `accuracy_score` sprawdza, jak wiele wartości dla wektora `y_pred` pokrywa się z wektorem `y` i pokazuje wynik w procentach. Więcej o metrykach będzie w następnym module.

# ### Losowość
# `DummyClassifier` ignoruje cechy i zwraca zawsze ten sam wynik, o ile ustawisz `random_state`. Jeśli nie ustawisz `random_state`, to wynik za każdym razem będzie się trochę różnił (możesz to sprawdzić).

# In[29]:


model = DummyClassifier(strategy = 'stratified', random_state=0)
model.fit(X, y)
y_pred = model.predict(X)
accuracy_score(y, y_pred)


# ## Liniowy model
# 
# Użyjmy teraz modelu liniowego `LogisticRegression` (pamiętaj, że regresja logistyczna to jest regresja liniowa + na końcu funkcja binarna, która zwraca 0 lub 1).
# 
# Jest wiele parametrów, które można sprecyzować dla modelu. W tym przypadku zdefiniujemy tylko `solver`, czyli algorytm, który jest wykorzystywany do obliczania modelu. Na tym etapie nie ma dużego znaczenia, który wybierzemy, więc użyjemy domyślnego dla `LogisticRegression`.
# 
# *Swoją drogą* nazwa `LogisticRegression` jest dość myląca, bo sama nazwa wskazuje na robienie regresji, jednak w rzeczywistości robi się klasyfikację. Skąd taka nazwa? Jak to często bywa w życiu, są na to pewne powody historyczne :D.
# 
# Może jeszcze raz powtórzę, na wszelki wypadek `LogisticRegression` to jest liniowy model dla **klasyfikacji** (nie regresji). Nazwa jest jaka jest, warto zapamiętać :).

# In[30]:


model = LogisticRegression(solver='lbfgs')
model.fit(X, y)
y_pred = model.predict(X)
accuracy_score(y, y_pred)


# Jak widać, jakość modelu już jest lepsza. Udało się nam osiągnąć **~61%** dokładności. Sprawdźmy, jak wygląda rozkład odpowiedzi.

# In[31]:


df['gender_pred'] = y_pred
df['gender_pred'].value_counts()


# To oznacza, że model zawsze zwrócił `1` (każde imię to imię męskie), bo akurat ta klasa była bardziej popularna. Zróbmy eksperyment, jeśli manualnie przypiszesz zawsze odpowiedź `1`,
# to dostaniesz ten sam wynik.

# In[32]:


y_pred = [1]*X.shape[0] #ilość jedynek powinna zgadzać się z ilością wierszy w macierzy X
accuracy_score(y, y_pred)


# Dlaczego tak się dzieje?
# 
# Na obecnych cechach model liniowy nie potrafił się lepiej nauczyć i uznał, że takie podejście jest najbardziej rozsądne. 
# 
# Dlaczego `accuracy` jest ok. 61% przy tak głupim podejściu? To wynika ze słabości tej metryki, która bardzo zależy od rozkładu (więcej o tym w drugim module).
# 
# Zaraz dodamy kolejną cechę, ale już można zauważyć, że poprzednia komórka składająca się z 4 linijek kodu będzie się powtarzać. To oznacza, że warto zrobić osobną funkcję, żeby ułatwić sobie życie w przyszłości. Niech to będzie funkcja o nazwie: `train_and_predict_model`.

# In[33]:


def train_and_predict_model(X, y, model, success_metric=accuracy_score):
    model.fit(X, y)
    y_pred = model.predict(X)
    
    print("Distribution:")
    print( pd.Series(y_pred).value_counts() )
    
    return success_metric(y, y_pred)


# **Uwaga!** 
# Możemy sobie wywołać `success_metric(y, y_pred)`, co przy wcześniejszej deklaracji `success_metric=accuracy_score` oznacza, że `accuracy_score` dostanie te same parametry, które przekazaliśmy do `success_metric`. Python umożliwia przekazywanie parametrów domyślnych do funkcji w taki sposób (co nie jest możliwe np. w takich językach jak Java czy PHP, ale jest normalne dla wszystkich języków funkcyjnych).

# ## Cechy
# Popracujemy nad samogłoskami. Być może ich liczba i kolejność wpływa na to, czy jest to imię męskie czy żeńskie.

# In[34]:


vowels = ['a', 'ą', 'e', 'ę', 'i', 'o', 'u', 'y']

def how_many_vowels(name):
    count = sum( map(lambda x: int(x in vowels), name.lower()) )
    
    return count

#how_many_vowels('Jana')

df['count_vowels'] = df['name'].map(how_many_vowels)
train_and_predict_model(df[['len_name', 'count_vowels'] ], y, LogisticRegression(solver='lbfgs'))


# Udało się polepszyć wynik o 10 punktów procentowych! Bardzo dobrze,  próbujmy dalej. Nowa cecha będzie sprawdzać, czy pierwsza litera jest samogłoską czy nie.
# 
# Zwróć uwagę, że rozkład odpowiedzi już jest w miarę sensowny **1082** vs **623** (nie tylko same "1", czyli imiona męskie).

# In[35]:


def first_is_vowel(name):
    return name.lower()[0] in vowels

#first_is_vowel('Ada')

df['first_is_vowel'] = df['name'].map(first_is_vowel)

train_and_predict_model(df[['len_name', 'first_is_vowel'] ], y, LogisticRegression(solver='lbfgs'))


# Jak widać, ta cecha w ogóle nie wpłynęła na jakość modelu... To jest normalnie. Tak naprawdę dość często będziemy próbować różnych pomysłów i większość z nich może nie działać. Trzeba być na to przygotowanym i żyć wg zasady: `Fail fast, learn faster`. 
# 
# Zwróć uwagę, że tym razem model zwrócił tylko 1 (imię męskie), czyli nie potrafił "wymyślić" nic lepszego. To oznacza, że cecha "czy pierwsza litera to samogłoska?" jest bezużyteczna (dla modelu liniowego).
# 
# Idziemy dalej. Sprawdźmy teraz razem trzy cechy: długość imienia, ilość samogłosek oraz czy pierwsza litera to samogłoska.

# In[36]:


X = df[['len_name', 'count_vowels', 'first_is_vowel'] ]
train_and_predict_model(X, y, LogisticRegression(solver='lbfgs'))


# Udało się ulepszyć model o kolejne **1.5%** (**0.714** vs **0.729**). Bardzo dobrze, idziemy dalej.
# 
# Tylko najpierw poznajmy lepiej funkcję `.factorize()`.

# In[37]:


pd.factorize(['blue', 'green', 'yellow', 'blue'])


# Jak widzisz, `pd.factorize()` zwróciła tuple z dwoma wynikami.
# - pierwsze to są unikalne ID `array([0, 1, 2, 0])`
# - drugi to etykietki do ID'ków, zobacz `blue=0` lub `yellow=2` (czyli `yellow`  ma indeks dwa w tablice `['blue', 'green', 'yellow']`)
# 
# W naszym przypadku będzie trzeba przekazać ID'ki dla modelu, czyli potrzebujemy tylko pierwszą część wyniku:
# `pd.factorize(['blue', 'green', 'yellow', 'blue'])[0]`. Zwróć uwagę, że na końcu pojawiło się `[0]`.

# In[38]:


pd.factorize(['blue', 'green', 'yellow', 'blue'])[0]


# Funkcję `.factorize()` możemy zrobić w taki sposób: `pd.factorize()` lub w taki `df['new_column'].factorize()` wynik działania będzie identyczny, ale druga wersja czasem jest wygodniejsza w pisaniu.
# 
# Wróćmy do naszych cech, czyli przypiszmy każdej literce unikalny ID.

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


# Trochę lepiej (zwłaszcza pierwszy pomysł z samogłoskami): **0.729** vs **0.738**. Spółgłoski trochę poprawiły model, ale mniej: **0.729** vs **0.731**. To raczej ma sens, prawda? Samogłoski mają większy wpływ na to, czy imię jest męskie czy żeńskie. 
# 
# Kontynuując myśl. Kolejna cecha, która może mieć wpływ, to jaka jest ostatnia litera. Jeśli jest to samogłoska, to raczej jest to imię żeńskie, na przykład: **Kamila** i **Kamil**, **Adriana** i **Adrian** czy **Jana** i **Jan**. Sprawdźmy to.

# In[42]:


def last_is_vowel(name):
    return name.lower()[-1] in vowels

#last_is_vowel('Ada')

df['last_is_vowel'] = df['name'].map(last_is_vowel)

X = df[['last_is_vowel'] ]
train_and_predict_model(X, y, LogisticRegression(solver='lbfgs', max_iter=200))


# Wow! Czy to widzisz? Tylko jedna cecha potrafi od razu dać tak dobry wynik - **95%**. To dlatego proces `feature engineering` jest takim ważnym procesem. 
# 
# Musisz się przyzwyczaić, że najpierw trzeba się namęczyć, ale jest szansa, że właśnie dzięki metodzie prób i błędów wymyślisz bardzo sensowną cechę.
# 
# *Swoją drogą*, czy możesz przypomnieć sobie męskie imię, które kończy się na "a"?

# In[43]:


feats = ['last_is_vowel', 'len_name', 'count_vowels', 'first_is_vowel', 'all_vowels_cnt', 'all_consonants_cnt']
X = df[ feats ]
train_and_predict_model(X, y, LogisticRegression(solver='lbfgs', max_iter=200))


# Pochwal się na Slacku w kanale #pml_module1, że udało Ci się przejść przez lekcję 1.3 i wytrenować pierwszy model, śmiało załącz screen z wynikiem :) 

# ### Ciekawostka
# Sprawdźmy, jak często imię męskie kończy się na "a" i imię żeńskie nie kończy się na literkę "a".

# In[44]:


df.columns


# In[45]:


df['lst_letter_a'] = df.name.map(lambda x: x[-1] == 'a')

df[ (df.gender == 'm') & df.lst_letter_a ]


# Mamy 4 męskie imiona, które kończą się na literkę "a". 
# 
# *Swoją drogą*, jak pytałem ludzi czy imię Batszeba jest męskie czy żeńskie, to głosy podzieliły się mniej więcej pół na pół :). Jaka jest Twoja opinia? Czy Batszeba to imię męskie czy żeńskie i dlaczego tak uważasz? 
# 
# Sprawdźmy, ile żeńskich imion nie kończy się na literkę "a".
# 
# *Zwróć uwagę*, że znak tyldy `~` neguję znak, to oznacza, że `(~df.lst_letter_a)` jest tym samym co `(False == df.lst_letter_a)`

# In[46]:


df[ (df.gender == 'f') & (~df.lst_letter_a) ]


# Wśród tych 10 imion, które nie kończą się na "a", ile jest polskich? :)
# 
# Muszę Ci się przyznać, że ten wynik jest trochę optymistyczny. Dlaczego? To jest dobre pytanie. Zastanówmy się nad tym.
# 
# **Pamiętaj**, że trenowanie modelu i weryfikowanie go na tych samych danych, to jest zły pomysł. To jest tak samo, jak przyjdziesz na egzamin i wraz z pytaniami dostaniesz odpowiedzi :).
# 
# Ten efekt jest nazywany przeuczaniem się (ang. [*`overfitting`*](https://en.wikipedia.org/wiki/Overfitting)) i sprawia dość duże problemy w uczeniu maszynowym. Trzeba nabrać wprawy, żeby z tym sobie radzić! 
# 
# Spokojnie damy radę. Już w kolejnym ćwiczeniu pokażę Ci pierwszy sposób, jak sobie z tym radzić, a w kolejnym module spędzimy jeszcze więcej czasu, żeby to zrozumieć.
# 
# Tak jak powiedziałem, to jest jedna z największych "bolączek" w uczeniu maszynowym, która sprowadza się do pytań: "Czy mogę zaufać modelowi? Czy ten model na pewno działa (wystarczająco) dobrze?".

# ## Przydatne linki:
# * [Machine Learning 101](https://bit.ly/3lZUcGo)
# * [Machine Learning Glossary](https://bit.ly/3m280Ao)
# * [Data Alone Isn’t Ground Truth](https://bit.ly/39o3I0Q)
# * [Numerical Optimization: Understanding L-BFGS](https://bit.ly/3w115f8)
# 

# In[ ]:




