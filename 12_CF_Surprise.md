# Collaborative Filtering using Surprise Library

ข้อมูลในตัวอย่างนี้ใช้จากไฟล์ [Raitngs](<Datasets/Recommender System/MovieLen_Rec/ratings.csv>) และ [Movies](<Datasets/Recommender System/MovieLen_Rec/movies.csv>)

การทำ Colaborative Filtering อาจแบ่งได้เป็น 3 ลักษณะหลักๆ ดังนี้
1. Memory-based Collaborative Filtering
2. Model-based Collaborative Filtering
3. Hybrid Models

ในเอกสารนี้จะเน้นแสดงตัวอย่างโค้ดตัวอย่างการทำ Memory-based Collaborative Filtering โดยใช้ Library ชื่อว่า Surprise ซึ่งเป็น Library ที่ใช้ในการทำ Recommendation System โดยเฉพาะในการทำ Collaborative Filtering และเพิ่มเติมตัวอย่างโค้ดสำหรับการทำ Model-based Collaborative Filtering โดยวิธี Singular Value Decomposition (SVD) ซึ่งเป็นวิธีที่ใช้ในการทำ Matrix Factorization

## 1. Memory-based Collaborative Filtering
ตัวอย่างโค้ดการใช้ Memory-based Collaborative Filtering ซึ่งจะแบ่งออกเป็น 2 ประเภท คือ User-based Collaborative Filtering และ Item-based Collaborative Filtering

### 1.1 User-based และ Item-based Collaborative Filtering

#### 1.1.1 การติดตั้ง Library Surprise
```python
!pip install scikit-surprise
```

#### 1.1.2 Import Library ที่จำเป็น
```python
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd
import matplotlib.pyplot as plt
```

#### 1.1.3 โหลดข้อมูล Dataset จากไฟล์ `ratings.csv`
```python
reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 5), skip_lines=1)
data = Dataset.load_from_file('ratings.csv', reader=reader)

# แบ่งข้อมูลเป็น trainset และ testset
trainset, testset = train_test_split(data, test_size=0.25)
```

#### 1.1.4 สำรวจข้อมูลเบื้องต้น
```python
print('Number of Users:', trainset.n_users)
print('Number of Items:', trainset.n_items)
print('Max Rating:', trainset.rating_scale[1])
print('Min Rating:', trainset.rating_scale[0])
print('Number of Ratings:', trainset.n_ratings)
```

#### 1.1.5 แสดงการกระจายของ Rating 

```python
import matplotlib.pyplot as plt

ratings = [0, 0, 0, 0, 0]
for rating in trainset.all_ratings():
    ratings[int(rating[2])-1] += 1

plt.bar(['1', '2', '3', '4', '5'], ratings)
plt.xlabel('Rating')
plt.ylabel('Number of Ratings')
plt.title('Distribution of Ratings')
plt.show()
``` 

#### 1.1.6 แสดงข้อมูลเกี่ยวกับหนังจาก `movies.csv`
```python
df_movies = pd.read_csv('movies.csv')
df_movies.drop(['release_date', 'video_release_date', 'imdb_url'], axis=1, inplace=True)
df_movies.head()
```

#### 1.1.7 Mapping รหัสหนังให้ตรงกับ `movies.csv`
```python
# สร้าง mapping ระหว่าง inner_id ของ Surprise กับ movie_id จริง
item_raw_ids = {trainset.to_inner_iid(str(movie_id)): movie_id for movie_id in df_movies['movie_id'].unique() if str(movie_id) in trainset._raw2inner_id_items}
```

#### 1.1.8 แสดงข้อมูล Rating ใน Pivot Table โดยใช้ `movie_id` จาก `movies.csv`
```python
df = pd.DataFrame(trainset.all_ratings(), columns=['User', 'Item', 'Rating'])
df['User'] = df['User'].map(lambda x: trainset.to_raw_uid(x))
df['Item'] = df['Item'].map(lambda x: trainset.to_raw_iid(x))
pivot_table = df.pivot_table(index='User', columns='Item', values='Rating')
pivot_table.head()
```

#### 1.1.9 แสดงชื่อหนังที่ได้ Rating สูงสุด 10 อันดับแรก
```python
item_ratings = {}
for user, item, rating in trainset.all_ratings():
    if item not in item_ratings:
        item_ratings[item] = []
    item_ratings[item].append(rating)

item_ratings = {item: (sum(ratings) / len(ratings), len(ratings)) for item, ratings in item_ratings.items()}
sorted_items = sorted(item_ratings.items(), key=lambda x: x[1][0], reverse=True)

for item_inner_id, (rating, count) in sorted_items[:10]:
    movie_id = trainset.to_raw_iid(item_inner_id)
    movie_title = df_movies.loc[df_movies['movie_id'] == int(movie_id), 'title'].values[0]
    print('Movie:', movie_title, 'Rating:', rating, 'Number of Ratings:', count)
```
#### 1.1.10 แสดงชื่อหนังที่ได้ Rating สูงสุด 10 อันดับแรก โดยเลือกเฉพาะหนังที่ได้ Rating มากกว่า 100 ครั้ง
```python
min_user_rate=100
top_x=10

item_ratings = {}
for user, item, rating in trainset.all_ratings():
    if item not in item_ratings:
        item_ratings[item] = []
    item_ratings[item].append(rating)

item_ratings = {item: (sum(ratings) / len(ratings), len(ratings)) for item, ratings in item_ratings.items()}
sorted_items = sorted(item_ratings.items(), key=lambda x: x[1][0], reverse=True)

filtered_items = [item for item in sorted_items if item[1][1] > min_user_rate][:top_x]
for item_inner_id, (rating, count) in filtered_items:
    movie_id = trainset.to_raw_iid(item_inner_id)
    movie_title = df_movies.loc[df_movies['movie_id'] == int(movie_id), 'title'].values[0]
    print('Movie:', movie_title, 'Rating:', rating, 'Number of Ratings:', count)
```

#### 1.1.11 แสดงชื่อหนังที่ได้ Rating น้อยสุด 10 อันดับแรก โดยเลือกเฉพาะหนังที่ได้ Rating มากกว่า 100 ครั้ง
```python
min_user_rate=100
top_x=10

lowest_filtered_items = [item for item in sorted_items if item[1][1] > min_user_rate][-top_x:]
for item_inner_id, (rating, count) in lowest_filtered_items:
    movie_id = trainset.to_raw_iid(item_inner_id)
    movie_title = df_movies.loc[df_movies['movie_id'] == int(movie_id), 'title'].values[0]
    print('Movie:', movie_title, 'Rating:', rating, 'Number of Ratings:', count)
```

#### 1.1.12 สร้างโมเดล User-based Collaborative Filtering
```python
model = KNNBasic(k=3, sim_options={'name': 'pearson', 'user_based': True})
model.fit(trainset)
```

#### 1.1.13 สร้างโมเดล Item-based Collaborative Filtering
```python
model = KNNBasic(k=3, sim_options={'name': 'pearson', 'user_based': False})
model.fit(trainset)
```

#### 1.1.14 ทดสอบโมเดลด้วย `testset`
```python
predictions = model.test(testset)
accuracy.rmse(predictions)
```

#### 1.1.15 ทำนายค่า Rating ของผู้ใช้รหัส 196 สำหรับหนังรหัส 314
```python
inner_id = trainset.to_inner_iid(str(314))  # แปลง movie_id เป็น inner_id
prediction = model.predict(196, trainset.to_raw_iid(inner_id))  # ทำนาย rating
print(f'Predicted rating for user 196 and movie 314: {prediction.est}')
```

#### 1.1.16 แสดงผลทำนาย Rating จาก model 
อ้างอิงโค้ด : [Building and Testing Recommender Systems With Surprise, Step-By-Step](https://medium.com/towards-data-science/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b)
```python
def get_Iu(uid):
    """ return the number of items rated by given user
    args: 
      uid: the id of the user
    returns: 
      the number of items rated by the user
    """
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError: # user was not part of the trainset
        return 0
    
def get_Ui(iid):
    """ return number of users that have rated given item
    args:
      iid: the raw id of the item
    returns:
      the number of users that have rated the item.
    """
    try: 
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:
        return 0
    

predictions = model.test(testset)
df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
#['user', 'movie', 'rating_actual', 'rating_predict', 'details']
df['Iu'] = df.uid.apply(get_Iu)
df['Ui'] = df.iid.apply(get_Ui)
df['err'] = abs(df.est - df.rui)

best_predictions = df.sort_values(by='err')[:10]
worst_predictions = df.sort_values(by='err')[-10:]
```

## 2. Model-based Collaborative Filtering
Model-based Collaborative Filtering จะใช้วิธีการทำ Matrix Factorization โดยใช้ Singular Value Decomposition (SVD) ซึ่งเป็นวิธีที่ใช้ในการทำ Matrix Factorization โดยใช้ Singular Value Decomposition (SVD)

### 2.1 Singular Value Decomposition (SVD)

อ่านเพิ่มเติมสำหรับการทำงานของ SVD ได้ที่:
* [https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD)
* [Various Implementations of Collaborative Filtering](https://medium.com/@Sumeet_Agrawal/various-implementations-of-collaborative-filtering-7429eec37ab9)

#### 2.1.1 สร้างโมเดล SVD
```python
from surprise import SVD

model = SVD()
model.fit(trainset)
```

#### 2.1.2 ทดสอบโมเดลด้วย `testset`
```python
predictions = model.test(testset)
accuracy.rmse(predictions)
```

#### 2.1.3 แนะนำหนัง 10 อันดับแรกสำหรับผู้ใช้รหัส 196
อ้างอิงที่มาโค้ด (มีการปรับปรุงบางส่วน) : [https://surprise.readthedocs.io/en/stable/FAQ.html#how-to-get-the-top-n-recommendations-for-each-user](https://surprise.readthedocs.io/en/stable/FAQ.html#how-to-get-the-top-n-recommendations-for-each-user)

```python
user_x='196'
# Get all unique movie IDs from the raw dataset
all_movie_ids = set(data.raw_ratings[i][1] for i in range(len(data.raw_ratings)))

# Convert user ID to the internal representation
inner_user_id = trainset.to_inner_uid(user_x)

# Get all movie IDs rated by user x (in the internal format)
movies_rated_by_user_x = set(trainset.ur[inner_user_id])

# Filter movies to include only those present in the trainset
movies_to_predict = [
    movie_id for movie_id in all_movie_ids
    if movie_id in trainset._raw2inner_id_items and trainset.to_inner_iid(movie_id) not in movies_rated_by_user_x
]

# Make predictions for user x
predictions = [model.predict(user_x, movie_id) for movie_id in movies_to_predict]

# Get top N recommendations
top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:10]

# Display recommendations
print("Top 10 Recommendations for User", user_x ,":")
for pred in top_n:
    print(f"Movie ID: {pred.iid}, Estimated Rating: {pred.est}")

```

## การเปรียบเทียบโมเดลหลายแบบโดยใช้ Cross Validation
ที่มา : [Building and Testing Recommender Systems With Surprise, Step-By-Step](https://medium.com/towards-data-science/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b)
```python
import pandas as pd
from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
from surprise.model_selection import cross_validate

# Initialize an empty list to store benchmark results
benchmark = []

# Iterate over all algorithms
for algorithm in [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]:
    # Perform cross validation
    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
    
    # Get results & append algorithm name using pd.concat()
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = pd.concat([tmp, pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm'])])
    benchmark.append(tmp)

# Convert results to DataFrame and sort by RMSE
df_results = pd.DataFrame(benchmark)
df_results.set_index('Algorithm', inplace=True)
df_results.sort_values('test_rmse')
```

