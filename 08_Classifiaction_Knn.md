# การจำแนกข้อมูล (Classification) โดยใช้ Scikit-learn ( Part 3 - K-nearest neighbors (K-NN) )


### การนำเข้า library และข้อมูล
ในการเรียกใช้ Library และข้อมูลที่ใช้ในตัวอย่างนี้ จะเหมือนกับการใช้งานในตัวอย่าง Decision Tree ดังนี้

```python
import sklearn
from sklearn import datasets
iris = datasets.load_iris()
```


### การสร้าง classifier 

การสร้าง classifier สำหรับ K-nearest neighbors ใน scikit-learn สามารถทำได้โดยใช้คำสั่งดังนี้

```python
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
```

โดยในตัวอย่างนี้เรากำหนดให้ใช้ k=3 หมายถึงจำนวนข้อมูลที่อยู่ใกล้เดียงมากที่สุด 3 ตัวแรก  และสามารถเปลี่ยนค่า k ได้ตามต้องการ โดย k 

### การใช้งาน train model
เมื่อสร้าง classifier ขึ้นมาแล้ว เราจะต้องทำการ train โมเดลก่อน โดยใช้คำสั่ง clf.fit(X, y)
โดยในตัวอย่างนี้เราจะเรียกใข้ข้อมูล iris ในการ train โมเดล ซึ่งเป็นลักษณะเดียวกับการทำงานของ Decision Tree เพียงปรับเปลี่ยนโมเดลเท่านั้น

```python
# นำเข้า Library สำหรับการทำงานกับข้อมูล
from sklearn import datasets
# นำเข้าข้อมูล iris
iris = datasets.load_iris()

# นำเข้า Library สำหรับ K-nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
# สร้างโมเดล K-nearest neighbors
clf = KNeighborsClassifier(n_neighbors=3)

# train โมเดล
clf.fit(iris.data, iris.target)
```



### ตัวอย่างโค้ดการแบ่งข้อมูลและ train model

การแบ่งข้อมูลบางส่วนสำหรับการ train โมเดล และแบ่งข้อมูลส่วนอื่น ๆ สำหรับการทดสอบโมเดล สามารถทำได้โดยใช้คำสั่ง train_test_split ใน scikit-learn โดยรายละเอียดอย่างละเอียดได้อธิบายไว้ในหัวข้อก่อนหน้านี้แล้ว โดยในตัวอย่างนี้จะแบ่งข้อมูล 70:30 

นอกจากนี้สำหรับโมเดล  K-nearest neighbors จำเป็นอย่างยิ่งที่ต้องมีการทำ Data Normalization ก่อนการ train โมเดล โดยใช้คำสั่ง StandardScaler หรือ MixMaxScaler ใน scikit-learn เนื่องจากการใช้ระยะทางระหว่างข้อมูลในการหาเพื่อนบ้านที่ใกล้ที่สุด หากมี scale ที่แตกต่างกันมากจะมีผลให้ได้ค่าที่ไม่ถูกต้อง ซึ่งขั้นตอนนี้จะทำในลักษณะเดียวกับ Naïve Bayes ที่ได้กล่าวไปแล้ว

```python
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#โหลดข้อมูล iris
iris = datasets.load_iris()

#แบ่งข้อมูลเป็น 70% สำหรับ train และ 30% สำหรับ test
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3 ,stratify=y)

#ทำ Data Normalization ด้วย StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# สร้างโมเดล K-nearest neighbors
clf = KNeighborsClassifier(n_neighbors=3)
#train โมเดล
clf.fit(X_train, y_train)


#train  โดย map  ค่า 0,1 และ 2 กับชื่อพันธุ์กล้วยไม้
#clf.fit(X_train, iris.target_names[y_train] )
```

หากต้องการใช้วิธี MinMaxScaler สามารถดูตัวอย่างได้ในหัวข้อก่อนหน้านี้



### การทำนายข้อมูลจากโมเดล และการประเมินประสิทธิภาพของโมเดล

ขั้นตอนการทำนายข้อมูลจากโมเดลและการประเมินประสิทธิภาพของโมเดล จะใช้คำสั่งเดียวกับ Decision Tree ดังนี้

**แสดงผลการทำนายของโมเดล**
```python
clf.predict(X_test)
```

**แสดงผลการทำนายและข้อมูลในรูปแบบ DataFrame**
```python
import pandas as pd

# Convert iris to pandas dataframe
df = pd.DataFrame(data=X_test, columns=iris.feature_names)

# Add target column to dataframe
df['class'] = y_test
df['predited_class'] =clf.predict(X_test)

# Map target values to target names
target_names = dict(enumerate(iris.target_names))
df['class'] = df['class'].map(target_names)
df['predited_class'] = df['predited_class'].map(target_names)

# Print dataframe
df[:10]
```


**แสดงรายละเอียดการประเมินประสิทธิภาพของโมเดล**

```python
from sklearn import metrics
print(metrics.classification_report(y_test, clf.predict(X_test)))
```


**แสดง Confusion Matrix ของโมเดล**

```python
metrics.confusion_matrix(y_test, clf.predict(X_test))
```

**แสดงค่าความถูกต้องของโมเดล**

```python
clf.score(X_test, y_test)
```

## ขั้นตอนการสร้างโมเดล (K-nearest neighbors) การจำแนกข้อมูลและทดสอบประสิทธิภาพ โดยข้อมูลจาก CSV File

ให้นักศึกษาทำการสร้างโมเดล K-nearest neighbors จากข้อมูล [ฺBreast Cancer](Datasets/breastCancer.csv) โดยทำการเตรียมข้อมูลรูปแบบเดียวกับตัวอย่างใน [การจำแนกข้อมูล (Classification) โดยใช้ Scikit-learn (Part 1 - Decision Tree)](06_Classification_DT.md) โดยเปลี่ยนจากการใช้โมเดล Decision Tree เป็น K-nearest neighborsและทำการประเมินประสิทธิภาพของโมเดลดและแสดงผลลัพธ์ที่ได้ พร้อมเปรียบเทียบประสิทธิภาพกับการใช้โมเดล decision tree ในหัวข้อก่อนหน้านี้

## แบบฝึกหัด

ให้นักศึกษาทำการสร้างโมเดล K-nearest neighbors จากข้อมูล [Titanic](Datasets/titanic.csv) 

โดยใช้โจทย์จากหัวข้อก่อนหน้านี้ [การจำแนกข้อมูล (Classification) โดยใช้ Scikit-learn (Part 1 - Decision Tree)](06_Classification_DT.md) โดยเปลี่ยนจากการใช้โมเดล Decision Tree เป็น K-nearest neighbors ทั้งนี้ในข้อที่ 7 นักศึกษาต้องแสดงผลโมเดลในรูปแบบ Tree ซึ่งไม่มีใน K-nearest neighbors จึงไม่ต้องทำในขั้นตอนนี้เพียงขั้นตอนเดียว และนำผลการวัดประสิทธิภาพที่ได้มาเปรียบเทียบกับโมเดล Decision Tree ที่ทำในหัวข้อก่อนหน้านี้
