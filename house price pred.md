```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_excel("HousePricePrediction.xlsx")

# Printing first 5 records of the dataset
print(dataset.head(5))

```

       Id  MSSubClass MSZoning  LotArea LotConfig BldgType  OverallCond  \
    0   0          60       RL     8450    Inside     1Fam            5   
    1   1          20       RL     9600       FR2     1Fam            8   
    2   2          60       RL    11250    Inside     1Fam            5   
    3   3          70       RL     9550    Corner     1Fam            5   
    4   4          60       RL    14260       FR2     1Fam            5   
    
       YearBuilt  YearRemodAdd Exterior1st  BsmtFinSF2  TotalBsmtSF  SalePrice  
    0       2003          2003     VinylSd         0.0        856.0   208500.0  
    1       1976          1976     MetalSd         0.0       1262.0   181500.0  
    2       2001          2002     VinylSd         0.0        920.0   223500.0  
    3       1915          1970     Wd Sdng         0.0        756.0   140000.0  
    4       2000          2000     VinylSd         0.0       1145.0   250000.0  
    


```python
dataset.shape

```




    (2919, 13)




```python
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:",len(object_cols))

int_ = (dataset.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:",len(num_cols))

fl = (dataset.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:",len(fl_cols))

```

    Categorical variables: 4
    Integer variables: 0
    Float variables: 3
    


```python
plt.figure(figsize=(12, 6))
sns.heatmap(dataset.corr(),
			cmap = 'BrBG',
			fmt = '.2f',
			linewidths = 2,
			annot = True)

```




    <AxesSubplot:>




    
![png](output_3_1.png)
    



```python
unique_values = []
for col in object_cols:
    unique_values.append(dataset[col].unique().size)
plt.figure(figsize=(10,6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols,y=unique_values)

```




    <AxesSubplot:title={'center':'No. Unique values of Categorical Features'}>




    
![png](output_4_1.png)
    



```python
plt.figure(figsize=(18, 36))
plt.title('Categorical Features: Distribution')
plt.xticks(rotation=90)
index = 1

for col in object_cols:
	y = dataset[col].value_counts()
	plt.subplot(11, 4, index)
	plt.xticks(rotation=90)
	sns.barplot(x=list(y.index), y=y)
	index += 1

```


    
![png](output_5_0.png)
    



```python
dataset.drop(['Id'],
			axis=1,
			inplace=True)

```


```python
dataset['SalePrice'] = dataset['SalePrice'].fillna(
dataset['SalePrice'].mean())

```


```python
new_dataset = dataset.dropna()

```


```python
new_dataset.isnull().sum()

```




    MSSubClass      0
    MSZoning        0
    LotArea         0
    LotConfig       0
    BldgType        0
    OverallCond     0
    YearBuilt       0
    YearRemodAdd    0
    Exterior1st     0
    BsmtFinSF2      0
    TotalBsmtSF     0
    SalePrice       0
    dtype: int64




```python
from sklearn.preprocessing import OneHotEncoder

s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
print('No. of. categorical features: ', 
	len(object_cols))

```

    Categorical variables:
    ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']
    No. of. categorical features:  4
    


```python
OH_encoder = OneHotEncoder(sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names()
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
      warnings.warn(msg, category=FutureWarning)
    


```python
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

# Split the training set into 
# training and validation set
X_train, X_valid, Y_train, Y_valid = train_test_split(
	X, Y, train_size=0.8, test_size=0.2, random_state=0)

```


```python
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error

model_SVR = svm.SVR()
model_SVR.fit(X_train,Y_train)
Y_pred = model_SVR.predict(X_valid)

print(mean_absolute_percentage_error(Y_valid, Y_pred))

```

    0.1870512931870423
    


```python
from sklearn.ensemble import RandomForestRegressor

model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_valid)

mean_absolute_percentage_error(Y_valid, Y_pred)

```




    0.18796009047065879




```python
from sklearn.linear_model import LinearRegression

model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)

print(mean_absolute_percentage_error(Y_valid, Y_pred))

```

    0.1874168384159985
    


```python
# This code is contributed by @amartajisce
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
cb_model = CatBoostRegressor()
cb_model.fit(X_train, Y_train)
preds = cb_model.predict(X_valid) 

cb_r2_score=r2_score(Y_valid, preds)
cb_r2_score

```

    Learning rate set to 0.046797
    0:	learn: 56473.5753002	total: 3.87ms	remaining: 3.87s
    1:	learn: 55780.1567575	total: 7.56ms	remaining: 3.77s
    2:	learn: 55060.9599505	total: 10.8ms	remaining: 3.6s
    3:	learn: 54456.1126921	total: 13.9ms	remaining: 3.47s
    4:	learn: 53901.1464265	total: 17.3ms	remaining: 3.43s
    5:	learn: 53334.6062357	total: 20.5ms	remaining: 3.4s
    6:	learn: 52824.4943646	total: 24ms	remaining: 3.41s
    7:	learn: 52381.4267520	total: 26.9ms	remaining: 3.34s
    8:	learn: 51841.0364316	total: 29.8ms	remaining: 3.28s
    9:	learn: 51281.5852560	total: 32.6ms	remaining: 3.23s
    10:	learn: 50825.3817438	total: 35.5ms	remaining: 3.19s
    11:	learn: 50477.2609796	total: 38.3ms	remaining: 3.16s
    12:	learn: 50081.4097641	total: 41.1ms	remaining: 3.12s
    13:	learn: 49663.6380360	total: 44.1ms	remaining: 3.1s
    14:	learn: 49334.5782662	total: 47.1ms	remaining: 3.09s
    15:	learn: 48967.5221026	total: 50.1ms	remaining: 3.08s
    16:	learn: 48676.4063222	total: 53ms	remaining: 3.06s
    17:	learn: 48359.1169955	total: 56ms	remaining: 3.05s
    18:	learn: 48108.6667822	total: 58.7ms	remaining: 3.03s
    19:	learn: 47853.0639353	total: 63.4ms	remaining: 3.11s
    20:	learn: 47575.7274461	total: 68.9ms	remaining: 3.21s
    21:	learn: 47347.9557265	total: 74ms	remaining: 3.29s
    22:	learn: 47094.5043954	total: 80.3ms	remaining: 3.41s
    23:	learn: 46866.9875959	total: 83.3ms	remaining: 3.39s
    24:	learn: 46635.7633655	total: 86.8ms	remaining: 3.38s
    25:	learn: 46412.6742943	total: 89.8ms	remaining: 3.37s
    26:	learn: 46181.1587691	total: 92.7ms	remaining: 3.34s
    27:	learn: 46026.2604603	total: 95.5ms	remaining: 3.31s
    28:	learn: 45840.8610207	total: 98.5ms	remaining: 3.3s
    29:	learn: 45666.0389880	total: 102ms	remaining: 3.29s
    30:	learn: 45491.5537775	total: 105ms	remaining: 3.27s
    31:	learn: 45325.7838312	total: 107ms	remaining: 3.25s
    32:	learn: 45165.7041397	total: 110ms	remaining: 3.23s
    33:	learn: 45052.9639176	total: 113ms	remaining: 3.21s
    34:	learn: 44929.0789749	total: 116ms	remaining: 3.2s
    35:	learn: 44755.7471135	total: 119ms	remaining: 3.18s
    36:	learn: 44602.2630280	total: 122ms	remaining: 3.17s
    37:	learn: 44470.0364376	total: 124ms	remaining: 3.15s
    38:	learn: 44269.8275925	total: 127ms	remaining: 3.13s
    39:	learn: 44076.2668388	total: 130ms	remaining: 3.12s
    40:	learn: 43994.7053592	total: 133ms	remaining: 3.11s
    41:	learn: 43814.6393920	total: 136ms	remaining: 3.09s
    42:	learn: 43699.0023883	total: 138ms	remaining: 3.08s
    43:	learn: 43616.9603228	total: 141ms	remaining: 3.07s
    44:	learn: 43444.3817405	total: 144ms	remaining: 3.06s
    45:	learn: 43364.4422078	total: 147ms	remaining: 3.05s
    46:	learn: 43215.9415689	total: 150ms	remaining: 3.04s
    47:	learn: 43076.9106932	total: 153ms	remaining: 3.03s
    48:	learn: 43005.2937635	total: 156ms	remaining: 3.02s
    49:	learn: 42963.6403331	total: 159ms	remaining: 3.01s
    50:	learn: 42897.6070964	total: 161ms	remaining: 3s
    51:	learn: 42807.6088200	total: 165ms	remaining: 3.01s
    52:	learn: 42747.8647218	total: 168ms	remaining: 3.01s
    53:	learn: 42645.3051298	total: 172ms	remaining: 3s
    54:	learn: 42486.8662133	total: 175ms	remaining: 3s
    55:	learn: 42388.1111197	total: 178ms	remaining: 3s
    56:	learn: 42222.7775035	total: 182ms	remaining: 3s
    57:	learn: 42129.7641766	total: 185ms	remaining: 3s
    58:	learn: 42026.2449870	total: 188ms	remaining: 3s
    59:	learn: 41948.6566912	total: 191ms	remaining: 2.99s
    60:	learn: 41851.1931571	total: 194ms	remaining: 2.98s
    61:	learn: 41793.3182490	total: 197ms	remaining: 2.97s
    62:	learn: 41753.0028806	total: 200ms	remaining: 2.97s
    63:	learn: 41707.6463645	total: 203ms	remaining: 2.96s
    64:	learn: 41675.5837537	total: 205ms	remaining: 2.96s
    65:	learn: 41617.5347175	total: 208ms	remaining: 2.95s
    66:	learn: 41572.9230512	total: 211ms	remaining: 2.94s
    67:	learn: 41491.4903802	total: 214ms	remaining: 2.93s
    68:	learn: 41341.1234852	total: 217ms	remaining: 2.93s
    69:	learn: 41221.3593157	total: 220ms	remaining: 2.92s
    70:	learn: 41182.2626037	total: 223ms	remaining: 2.92s
    71:	learn: 41087.6523607	total: 226ms	remaining: 2.91s
    72:	learn: 40993.7325591	total: 229ms	remaining: 2.91s
    73:	learn: 40914.2176596	total: 232ms	remaining: 2.91s
    74:	learn: 40844.8426322	total: 235ms	remaining: 2.9s
    75:	learn: 40802.5484192	total: 239ms	remaining: 2.9s
    76:	learn: 40753.3684254	total: 242ms	remaining: 2.9s
    77:	learn: 40703.8481869	total: 245ms	remaining: 2.89s
    78:	learn: 40678.0263771	total: 248ms	remaining: 2.89s
    79:	learn: 40599.6196088	total: 251ms	remaining: 2.89s
    80:	learn: 40499.6701227	total: 254ms	remaining: 2.88s
    81:	learn: 40392.9539987	total: 257ms	remaining: 2.88s
    82:	learn: 40337.1832668	total: 260ms	remaining: 2.87s
    83:	learn: 40258.4687906	total: 263ms	remaining: 2.87s
    84:	learn: 40234.5671529	total: 266ms	remaining: 2.87s
    85:	learn: 40188.7622075	total: 270ms	remaining: 2.87s
    86:	learn: 40154.4595135	total: 273ms	remaining: 2.86s
    87:	learn: 40120.1537361	total: 276ms	remaining: 2.86s
    88:	learn: 40037.9782334	total: 279ms	remaining: 2.85s
    89:	learn: 40014.2876282	total: 282ms	remaining: 2.85s
    90:	learn: 39938.6585167	total: 285ms	remaining: 2.85s
    91:	learn: 39911.2484281	total: 288ms	remaining: 2.84s
    92:	learn: 39848.9357146	total: 291ms	remaining: 2.84s
    93:	learn: 39821.7792432	total: 294ms	remaining: 2.83s
    94:	learn: 39793.5230928	total: 297ms	remaining: 2.83s
    95:	learn: 39696.9244502	total: 299ms	remaining: 2.82s
    96:	learn: 39674.4773763	total: 302ms	remaining: 2.81s
    97:	learn: 39598.0479350	total: 305ms	remaining: 2.81s
    98:	learn: 39553.7653059	total: 308ms	remaining: 2.8s
    99:	learn: 39506.5137422	total: 310ms	remaining: 2.79s
    100:	learn: 39456.0563626	total: 313ms	remaining: 2.79s
    101:	learn: 39400.0656101	total: 316ms	remaining: 2.78s
    102:	learn: 39362.4420491	total: 319ms	remaining: 2.78s
    103:	learn: 39319.2404734	total: 322ms	remaining: 2.77s
    104:	learn: 39293.6944131	total: 325ms	remaining: 2.77s
    105:	learn: 39277.4342849	total: 328ms	remaining: 2.77s
    106:	learn: 39248.5680876	total: 331ms	remaining: 2.77s
    107:	learn: 39212.5180179	total: 334ms	remaining: 2.76s
    108:	learn: 39168.8684995	total: 337ms	remaining: 2.76s
    109:	learn: 39134.5177507	total: 341ms	remaining: 2.76s
    110:	learn: 39086.1493859	total: 344ms	remaining: 2.76s
    111:	learn: 39062.5194613	total: 348ms	remaining: 2.76s
    112:	learn: 39035.2071159	total: 351ms	remaining: 2.75s
    113:	learn: 38959.5398794	total: 355ms	remaining: 2.76s
    114:	learn: 38916.4851034	total: 359ms	remaining: 2.76s
    115:	learn: 38896.0240853	total: 363ms	remaining: 2.77s
    116:	learn: 38869.0433989	total: 367ms	remaining: 2.77s
    117:	learn: 38832.1515559	total: 371ms	remaining: 2.77s
    118:	learn: 38750.3970410	total: 374ms	remaining: 2.77s
    119:	learn: 38729.1981493	total: 377ms	remaining: 2.76s
    120:	learn: 38706.1685718	total: 380ms	remaining: 2.76s
    121:	learn: 38674.4144294	total: 383ms	remaining: 2.76s
    122:	learn: 38633.4609250	total: 386ms	remaining: 2.75s
    123:	learn: 38614.6126626	total: 389ms	remaining: 2.75s
    124:	learn: 38553.5861187	total: 392ms	remaining: 2.75s
    125:	learn: 38497.7249392	total: 395ms	remaining: 2.74s
    126:	learn: 38460.4427979	total: 398ms	remaining: 2.73s
    127:	learn: 38405.7511948	total: 401ms	remaining: 2.73s
    128:	learn: 38367.7706919	total: 403ms	remaining: 2.72s
    129:	learn: 38349.9029649	total: 406ms	remaining: 2.72s
    130:	learn: 38287.3920474	total: 409ms	remaining: 2.71s
    131:	learn: 38232.9662086	total: 412ms	remaining: 2.71s
    132:	learn: 38216.2402812	total: 415ms	remaining: 2.7s
    133:	learn: 38173.6477516	total: 418ms	remaining: 2.7s
    134:	learn: 38126.7715445	total: 420ms	remaining: 2.69s
    135:	learn: 38083.4777742	total: 423ms	remaining: 2.69s
    136:	learn: 38057.2206726	total: 426ms	remaining: 2.68s
    137:	learn: 38008.5761133	total: 429ms	remaining: 2.68s
    138:	learn: 37952.5758381	total: 431ms	remaining: 2.67s
    139:	learn: 37905.3565177	total: 434ms	remaining: 2.67s
    140:	learn: 37841.7255324	total: 437ms	remaining: 2.66s
    141:	learn: 37784.8376675	total: 440ms	remaining: 2.66s
    142:	learn: 37757.1446159	total: 443ms	remaining: 2.65s
    143:	learn: 37723.7275677	total: 446ms	remaining: 2.65s
    144:	learn: 37708.7126906	total: 448ms	remaining: 2.64s
    145:	learn: 37693.6104088	total: 451ms	remaining: 2.64s
    146:	learn: 37653.6408649	total: 454ms	remaining: 2.63s
    147:	learn: 37607.4153485	total: 457ms	remaining: 2.63s
    148:	learn: 37554.2144227	total: 460ms	remaining: 2.62s
    149:	learn: 37538.9543390	total: 462ms	remaining: 2.62s
    150:	learn: 37474.2980646	total: 465ms	remaining: 2.62s
    151:	learn: 37399.5795543	total: 468ms	remaining: 2.61s
    152:	learn: 37346.6201438	total: 471ms	remaining: 2.61s
    153:	learn: 37309.5514187	total: 474ms	remaining: 2.6s
    154:	learn: 37230.7920457	total: 476ms	remaining: 2.6s
    155:	learn: 37218.1280023	total: 479ms	remaining: 2.59s
    156:	learn: 37178.4227102	total: 482ms	remaining: 2.59s
    157:	learn: 37152.8623869	total: 485ms	remaining: 2.59s
    158:	learn: 37116.1450763	total: 489ms	remaining: 2.58s
    159:	learn: 37098.5199041	total: 492ms	remaining: 2.58s
    160:	learn: 37046.9019806	total: 496ms	remaining: 2.58s
    161:	learn: 37034.2261671	total: 499ms	remaining: 2.58s
    162:	learn: 36982.4863508	total: 502ms	remaining: 2.58s
    163:	learn: 36959.1884736	total: 506ms	remaining: 2.58s
    164:	learn: 36912.4658506	total: 509ms	remaining: 2.58s
    165:	learn: 36901.5083251	total: 513ms	remaining: 2.58s
    166:	learn: 36879.7290699	total: 516ms	remaining: 2.57s
    167:	learn: 36818.3994011	total: 519ms	remaining: 2.57s
    168:	learn: 36809.7009497	total: 523ms	remaining: 2.57s
    169:	learn: 36753.5090202	total: 525ms	remaining: 2.56s
    170:	learn: 36699.6528029	total: 528ms	remaining: 2.56s
    171:	learn: 36655.1557063	total: 531ms	remaining: 2.56s
    172:	learn: 36605.5511970	total: 534ms	remaining: 2.55s
    173:	learn: 36579.4625948	total: 537ms	remaining: 2.55s
    174:	learn: 36543.9282253	total: 541ms	remaining: 2.55s
    175:	learn: 36465.6936476	total: 544ms	remaining: 2.54s
    176:	learn: 36414.4346420	total: 547ms	remaining: 2.54s
    177:	learn: 36410.3083495	total: 550ms	remaining: 2.54s
    178:	learn: 36392.7209783	total: 553ms	remaining: 2.54s
    179:	learn: 36307.9197006	total: 556ms	remaining: 2.53s
    180:	learn: 36291.9946353	total: 559ms	remaining: 2.53s
    181:	learn: 36251.9542530	total: 562ms	remaining: 2.53s
    182:	learn: 36181.4055421	total: 565ms	remaining: 2.52s
    183:	learn: 36157.7105288	total: 568ms	remaining: 2.52s
    184:	learn: 36103.2908091	total: 571ms	remaining: 2.51s
    185:	learn: 36065.1382253	total: 573ms	remaining: 2.51s
    186:	learn: 36042.1262833	total: 576ms	remaining: 2.5s
    187:	learn: 36016.4158790	total: 579ms	remaining: 2.5s
    188:	learn: 35987.8006702	total: 582ms	remaining: 2.5s
    189:	learn: 35949.1618555	total: 585ms	remaining: 2.49s
    190:	learn: 35928.7328737	total: 588ms	remaining: 2.49s
    191:	learn: 35917.1918326	total: 591ms	remaining: 2.48s
    192:	learn: 35880.8598420	total: 593ms	remaining: 2.48s
    193:	learn: 35845.6369090	total: 596ms	remaining: 2.48s
    194:	learn: 35831.3873203	total: 599ms	remaining: 2.47s
    195:	learn: 35812.1011624	total: 601ms	remaining: 2.47s
    196:	learn: 35785.7026882	total: 604ms	remaining: 2.46s
    197:	learn: 35769.9634355	total: 607ms	remaining: 2.46s
    198:	learn: 35749.7209476	total: 610ms	remaining: 2.45s
    199:	learn: 35728.4131874	total: 612ms	remaining: 2.45s
    200:	learn: 35702.3004867	total: 615ms	remaining: 2.44s
    201:	learn: 35667.4013499	total: 618ms	remaining: 2.44s
    202:	learn: 35649.4103685	total: 621ms	remaining: 2.44s
    203:	learn: 35618.5426638	total: 623ms	remaining: 2.43s
    204:	learn: 35606.7693976	total: 626ms	remaining: 2.43s
    205:	learn: 35579.8601473	total: 629ms	remaining: 2.42s
    206:	learn: 35574.3641239	total: 632ms	remaining: 2.42s
    207:	learn: 35541.6332262	total: 634ms	remaining: 2.42s
    208:	learn: 35510.9749321	total: 637ms	remaining: 2.41s
    209:	learn: 35498.3273240	total: 640ms	remaining: 2.41s
    210:	learn: 35490.7288543	total: 643ms	remaining: 2.4s
    211:	learn: 35476.7549124	total: 646ms	remaining: 2.4s
    212:	learn: 35429.2461185	total: 649ms	remaining: 2.4s
    213:	learn: 35351.2736556	total: 652ms	remaining: 2.4s
    214:	learn: 35309.7722263	total: 655ms	remaining: 2.39s
    215:	learn: 35263.5896403	total: 659ms	remaining: 2.39s
    216:	learn: 35235.1682161	total: 662ms	remaining: 2.39s
    217:	learn: 35223.6283534	total: 666ms	remaining: 2.39s
    218:	learn: 35193.5159629	total: 670ms	remaining: 2.39s
    219:	learn: 35147.4814148	total: 673ms	remaining: 2.39s
    220:	learn: 35104.6993743	total: 677ms	remaining: 2.38s
    221:	learn: 35036.3870121	total: 680ms	remaining: 2.38s
    222:	learn: 35012.2384427	total: 683ms	remaining: 2.38s
    223:	learn: 35001.2105947	total: 686ms	remaining: 2.38s
    224:	learn: 34970.3314687	total: 689ms	remaining: 2.37s
    225:	learn: 34906.9637442	total: 693ms	remaining: 2.37s
    226:	learn: 34896.0425398	total: 696ms	remaining: 2.37s
    227:	learn: 34867.8686208	total: 699ms	remaining: 2.37s
    228:	learn: 34849.8025535	total: 702ms	remaining: 2.36s
    229:	learn: 34810.5712330	total: 706ms	remaining: 2.36s
    230:	learn: 34780.0554139	total: 709ms	remaining: 2.36s
    231:	learn: 34769.4794443	total: 712ms	remaining: 2.36s
    232:	learn: 34764.8045180	total: 715ms	remaining: 2.35s
    233:	learn: 34745.9974300	total: 718ms	remaining: 2.35s
    234:	learn: 34722.9339693	total: 721ms	remaining: 2.35s
    235:	learn: 34663.0456906	total: 724ms	remaining: 2.35s
    236:	learn: 34622.9629595	total: 727ms	remaining: 2.34s
    237:	learn: 34610.7408221	total: 730ms	remaining: 2.34s
    238:	learn: 34590.5004989	total: 734ms	remaining: 2.33s
    239:	learn: 34580.7220869	total: 737ms	remaining: 2.33s
    240:	learn: 34536.9514911	total: 740ms	remaining: 2.33s
    241:	learn: 34495.0525852	total: 744ms	remaining: 2.33s
    242:	learn: 34451.3400745	total: 747ms	remaining: 2.33s
    243:	learn: 34388.6643419	total: 750ms	remaining: 2.32s
    244:	learn: 34349.4846504	total: 753ms	remaining: 2.32s
    245:	learn: 34332.9296320	total: 757ms	remaining: 2.32s
    246:	learn: 34312.6609499	total: 760ms	remaining: 2.32s
    247:	learn: 34289.7780619	total: 763ms	remaining: 2.31s
    248:	learn: 34242.4124829	total: 767ms	remaining: 2.31s
    249:	learn: 34234.6557912	total: 770ms	remaining: 2.31s
    250:	learn: 34202.0437870	total: 773ms	remaining: 2.31s
    251:	learn: 34173.1068824	total: 777ms	remaining: 2.31s
    252:	learn: 34127.3932344	total: 780ms	remaining: 2.3s
    253:	learn: 34114.3831422	total: 783ms	remaining: 2.3s
    254:	learn: 34066.7481942	total: 786ms	remaining: 2.3s
    255:	learn: 34058.5727684	total: 789ms	remaining: 2.29s
    256:	learn: 34022.0854263	total: 793ms	remaining: 2.29s
    257:	learn: 34013.0585666	total: 796ms	remaining: 2.29s
    258:	learn: 33988.3509963	total: 799ms	remaining: 2.29s
    259:	learn: 33984.9893767	total: 802ms	remaining: 2.28s
    260:	learn: 33959.7806988	total: 806ms	remaining: 2.28s
    261:	learn: 33957.5510725	total: 808ms	remaining: 2.28s
    262:	learn: 33925.7343653	total: 812ms	remaining: 2.27s
    263:	learn: 33905.7689352	total: 815ms	remaining: 2.27s
    264:	learn: 33879.6944778	total: 818ms	remaining: 2.27s
    265:	learn: 33857.5833357	total: 822ms	remaining: 2.27s
    266:	learn: 33809.8811651	total: 826ms	remaining: 2.27s
    267:	learn: 33774.0731389	total: 829ms	remaining: 2.26s
    268:	learn: 33761.9125396	total: 832ms	remaining: 2.26s
    269:	learn: 33750.0044874	total: 836ms	remaining: 2.26s
    270:	learn: 33703.4669238	total: 839ms	remaining: 2.26s
    271:	learn: 33669.6786657	total: 843ms	remaining: 2.25s
    272:	learn: 33660.0470206	total: 846ms	remaining: 2.25s
    273:	learn: 33641.1915939	total: 849ms	remaining: 2.25s
    274:	learn: 33620.8735622	total: 852ms	remaining: 2.25s
    275:	learn: 33589.4208960	total: 855ms	remaining: 2.24s
    276:	learn: 33578.4534531	total: 859ms	remaining: 2.24s
    277:	learn: 33554.7358677	total: 863ms	remaining: 2.24s
    278:	learn: 33542.2031956	total: 866ms	remaining: 2.24s
    279:	learn: 33522.8641847	total: 870ms	remaining: 2.24s
    280:	learn: 33514.5578831	total: 874ms	remaining: 2.23s
    281:	learn: 33489.9534932	total: 877ms	remaining: 2.23s
    282:	learn: 33476.5897562	total: 880ms	remaining: 2.23s
    283:	learn: 33445.4254500	total: 883ms	remaining: 2.23s
    284:	learn: 33429.9649716	total: 886ms	remaining: 2.22s
    285:	learn: 33390.4311514	total: 890ms	remaining: 2.22s
    286:	learn: 33379.2454270	total: 893ms	remaining: 2.22s
    287:	learn: 33362.9544758	total: 896ms	remaining: 2.21s
    288:	learn: 33349.9636532	total: 899ms	remaining: 2.21s
    289:	learn: 33311.9396578	total: 902ms	remaining: 2.21s
    290:	learn: 33302.5056011	total: 905ms	remaining: 2.21s
    291:	learn: 33281.0051691	total: 909ms	remaining: 2.2s
    292:	learn: 33263.2183598	total: 912ms	remaining: 2.2s
    293:	learn: 33238.2896251	total: 915ms	remaining: 2.2s
    294:	learn: 33213.4734195	total: 918ms	remaining: 2.19s
    295:	learn: 33175.2402163	total: 922ms	remaining: 2.19s
    296:	learn: 33153.9006111	total: 925ms	remaining: 2.19s
    297:	learn: 33142.9106927	total: 928ms	remaining: 2.19s
    298:	learn: 33139.4437591	total: 931ms	remaining: 2.18s
    299:	learn: 33114.2865937	total: 934ms	remaining: 2.18s
    300:	learn: 33081.5915302	total: 938ms	remaining: 2.18s
    301:	learn: 33060.8770896	total: 941ms	remaining: 2.17s
    302:	learn: 33052.2005423	total: 944ms	remaining: 2.17s
    303:	learn: 33024.3909020	total: 948ms	remaining: 2.17s
    304:	learn: 32995.4781441	total: 951ms	remaining: 2.17s
    305:	learn: 32976.0030517	total: 954ms	remaining: 2.16s
    306:	learn: 32974.0891709	total: 957ms	remaining: 2.16s
    307:	learn: 32964.9686318	total: 960ms	remaining: 2.16s
    308:	learn: 32956.4991728	total: 963ms	remaining: 2.15s
    309:	learn: 32943.1594567	total: 967ms	remaining: 2.15s
    310:	learn: 32919.3862155	total: 970ms	remaining: 2.15s
    311:	learn: 32878.8200553	total: 973ms	remaining: 2.15s
    312:	learn: 32847.7595617	total: 976ms	remaining: 2.14s
    313:	learn: 32842.5235768	total: 979ms	remaining: 2.14s
    314:	learn: 32824.5109330	total: 982ms	remaining: 2.14s
    315:	learn: 32818.9612078	total: 986ms	remaining: 2.13s
    316:	learn: 32779.3109671	total: 990ms	remaining: 2.13s
    317:	learn: 32762.4361952	total: 993ms	remaining: 2.13s
    318:	learn: 32745.3517483	total: 996ms	remaining: 2.13s
    319:	learn: 32735.4944654	total: 999ms	remaining: 2.12s
    320:	learn: 32696.2270902	total: 1s	remaining: 2.12s
    321:	learn: 32689.3330856	total: 1.01s	remaining: 2.12s
    322:	learn: 32671.2562792	total: 1.01s	remaining: 2.12s
    323:	learn: 32654.9461941	total: 1.01s	remaining: 2.12s
    324:	learn: 32644.7743588	total: 1.02s	remaining: 2.11s
    325:	learn: 32638.0281162	total: 1.02s	remaining: 2.11s
    326:	learn: 32629.7138419	total: 1.02s	remaining: 2.11s
    327:	learn: 32605.3209537	total: 1.03s	remaining: 2.1s
    328:	learn: 32597.9599298	total: 1.03s	remaining: 2.1s
    329:	learn: 32574.3223605	total: 1.03s	remaining: 2.1s
    330:	learn: 32552.0392973	total: 1.04s	remaining: 2.1s
    331:	learn: 32508.1543408	total: 1.04s	remaining: 2.1s
    332:	learn: 32488.5691996	total: 1.04s	remaining: 2.09s
    333:	learn: 32458.0652506	total: 1.05s	remaining: 2.09s
    334:	learn: 32445.2318948	total: 1.05s	remaining: 2.09s
    335:	learn: 32422.0001702	total: 1.06s	remaining: 2.09s
    336:	learn: 32408.5529194	total: 1.06s	remaining: 2.09s
    337:	learn: 32390.8218054	total: 1.06s	remaining: 2.08s
    338:	learn: 32378.6639326	total: 1.07s	remaining: 2.08s
    339:	learn: 32361.8784938	total: 1.07s	remaining: 2.08s
    340:	learn: 32345.6541102	total: 1.08s	remaining: 2.08s
    341:	learn: 32332.8240190	total: 1.08s	remaining: 2.08s
    342:	learn: 32295.4043157	total: 1.08s	remaining: 2.08s
    343:	learn: 32280.9805539	total: 1.09s	remaining: 2.07s
    344:	learn: 32276.6362955	total: 1.09s	remaining: 2.07s
    345:	learn: 32271.7904314	total: 1.09s	remaining: 2.07s
    346:	learn: 32252.7899611	total: 1.1s	remaining: 2.06s
    347:	learn: 32231.6207648	total: 1.1s	remaining: 2.06s
    348:	learn: 32218.5159007	total: 1.1s	remaining: 2.06s
    349:	learn: 32199.6115004	total: 1.11s	remaining: 2.06s
    350:	learn: 32194.8208188	total: 1.11s	remaining: 2.06s
    351:	learn: 32176.5226611	total: 1.11s	remaining: 2.05s
    352:	learn: 32156.4120078	total: 1.12s	remaining: 2.05s
    353:	learn: 32155.0014563	total: 1.13s	remaining: 2.07s
    354:	learn: 32130.1088453	total: 1.14s	remaining: 2.06s
    355:	learn: 32100.7448218	total: 1.14s	remaining: 2.06s
    356:	learn: 32064.5418014	total: 1.14s	remaining: 2.06s
    357:	learn: 32055.6209191	total: 1.15s	remaining: 2.06s
    358:	learn: 32049.1737059	total: 1.15s	remaining: 2.06s
    359:	learn: 32014.5157077	total: 1.16s	remaining: 2.06s
    360:	learn: 32003.9385204	total: 1.16s	remaining: 2.06s
    361:	learn: 31992.2246163	total: 1.17s	remaining: 2.05s
    362:	learn: 31973.5797601	total: 1.17s	remaining: 2.05s
    363:	learn: 31946.2424370	total: 1.17s	remaining: 2.05s
    364:	learn: 31930.6931401	total: 1.18s	remaining: 2.05s
    365:	learn: 31912.4063410	total: 1.18s	remaining: 2.05s
    366:	learn: 31896.0170944	total: 1.19s	remaining: 2.04s
    367:	learn: 31876.6770189	total: 1.19s	remaining: 2.04s
    368:	learn: 31865.5532745	total: 1.19s	remaining: 2.04s
    369:	learn: 31864.2368691	total: 1.2s	remaining: 2.04s
    370:	learn: 31848.3585056	total: 1.2s	remaining: 2.03s
    371:	learn: 31832.1161721	total: 1.2s	remaining: 2.03s
    372:	learn: 31820.7928754	total: 1.21s	remaining: 2.03s
    373:	learn: 31819.5207840	total: 1.21s	remaining: 2.03s
    374:	learn: 31795.6204227	total: 1.21s	remaining: 2.02s
    375:	learn: 31794.5602289	total: 1.22s	remaining: 2.02s
    376:	learn: 31771.0692670	total: 1.22s	remaining: 2.02s
    377:	learn: 31739.1893560	total: 1.22s	remaining: 2.01s
    378:	learn: 31729.2100149	total: 1.23s	remaining: 2.01s
    379:	learn: 31712.8416310	total: 1.23s	remaining: 2.01s
    380:	learn: 31692.8699615	total: 1.24s	remaining: 2.01s
    381:	learn: 31691.7649609	total: 1.24s	remaining: 2s
    382:	learn: 31683.3983862	total: 1.24s	remaining: 2s
    383:	learn: 31663.6066672	total: 1.25s	remaining: 2s
    384:	learn: 31662.7909267	total: 1.25s	remaining: 2s
    385:	learn: 31640.1871632	total: 1.25s	remaining: 1.99s
    386:	learn: 31622.1911703	total: 1.26s	remaining: 1.99s
    387:	learn: 31604.9711454	total: 1.26s	remaining: 1.99s
    388:	learn: 31603.6343929	total: 1.26s	remaining: 1.99s
    389:	learn: 31583.2381019	total: 1.27s	remaining: 1.98s
    390:	learn: 31558.6901607	total: 1.27s	remaining: 1.98s
    391:	learn: 31537.2835666	total: 1.27s	remaining: 1.98s
    392:	learn: 31501.9854481	total: 1.28s	remaining: 1.98s
    393:	learn: 31475.5621723	total: 1.28s	remaining: 1.97s
    394:	learn: 31459.2061572	total: 1.28s	remaining: 1.97s
    395:	learn: 31452.8437625	total: 1.29s	remaining: 1.97s
    396:	learn: 31433.0033412	total: 1.29s	remaining: 1.96s
    397:	learn: 31423.6309837	total: 1.3s	remaining: 1.96s
    398:	learn: 31404.9139498	total: 1.3s	remaining: 1.96s
    399:	learn: 31387.6614726	total: 1.3s	remaining: 1.96s
    400:	learn: 31386.8491363	total: 1.31s	remaining: 1.95s
    401:	learn: 31347.8827309	total: 1.31s	remaining: 1.95s
    402:	learn: 31333.3092914	total: 1.31s	remaining: 1.95s
    403:	learn: 31323.8306025	total: 1.32s	remaining: 1.95s
    404:	learn: 31287.4401854	total: 1.32s	remaining: 1.94s
    405:	learn: 31273.1305751	total: 1.33s	remaining: 1.94s
    406:	learn: 31254.1659533	total: 1.33s	remaining: 1.94s
    407:	learn: 31249.9586656	total: 1.33s	remaining: 1.94s
    408:	learn: 31231.4762074	total: 1.34s	remaining: 1.93s
    409:	learn: 31218.5278096	total: 1.34s	remaining: 1.93s
    410:	learn: 31189.6213209	total: 1.35s	remaining: 1.93s
    411:	learn: 31175.6186112	total: 1.35s	remaining: 1.93s
    412:	learn: 31134.3337315	total: 1.35s	remaining: 1.92s
    413:	learn: 31107.9902047	total: 1.36s	remaining: 1.92s
    414:	learn: 31071.2930767	total: 1.36s	remaining: 1.92s
    415:	learn: 31059.8709101	total: 1.36s	remaining: 1.91s
    416:	learn: 31036.5483976	total: 1.37s	remaining: 1.91s
    417:	learn: 31035.6832782	total: 1.37s	remaining: 1.91s
    418:	learn: 31002.2661879	total: 1.37s	remaining: 1.91s
    419:	learn: 30988.4304638	total: 1.38s	remaining: 1.9s
    420:	learn: 30973.3237141	total: 1.38s	remaining: 1.9s
    421:	learn: 30942.2800427	total: 1.38s	remaining: 1.9s
    422:	learn: 30922.9640263	total: 1.39s	remaining: 1.89s
    423:	learn: 30912.8481093	total: 1.39s	remaining: 1.89s
    424:	learn: 30897.8481094	total: 1.39s	remaining: 1.89s
    425:	learn: 30873.5088623	total: 1.4s	remaining: 1.88s
    426:	learn: 30863.6892384	total: 1.4s	remaining: 1.88s
    427:	learn: 30857.2942773	total: 1.4s	remaining: 1.88s
    428:	learn: 30825.4972122	total: 1.41s	remaining: 1.87s
    429:	learn: 30809.1843410	total: 1.41s	remaining: 1.87s
    430:	learn: 30801.8756614	total: 1.41s	remaining: 1.87s
    431:	learn: 30788.8253290	total: 1.42s	remaining: 1.86s
    432:	learn: 30783.7349439	total: 1.42s	remaining: 1.86s
    433:	learn: 30782.7965809	total: 1.42s	remaining: 1.85s
    434:	learn: 30763.6377908	total: 1.43s	remaining: 1.85s
    435:	learn: 30750.8702142	total: 1.43s	remaining: 1.85s
    436:	learn: 30735.7370920	total: 1.43s	remaining: 1.84s
    437:	learn: 30721.0496888	total: 1.44s	remaining: 1.84s
    438:	learn: 30706.8931202	total: 1.44s	remaining: 1.84s
    439:	learn: 30690.0241920	total: 1.44s	remaining: 1.83s
    440:	learn: 30674.5005271	total: 1.45s	remaining: 1.83s
    441:	learn: 30662.2859072	total: 1.45s	remaining: 1.83s
    442:	learn: 30647.6952624	total: 1.45s	remaining: 1.82s
    443:	learn: 30634.9298940	total: 1.46s	remaining: 1.82s
    444:	learn: 30601.0020793	total: 1.46s	remaining: 1.82s
    445:	learn: 30582.2159832	total: 1.46s	remaining: 1.82s
    446:	learn: 30558.9039606	total: 1.47s	remaining: 1.81s
    447:	learn: 30543.7151898	total: 1.47s	remaining: 1.81s
    448:	learn: 30525.5825955	total: 1.47s	remaining: 1.81s
    449:	learn: 30508.2389081	total: 1.48s	remaining: 1.81s
    450:	learn: 30507.5758763	total: 1.48s	remaining: 1.8s
    451:	learn: 30492.4552916	total: 1.49s	remaining: 1.8s
    452:	learn: 30491.7712064	total: 1.49s	remaining: 1.8s
    453:	learn: 30485.5538551	total: 1.49s	remaining: 1.79s
    454:	learn: 30469.0200656	total: 1.5s	remaining: 1.79s
    455:	learn: 30466.5818221	total: 1.5s	remaining: 1.79s
    456:	learn: 30465.9750712	total: 1.5s	remaining: 1.79s
    457:	learn: 30455.1888557	total: 1.51s	remaining: 1.78s
    458:	learn: 30419.7678303	total: 1.51s	remaining: 1.78s
    459:	learn: 30414.8746690	total: 1.51s	remaining: 1.78s
    460:	learn: 30397.7207899	total: 1.52s	remaining: 1.77s
    461:	learn: 30391.5241271	total: 1.52s	remaining: 1.77s
    462:	learn: 30376.4851445	total: 1.52s	remaining: 1.77s
    463:	learn: 30345.5690790	total: 1.53s	remaining: 1.76s
    464:	learn: 30328.7813947	total: 1.53s	remaining: 1.76s
    465:	learn: 30311.4550831	total: 1.53s	remaining: 1.76s
    466:	learn: 30292.8986228	total: 1.54s	remaining: 1.76s
    467:	learn: 30285.1124693	total: 1.54s	remaining: 1.75s
    468:	learn: 30273.8773150	total: 1.54s	remaining: 1.75s
    469:	learn: 30236.6728315	total: 1.55s	remaining: 1.75s
    470:	learn: 30219.2175700	total: 1.55s	remaining: 1.74s
    471:	learn: 30194.0106940	total: 1.55s	remaining: 1.74s
    472:	learn: 30183.8861645	total: 1.56s	remaining: 1.74s
    473:	learn: 30164.8229691	total: 1.56s	remaining: 1.73s
    474:	learn: 30149.1223050	total: 1.56s	remaining: 1.73s
    475:	learn: 30131.6428643	total: 1.57s	remaining: 1.73s
    476:	learn: 30104.3076206	total: 1.57s	remaining: 1.72s
    477:	learn: 30092.8173295	total: 1.57s	remaining: 1.72s
    478:	learn: 30081.4552493	total: 1.58s	remaining: 1.72s
    479:	learn: 30068.3458234	total: 1.58s	remaining: 1.72s
    480:	learn: 30067.5448403	total: 1.59s	remaining: 1.71s
    481:	learn: 30062.1761666	total: 1.59s	remaining: 1.71s
    482:	learn: 30025.7909606	total: 1.59s	remaining: 1.7s
    483:	learn: 30010.3925138	total: 1.6s	remaining: 1.7s
    484:	learn: 29979.4771893	total: 1.6s	remaining: 1.7s
    485:	learn: 29952.1561010	total: 1.6s	remaining: 1.7s
    486:	learn: 29942.4234967	total: 1.61s	remaining: 1.69s
    487:	learn: 29923.8806579	total: 1.61s	remaining: 1.69s
    488:	learn: 29919.4235220	total: 1.61s	remaining: 1.69s
    489:	learn: 29915.0140537	total: 1.62s	remaining: 1.68s
    490:	learn: 29908.7767253	total: 1.62s	remaining: 1.68s
    491:	learn: 29894.4826947	total: 1.62s	remaining: 1.68s
    492:	learn: 29877.7470785	total: 1.63s	remaining: 1.67s
    493:	learn: 29862.6670317	total: 1.63s	remaining: 1.67s
    494:	learn: 29834.5203370	total: 1.64s	remaining: 1.67s
    495:	learn: 29809.8353901	total: 1.64s	remaining: 1.67s
    496:	learn: 29798.2389078	total: 1.64s	remaining: 1.66s
    497:	learn: 29787.4789511	total: 1.65s	remaining: 1.66s
    498:	learn: 29786.5437597	total: 1.65s	remaining: 1.66s
    499:	learn: 29774.1973240	total: 1.65s	remaining: 1.65s
    500:	learn: 29752.3475583	total: 1.66s	remaining: 1.65s
    501:	learn: 29732.4272117	total: 1.66s	remaining: 1.65s
    502:	learn: 29720.8126444	total: 1.66s	remaining: 1.64s
    503:	learn: 29703.9127095	total: 1.67s	remaining: 1.64s
    504:	learn: 29690.4098733	total: 1.67s	remaining: 1.64s
    505:	learn: 29674.4415028	total: 1.68s	remaining: 1.64s
    506:	learn: 29644.4668080	total: 1.68s	remaining: 1.63s
    507:	learn: 29635.8211436	total: 1.68s	remaining: 1.63s
    508:	learn: 29623.5912787	total: 1.69s	remaining: 1.63s
    509:	learn: 29621.4475516	total: 1.69s	remaining: 1.62s
    510:	learn: 29607.8775894	total: 1.69s	remaining: 1.62s
    511:	learn: 29597.6740157	total: 1.7s	remaining: 1.62s
    512:	learn: 29595.5179625	total: 1.7s	remaining: 1.61s
    513:	learn: 29571.9068429	total: 1.7s	remaining: 1.61s
    514:	learn: 29551.0723215	total: 1.71s	remaining: 1.6s
    515:	learn: 29545.7812676	total: 1.71s	remaining: 1.6s
    516:	learn: 29525.5954073	total: 1.71s	remaining: 1.6s
    517:	learn: 29505.2627588	total: 1.71s	remaining: 1.59s
    518:	learn: 29489.4504641	total: 1.72s	remaining: 1.59s
    519:	learn: 29472.8879574	total: 1.72s	remaining: 1.59s
    520:	learn: 29457.7911901	total: 1.72s	remaining: 1.58s
    521:	learn: 29436.6970790	total: 1.73s	remaining: 1.58s
    522:	learn: 29422.0296841	total: 1.73s	remaining: 1.58s
    523:	learn: 29416.9535024	total: 1.74s	remaining: 1.58s
    524:	learn: 29397.4733548	total: 1.74s	remaining: 1.57s
    525:	learn: 29373.2851279	total: 1.74s	remaining: 1.57s
    526:	learn: 29352.7337258	total: 1.75s	remaining: 1.57s
    527:	learn: 29350.1274916	total: 1.75s	remaining: 1.57s
    528:	learn: 29326.0871757	total: 1.75s	remaining: 1.56s
    529:	learn: 29314.7075176	total: 1.76s	remaining: 1.56s
    530:	learn: 29314.2044396	total: 1.76s	remaining: 1.56s
    531:	learn: 29296.3646463	total: 1.77s	remaining: 1.55s
    532:	learn: 29278.4376471	total: 1.77s	remaining: 1.55s
    533:	learn: 29262.6787723	total: 1.77s	remaining: 1.55s
    534:	learn: 29251.1672973	total: 1.78s	remaining: 1.55s
    535:	learn: 29230.0358048	total: 1.78s	remaining: 1.54s
    536:	learn: 29220.2458222	total: 1.79s	remaining: 1.54s
    537:	learn: 29208.8344048	total: 1.79s	remaining: 1.54s
    538:	learn: 29193.9632538	total: 1.8s	remaining: 1.54s
    539:	learn: 29173.3137520	total: 1.81s	remaining: 1.54s
    540:	learn: 29153.6951464	total: 1.81s	remaining: 1.54s
    541:	learn: 29137.4260953	total: 1.82s	remaining: 1.53s
    542:	learn: 29120.9418826	total: 1.82s	remaining: 1.53s
    543:	learn: 29101.0896427	total: 1.83s	remaining: 1.53s
    544:	learn: 29088.9373949	total: 1.83s	remaining: 1.53s
    545:	learn: 29082.4822987	total: 1.83s	remaining: 1.52s
    546:	learn: 29057.5774152	total: 1.84s	remaining: 1.52s
    547:	learn: 29047.3352527	total: 1.84s	remaining: 1.52s
    548:	learn: 29036.8058484	total: 1.84s	remaining: 1.52s
    549:	learn: 29018.8716696	total: 1.85s	remaining: 1.51s
    550:	learn: 29018.2001346	total: 1.85s	remaining: 1.51s
    551:	learn: 29012.8100606	total: 1.86s	remaining: 1.51s
    552:	learn: 28998.7682507	total: 1.86s	remaining: 1.51s
    553:	learn: 28991.9589635	total: 1.87s	remaining: 1.5s
    554:	learn: 28970.8814638	total: 1.87s	remaining: 1.5s
    555:	learn: 28952.1377246	total: 1.87s	remaining: 1.5s
    556:	learn: 28942.0852335	total: 1.88s	remaining: 1.49s
    557:	learn: 28917.0350917	total: 1.88s	remaining: 1.49s
    558:	learn: 28889.2123888	total: 1.89s	remaining: 1.49s
    559:	learn: 28876.5037210	total: 1.89s	remaining: 1.48s
    560:	learn: 28865.5024790	total: 1.89s	remaining: 1.48s
    561:	learn: 28852.2935947	total: 1.9s	remaining: 1.48s
    562:	learn: 28831.6781065	total: 1.9s	remaining: 1.47s
    563:	learn: 28805.7850217	total: 1.9s	remaining: 1.47s
    564:	learn: 28789.1015234	total: 1.91s	remaining: 1.47s
    565:	learn: 28771.2662481	total: 1.91s	remaining: 1.46s
    566:	learn: 28763.3730279	total: 1.91s	remaining: 1.46s
    567:	learn: 28752.1228173	total: 1.92s	remaining: 1.46s
    568:	learn: 28741.8497206	total: 1.92s	remaining: 1.45s
    569:	learn: 28725.5866734	total: 1.92s	remaining: 1.45s
    570:	learn: 28710.9524371	total: 1.93s	remaining: 1.45s
    571:	learn: 28708.8276816	total: 1.93s	remaining: 1.44s
    572:	learn: 28693.7282652	total: 1.93s	remaining: 1.44s
    573:	learn: 28683.6442476	total: 1.94s	remaining: 1.44s
    574:	learn: 28670.0530811	total: 1.94s	remaining: 1.43s
    575:	learn: 28642.4405526	total: 1.95s	remaining: 1.43s
    576:	learn: 28625.3362659	total: 1.95s	remaining: 1.43s
    577:	learn: 28613.1214526	total: 1.95s	remaining: 1.43s
    578:	learn: 28611.4277306	total: 1.96s	remaining: 1.42s
    579:	learn: 28592.6055085	total: 1.96s	remaining: 1.42s
    580:	learn: 28566.6142102	total: 1.96s	remaining: 1.42s
    581:	learn: 28556.7323873	total: 1.97s	remaining: 1.41s
    582:	learn: 28550.1453957	total: 1.97s	remaining: 1.41s
    583:	learn: 28540.5355790	total: 1.98s	remaining: 1.41s
    584:	learn: 28537.4486993	total: 1.98s	remaining: 1.4s
    585:	learn: 28520.4855149	total: 1.98s	remaining: 1.4s
    586:	learn: 28510.9542296	total: 1.99s	remaining: 1.4s
    587:	learn: 28487.6223581	total: 1.99s	remaining: 1.39s
    588:	learn: 28466.3388189	total: 1.99s	remaining: 1.39s
    589:	learn: 28454.0515296	total: 2s	remaining: 1.39s
    590:	learn: 28446.4850351	total: 2s	remaining: 1.38s
    591:	learn: 28446.0656623	total: 2s	remaining: 1.38s
    592:	learn: 28437.2428667	total: 2s	remaining: 1.38s
    593:	learn: 28420.3138032	total: 2.01s	remaining: 1.37s
    594:	learn: 28419.9665640	total: 2.01s	remaining: 1.37s
    595:	learn: 28411.4585865	total: 2.01s	remaining: 1.36s
    596:	learn: 28398.8263440	total: 2.02s	remaining: 1.36s
    597:	learn: 28378.2969769	total: 2.02s	remaining: 1.36s
    598:	learn: 28370.8673636	total: 2.02s	remaining: 1.35s
    599:	learn: 28361.4279286	total: 2.03s	remaining: 1.35s
    600:	learn: 28346.9914081	total: 2.03s	remaining: 1.35s
    601:	learn: 28346.5751806	total: 2.03s	remaining: 1.34s
    602:	learn: 28330.7726540	total: 2.04s	remaining: 1.34s
    603:	learn: 28313.6356418	total: 2.04s	remaining: 1.34s
    604:	learn: 28287.6899075	total: 2.04s	remaining: 1.33s
    605:	learn: 28276.7219884	total: 2.05s	remaining: 1.33s
    606:	learn: 28258.2333270	total: 2.05s	remaining: 1.33s
    607:	learn: 28257.8582075	total: 2.05s	remaining: 1.32s
    608:	learn: 28241.3339552	total: 2.06s	remaining: 1.32s
    609:	learn: 28234.1065658	total: 2.06s	remaining: 1.32s
    610:	learn: 28227.2365291	total: 2.06s	remaining: 1.31s
    611:	learn: 28210.0947531	total: 2.07s	remaining: 1.31s
    612:	learn: 28194.3868173	total: 2.07s	remaining: 1.31s
    613:	learn: 28193.4836836	total: 2.07s	remaining: 1.3s
    614:	learn: 28168.2496843	total: 2.08s	remaining: 1.3s
    615:	learn: 28160.9161050	total: 2.08s	remaining: 1.3s
    616:	learn: 28149.8483305	total: 2.08s	remaining: 1.29s
    617:	learn: 28134.8333748	total: 2.08s	remaining: 1.29s
    618:	learn: 28116.8101072	total: 2.09s	remaining: 1.28s
    619:	learn: 28094.7342994	total: 2.09s	remaining: 1.28s
    620:	learn: 28078.4638037	total: 2.1s	remaining: 1.28s
    621:	learn: 28057.6936953	total: 2.1s	remaining: 1.27s
    622:	learn: 28047.0246738	total: 2.1s	remaining: 1.27s
    623:	learn: 28034.9284558	total: 2.1s	remaining: 1.27s
    624:	learn: 28013.8702383	total: 2.11s	remaining: 1.26s
    625:	learn: 28013.4874867	total: 2.11s	remaining: 1.26s
    626:	learn: 28012.9010855	total: 2.12s	remaining: 1.26s
    627:	learn: 28012.6166287	total: 2.12s	remaining: 1.25s
    628:	learn: 27990.1016798	total: 2.12s	remaining: 1.25s
    629:	learn: 27977.0044581	total: 2.13s	remaining: 1.25s
    630:	learn: 27972.5292051	total: 2.13s	remaining: 1.25s
    631:	learn: 27970.8139231	total: 2.13s	remaining: 1.24s
    632:	learn: 27969.8653144	total: 2.14s	remaining: 1.24s
    633:	learn: 27952.4018628	total: 2.14s	remaining: 1.24s
    634:	learn: 27948.8636511	total: 2.14s	remaining: 1.23s
    635:	learn: 27929.4622829	total: 2.15s	remaining: 1.23s
    636:	learn: 27910.4968670	total: 2.15s	remaining: 1.23s
    637:	learn: 27895.3593202	total: 2.15s	remaining: 1.22s
    638:	learn: 27877.9621084	total: 2.16s	remaining: 1.22s
    639:	learn: 27877.5957324	total: 2.16s	remaining: 1.21s
    640:	learn: 27877.2332277	total: 2.16s	remaining: 1.21s
    641:	learn: 27866.3189366	total: 2.17s	remaining: 1.21s
    642:	learn: 27853.6918317	total: 2.17s	remaining: 1.2s
    643:	learn: 27841.6398266	total: 2.17s	remaining: 1.2s
    644:	learn: 27828.7853649	total: 2.17s	remaining: 1.2s
    645:	learn: 27821.5248693	total: 2.18s	remaining: 1.19s
    646:	learn: 27794.9564068	total: 2.18s	remaining: 1.19s
    647:	learn: 27793.7227123	total: 2.18s	remaining: 1.19s
    648:	learn: 27781.0958063	total: 2.19s	remaining: 1.18s
    649:	learn: 27765.9127192	total: 2.19s	remaining: 1.18s
    650:	learn: 27747.3129177	total: 2.19s	remaining: 1.18s
    651:	learn: 27728.1030742	total: 2.19s	remaining: 1.17s
    652:	learn: 27712.3025504	total: 2.2s	remaining: 1.17s
    653:	learn: 27698.9092952	total: 2.2s	remaining: 1.16s
    654:	learn: 27698.5718761	total: 2.2s	remaining: 1.16s
    655:	learn: 27688.0041765	total: 2.21s	remaining: 1.16s
    656:	learn: 27674.9896393	total: 2.21s	remaining: 1.15s
    657:	learn: 27674.8177633	total: 2.21s	remaining: 1.15s
    658:	learn: 27667.2014492	total: 2.22s	remaining: 1.15s
    659:	learn: 27643.8415056	total: 2.22s	remaining: 1.14s
    660:	learn: 27643.3781691	total: 2.22s	remaining: 1.14s
    661:	learn: 27634.8390849	total: 2.23s	remaining: 1.14s
    662:	learn: 27625.5382593	total: 2.23s	remaining: 1.13s
    663:	learn: 27609.1204659	total: 2.23s	remaining: 1.13s
    664:	learn: 27605.5285211	total: 2.24s	remaining: 1.13s
    665:	learn: 27585.0214595	total: 2.24s	remaining: 1.12s
    666:	learn: 27577.6365589	total: 2.24s	remaining: 1.12s
    667:	learn: 27575.6535974	total: 2.25s	remaining: 1.12s
    668:	learn: 27563.0204494	total: 2.25s	remaining: 1.11s
    669:	learn: 27549.0298957	total: 2.25s	remaining: 1.11s
    670:	learn: 27541.2922882	total: 2.26s	remaining: 1.11s
    671:	learn: 27540.6698577	total: 2.26s	remaining: 1.1s
    672:	learn: 27538.3392420	total: 2.26s	remaining: 1.1s
    673:	learn: 27517.6869652	total: 2.27s	remaining: 1.1s
    674:	learn: 27513.8175272	total: 2.27s	remaining: 1.09s
    675:	learn: 27503.6502211	total: 2.27s	remaining: 1.09s
    676:	learn: 27490.1100591	total: 2.28s	remaining: 1.09s
    677:	learn: 27478.5211388	total: 2.28s	remaining: 1.08s
    678:	learn: 27470.6854697	total: 2.28s	remaining: 1.08s
    679:	learn: 27451.2386733	total: 2.29s	remaining: 1.08s
    680:	learn: 27438.4879172	total: 2.29s	remaining: 1.07s
    681:	learn: 27438.1711596	total: 2.29s	remaining: 1.07s
    682:	learn: 27428.0125664	total: 2.3s	remaining: 1.07s
    683:	learn: 27427.6070280	total: 2.3s	remaining: 1.06s
    684:	learn: 27412.1148726	total: 2.3s	remaining: 1.06s
    685:	learn: 27410.6119712	total: 2.31s	remaining: 1.06s
    686:	learn: 27404.8184223	total: 2.31s	remaining: 1.05s
    687:	learn: 27379.6603948	total: 2.31s	remaining: 1.05s
    688:	learn: 27364.3112531	total: 2.32s	remaining: 1.05s
    689:	learn: 27353.4227683	total: 2.32s	remaining: 1.04s
    690:	learn: 27353.1564866	total: 2.32s	remaining: 1.04s
    691:	learn: 27347.0007167	total: 2.33s	remaining: 1.03s
    692:	learn: 27341.4485970	total: 2.33s	remaining: 1.03s
    693:	learn: 27335.2057266	total: 2.34s	remaining: 1.03s
    694:	learn: 27320.3452657	total: 2.34s	remaining: 1.03s
    695:	learn: 27312.4817639	total: 2.34s	remaining: 1.02s
    696:	learn: 27306.6696127	total: 2.35s	remaining: 1.02s
    697:	learn: 27295.8448465	total: 2.35s	remaining: 1.02s
    698:	learn: 27285.0234215	total: 2.35s	remaining: 1.01s
    699:	learn: 27272.1486202	total: 2.36s	remaining: 1.01s
    700:	learn: 27271.1529210	total: 2.36s	remaining: 1.01s
    701:	learn: 27266.2747421	total: 2.36s	remaining: 1s
    702:	learn: 27266.0261456	total: 2.37s	remaining: 1000ms
    703:	learn: 27246.8086514	total: 2.37s	remaining: 996ms
    704:	learn: 27246.5943689	total: 2.37s	remaining: 993ms
    705:	learn: 27246.3406194	total: 2.38s	remaining: 989ms
    706:	learn: 27228.6206425	total: 2.38s	remaining: 986ms
    707:	learn: 27228.4151839	total: 2.38s	remaining: 982ms
    708:	learn: 27205.8318010	total: 2.38s	remaining: 979ms
    709:	learn: 27196.7365845	total: 2.39s	remaining: 975ms
    710:	learn: 27184.4579213	total: 2.39s	remaining: 972ms
    711:	learn: 27175.8855665	total: 2.39s	remaining: 969ms
    712:	learn: 27172.1932286	total: 2.4s	remaining: 965ms
    713:	learn: 27171.9056014	total: 2.4s	remaining: 962ms
    714:	learn: 27169.6523993	total: 2.4s	remaining: 958ms
    715:	learn: 27169.2916716	total: 2.41s	remaining: 955ms
    716:	learn: 27153.8757233	total: 2.41s	remaining: 951ms
    717:	learn: 27132.7602655	total: 2.41s	remaining: 948ms
    718:	learn: 27129.5458144	total: 2.42s	remaining: 944ms
    719:	learn: 27109.1179089	total: 2.42s	remaining: 941ms
    720:	learn: 27102.1948575	total: 2.42s	remaining: 937ms
    721:	learn: 27088.8857415	total: 2.42s	remaining: 934ms
    722:	learn: 27079.4800543	total: 2.43s	remaining: 930ms
    723:	learn: 27079.2551735	total: 2.43s	remaining: 927ms
    724:	learn: 27071.9219352	total: 2.43s	remaining: 923ms
    725:	learn: 27059.7416799	total: 2.44s	remaining: 921ms
    726:	learn: 27036.0621884	total: 2.44s	remaining: 917ms
    727:	learn: 27024.5930278	total: 2.45s	remaining: 914ms
    728:	learn: 27012.5473817	total: 2.45s	remaining: 911ms
    729:	learn: 26998.5434964	total: 2.45s	remaining: 908ms
    730:	learn: 26972.1685280	total: 2.46s	remaining: 905ms
    731:	learn: 26966.0865382	total: 2.46s	remaining: 901ms
    732:	learn: 26958.1898759	total: 2.46s	remaining: 898ms
    733:	learn: 26948.5913680	total: 2.47s	remaining: 894ms
    734:	learn: 26948.1503376	total: 2.47s	remaining: 891ms
    735:	learn: 26937.0931479	total: 2.47s	remaining: 887ms
    736:	learn: 26927.2565826	total: 2.48s	remaining: 884ms
    737:	learn: 26917.6066751	total: 2.48s	remaining: 880ms
    738:	learn: 26916.9837109	total: 2.48s	remaining: 877ms
    739:	learn: 26916.6493229	total: 2.49s	remaining: 874ms
    740:	learn: 26905.3189955	total: 2.49s	remaining: 870ms
    741:	learn: 26895.1223172	total: 2.49s	remaining: 867ms
    742:	learn: 26878.1574869	total: 2.5s	remaining: 863ms
    743:	learn: 26872.3652594	total: 2.5s	remaining: 860ms
    744:	learn: 26859.8646587	total: 2.5s	remaining: 856ms
    745:	learn: 26844.4687327	total: 2.5s	remaining: 853ms
    746:	learn: 26827.9978366	total: 2.51s	remaining: 850ms
    747:	learn: 26817.3008566	total: 2.51s	remaining: 846ms
    748:	learn: 26814.2415907	total: 2.51s	remaining: 843ms
    749:	learn: 26814.0595147	total: 2.52s	remaining: 839ms
    750:	learn: 26800.7338836	total: 2.52s	remaining: 836ms
    751:	learn: 26791.2088464	total: 2.52s	remaining: 832ms
    752:	learn: 26772.0692995	total: 2.53s	remaining: 829ms
    753:	learn: 26767.4542366	total: 2.53s	remaining: 826ms
    754:	learn: 26758.3945391	total: 2.53s	remaining: 822ms
    755:	learn: 26741.4367300	total: 2.54s	remaining: 819ms
    756:	learn: 26731.7806573	total: 2.54s	remaining: 815ms
    757:	learn: 26731.5138684	total: 2.54s	remaining: 812ms
    758:	learn: 26718.3782080	total: 2.55s	remaining: 809ms
    759:	learn: 26699.5864120	total: 2.55s	remaining: 805ms
    760:	learn: 26686.3184537	total: 2.55s	remaining: 802ms
    761:	learn: 26671.6915121	total: 2.56s	remaining: 798ms
    762:	learn: 26659.8604499	total: 2.56s	remaining: 795ms
    763:	learn: 26645.2878994	total: 2.56s	remaining: 791ms
    764:	learn: 26628.0492616	total: 2.56s	remaining: 788ms
    765:	learn: 26617.2108302	total: 2.57s	remaining: 785ms
    766:	learn: 26617.0753361	total: 2.57s	remaining: 781ms
    767:	learn: 26606.1949846	total: 2.57s	remaining: 778ms
    768:	learn: 26595.1292509	total: 2.58s	remaining: 774ms
    769:	learn: 26588.2538322	total: 2.58s	remaining: 771ms
    770:	learn: 26574.3046428	total: 2.58s	remaining: 768ms
    771:	learn: 26558.0409614	total: 2.59s	remaining: 764ms
    772:	learn: 26542.0882970	total: 2.59s	remaining: 761ms
    773:	learn: 26537.4157447	total: 2.59s	remaining: 757ms
    774:	learn: 26519.4542599	total: 2.6s	remaining: 754ms
    775:	learn: 26519.1044562	total: 2.6s	remaining: 751ms
    776:	learn: 26504.2466356	total: 2.6s	remaining: 747ms
    777:	learn: 26495.3839174	total: 2.61s	remaining: 744ms
    778:	learn: 26493.6373891	total: 2.61s	remaining: 741ms
    779:	learn: 26487.2910916	total: 2.61s	remaining: 738ms
    780:	learn: 26469.0675764	total: 2.62s	remaining: 734ms
    781:	learn: 26459.0878510	total: 2.62s	remaining: 731ms
    782:	learn: 26458.9398817	total: 2.63s	remaining: 728ms
    783:	learn: 26455.8526221	total: 2.63s	remaining: 724ms
    784:	learn: 26443.3085484	total: 2.63s	remaining: 721ms
    785:	learn: 26434.1408269	total: 2.63s	remaining: 717ms
    786:	learn: 26419.6581360	total: 2.64s	remaining: 714ms
    787:	learn: 26405.6174817	total: 2.64s	remaining: 711ms
    788:	learn: 26405.3429894	total: 2.64s	remaining: 707ms
    789:	learn: 26388.6514879	total: 2.65s	remaining: 704ms
    790:	learn: 26379.8637424	total: 2.65s	remaining: 700ms
    791:	learn: 26372.4253928	total: 2.65s	remaining: 697ms
    792:	learn: 26359.9028167	total: 2.66s	remaining: 694ms
    793:	learn: 26351.0544422	total: 2.66s	remaining: 690ms
    794:	learn: 26350.2262592	total: 2.66s	remaining: 687ms
    795:	learn: 26338.7550872	total: 2.67s	remaining: 683ms
    796:	learn: 26338.3434144	total: 2.67s	remaining: 680ms
    797:	learn: 26322.5405210	total: 2.67s	remaining: 677ms
    798:	learn: 26309.0833278	total: 2.68s	remaining: 673ms
    799:	learn: 26294.0581756	total: 2.68s	remaining: 670ms
    800:	learn: 26282.1808342	total: 2.68s	remaining: 666ms
    801:	learn: 26270.6392303	total: 2.69s	remaining: 663ms
    802:	learn: 26245.7315424	total: 2.69s	remaining: 660ms
    803:	learn: 26237.3142975	total: 2.69s	remaining: 656ms
    804:	learn: 26230.8341769	total: 2.69s	remaining: 653ms
    805:	learn: 26214.3421236	total: 2.7s	remaining: 649ms
    806:	learn: 26208.2866050	total: 2.7s	remaining: 646ms
    807:	learn: 26194.0493527	total: 2.7s	remaining: 642ms
    808:	learn: 26193.6803092	total: 2.71s	remaining: 639ms
    809:	learn: 26184.3432169	total: 2.71s	remaining: 636ms
    810:	learn: 26184.1901271	total: 2.71s	remaining: 632ms
    811:	learn: 26174.4610456	total: 2.71s	remaining: 629ms
    812:	learn: 26167.8336127	total: 2.72s	remaining: 625ms
    813:	learn: 26167.3808815	total: 2.72s	remaining: 622ms
    814:	learn: 26154.9864408	total: 2.72s	remaining: 618ms
    815:	learn: 26153.8168248	total: 2.73s	remaining: 615ms
    816:	learn: 26153.6655948	total: 2.73s	remaining: 612ms
    817:	learn: 26131.4758672	total: 2.73s	remaining: 608ms
    818:	learn: 26123.3319005	total: 2.74s	remaining: 605ms
    819:	learn: 26120.2656452	total: 2.74s	remaining: 602ms
    820:	learn: 26114.5798934	total: 2.74s	remaining: 598ms
    821:	learn: 26086.1956712	total: 2.75s	remaining: 595ms
    822:	learn: 26067.9031576	total: 2.75s	remaining: 591ms
    823:	learn: 26059.4118202	total: 2.75s	remaining: 588ms
    824:	learn: 26052.8441402	total: 2.76s	remaining: 585ms
    825:	learn: 26044.2297265	total: 2.76s	remaining: 581ms
    826:	learn: 26035.5935632	total: 2.76s	remaining: 578ms
    827:	learn: 26024.5475067	total: 2.77s	remaining: 575ms
    828:	learn: 26007.2529486	total: 2.77s	remaining: 571ms
    829:	learn: 25998.7552313	total: 2.77s	remaining: 568ms
    830:	learn: 25987.5907692	total: 2.78s	remaining: 565ms
    831:	learn: 25981.3675435	total: 2.78s	remaining: 562ms
    832:	learn: 25979.7330834	total: 2.79s	remaining: 558ms
    833:	learn: 25975.9805840	total: 2.79s	remaining: 555ms
    834:	learn: 25964.6035404	total: 2.79s	remaining: 552ms
    835:	learn: 25964.4248536	total: 2.79s	remaining: 548ms
    836:	learn: 25959.4500257	total: 2.8s	remaining: 545ms
    837:	learn: 25958.0520174	total: 2.8s	remaining: 542ms
    838:	learn: 25957.8987813	total: 2.81s	remaining: 538ms
    839:	learn: 25955.8869708	total: 2.81s	remaining: 535ms
    840:	learn: 25946.2686721	total: 2.81s	remaining: 532ms
    841:	learn: 25938.7202236	total: 2.82s	remaining: 529ms
    842:	learn: 25923.9577886	total: 2.82s	remaining: 525ms
    843:	learn: 25921.6340071	total: 2.82s	remaining: 522ms
    844:	learn: 25916.1260064	total: 2.83s	remaining: 519ms
    845:	learn: 25915.8983515	total: 2.83s	remaining: 515ms
    846:	learn: 25913.6220834	total: 2.83s	remaining: 512ms
    847:	learn: 25901.8434595	total: 2.84s	remaining: 509ms
    848:	learn: 25885.2380248	total: 2.84s	remaining: 506ms
    849:	learn: 25873.9897815	total: 2.85s	remaining: 503ms
    850:	learn: 25865.0738707	total: 2.85s	remaining: 500ms
    851:	learn: 25846.1599895	total: 2.86s	remaining: 497ms
    852:	learn: 25833.4755269	total: 2.86s	remaining: 493ms
    853:	learn: 25819.3097546	total: 2.86s	remaining: 490ms
    854:	learn: 25802.9817330	total: 2.87s	remaining: 487ms
    855:	learn: 25793.8967964	total: 2.87s	remaining: 483ms
    856:	learn: 25789.1517935	total: 2.88s	remaining: 480ms
    857:	learn: 25784.9785368	total: 2.88s	remaining: 476ms
    858:	learn: 25775.6780461	total: 2.88s	remaining: 473ms
    859:	learn: 25767.6698711	total: 2.89s	remaining: 470ms
    860:	learn: 25746.4177216	total: 2.89s	remaining: 466ms
    861:	learn: 25742.3695268	total: 2.89s	remaining: 463ms
    862:	learn: 25731.7796040	total: 2.9s	remaining: 460ms
    863:	learn: 25731.6387665	total: 2.9s	remaining: 456ms
    864:	learn: 25718.4450056	total: 2.9s	remaining: 453ms
    865:	learn: 25700.5411924	total: 2.91s	remaining: 450ms
    866:	learn: 25695.4600025	total: 2.91s	remaining: 446ms
    867:	learn: 25675.5040061	total: 2.91s	remaining: 443ms
    868:	learn: 25667.3610223	total: 2.92s	remaining: 440ms
    869:	learn: 25653.6170214	total: 2.92s	remaining: 437ms
    870:	learn: 25641.1075286	total: 2.93s	remaining: 433ms
    871:	learn: 25620.8701018	total: 2.93s	remaining: 430ms
    872:	learn: 25604.1611441	total: 2.93s	remaining: 427ms
    873:	learn: 25591.4170990	total: 2.94s	remaining: 423ms
    874:	learn: 25587.1724957	total: 2.94s	remaining: 420ms
    875:	learn: 25568.5963544	total: 2.94s	remaining: 417ms
    876:	learn: 25557.8386198	total: 2.95s	remaining: 414ms
    877:	learn: 25548.5719271	total: 2.95s	remaining: 410ms
    878:	learn: 25532.5930200	total: 2.96s	remaining: 407ms
    879:	learn: 25526.0280294	total: 2.96s	remaining: 403ms
    880:	learn: 25512.3923989	total: 2.96s	remaining: 400ms
    881:	learn: 25503.0194940	total: 2.96s	remaining: 397ms
    882:	learn: 25486.9778673	total: 2.97s	remaining: 393ms
    883:	learn: 25475.2009941	total: 2.97s	remaining: 390ms
    884:	learn: 25457.2323657	total: 2.97s	remaining: 386ms
    885:	learn: 25444.2802341	total: 2.98s	remaining: 383ms
    886:	learn: 25436.7941959	total: 2.98s	remaining: 380ms
    887:	learn: 25418.7504499	total: 2.98s	remaining: 376ms
    888:	learn: 25408.8718543	total: 2.98s	remaining: 373ms
    889:	learn: 25398.8125270	total: 2.99s	remaining: 369ms
    890:	learn: 25389.4523068	total: 2.99s	remaining: 366ms
    891:	learn: 25382.6071525	total: 2.99s	remaining: 363ms
    892:	learn: 25370.2027684	total: 3s	remaining: 359ms
    893:	learn: 25364.4690074	total: 3s	remaining: 356ms
    894:	learn: 25352.2645554	total: 3s	remaining: 352ms
    895:	learn: 25340.9332388	total: 3.01s	remaining: 349ms
    896:	learn: 25338.3943220	total: 3.01s	remaining: 346ms
    897:	learn: 25326.7166309	total: 3.01s	remaining: 342ms
    898:	learn: 25314.4565648	total: 3.02s	remaining: 339ms
    899:	learn: 25309.2515922	total: 3.02s	remaining: 336ms
    900:	learn: 25291.9233241	total: 3.02s	remaining: 332ms
    901:	learn: 25289.5912009	total: 3.03s	remaining: 329ms
    902:	learn: 25285.7395880	total: 3.03s	remaining: 325ms
    903:	learn: 25277.5361220	total: 3.03s	remaining: 322ms
    904:	learn: 25272.5983210	total: 3.04s	remaining: 319ms
    905:	learn: 25262.7790492	total: 3.04s	remaining: 315ms
    906:	learn: 25245.9400966	total: 3.04s	remaining: 312ms
    907:	learn: 25233.1898642	total: 3.04s	remaining: 309ms
    908:	learn: 25227.3616575	total: 3.05s	remaining: 305ms
    909:	learn: 25218.8602898	total: 3.05s	remaining: 302ms
    910:	learn: 25204.7999613	total: 3.06s	remaining: 298ms
    911:	learn: 25195.6236941	total: 3.06s	remaining: 295ms
    912:	learn: 25178.5213742	total: 3.06s	remaining: 292ms
    913:	learn: 25172.3619843	total: 3.06s	remaining: 288ms
    914:	learn: 25166.6187940	total: 3.07s	remaining: 285ms
    915:	learn: 25154.6254608	total: 3.07s	remaining: 282ms
    916:	learn: 25142.6557308	total: 3.07s	remaining: 278ms
    917:	learn: 25126.2299012	total: 3.08s	remaining: 275ms
    918:	learn: 25115.3999940	total: 3.08s	remaining: 271ms
    919:	learn: 25101.2033206	total: 3.08s	remaining: 268ms
    920:	learn: 25083.4965685	total: 3.09s	remaining: 265ms
    921:	learn: 25082.5508209	total: 3.09s	remaining: 261ms
    922:	learn: 25077.7216444	total: 3.09s	remaining: 258ms
    923:	learn: 25071.9231907	total: 3.1s	remaining: 255ms
    924:	learn: 25055.8187824	total: 3.1s	remaining: 251ms
    925:	learn: 25048.9602330	total: 3.1s	remaining: 248ms
    926:	learn: 25041.7634115	total: 3.11s	remaining: 245ms
    927:	learn: 25032.9113609	total: 3.11s	remaining: 241ms
    928:	learn: 25024.3812511	total: 3.12s	remaining: 238ms
    929:	learn: 25018.7259168	total: 3.12s	remaining: 235ms
    930:	learn: 25006.8646179	total: 3.12s	remaining: 231ms
    931:	learn: 24993.5783552	total: 3.12s	remaining: 228ms
    932:	learn: 24988.0324328	total: 3.13s	remaining: 225ms
    933:	learn: 24980.1969659	total: 3.13s	remaining: 221ms
    934:	learn: 24971.7256945	total: 3.13s	remaining: 218ms
    935:	learn: 24963.5902258	total: 3.14s	remaining: 215ms
    936:	learn: 24954.0264969	total: 3.14s	remaining: 211ms
    937:	learn: 24942.2503828	total: 3.14s	remaining: 208ms
    938:	learn: 24937.2724982	total: 3.15s	remaining: 204ms
    939:	learn: 24929.2841625	total: 3.15s	remaining: 201ms
    940:	learn: 24914.1221176	total: 3.15s	remaining: 198ms
    941:	learn: 24906.9853096	total: 3.16s	remaining: 194ms
    942:	learn: 24901.5335296	total: 3.16s	remaining: 191ms
    943:	learn: 24895.9359012	total: 3.16s	remaining: 188ms
    944:	learn: 24885.8891683	total: 3.17s	remaining: 184ms
    945:	learn: 24869.3289658	total: 3.17s	remaining: 181ms
    946:	learn: 24857.4844584	total: 3.17s	remaining: 178ms
    947:	learn: 24850.1350142	total: 3.18s	remaining: 174ms
    948:	learn: 24831.5426380	total: 3.18s	remaining: 171ms
    949:	learn: 24820.4710430	total: 3.19s	remaining: 168ms
    950:	learn: 24805.8606697	total: 3.19s	remaining: 164ms
    951:	learn: 24795.2701657	total: 3.19s	remaining: 161ms
    952:	learn: 24782.0420402	total: 3.2s	remaining: 158ms
    953:	learn: 24781.1188351	total: 3.2s	remaining: 154ms
    954:	learn: 24771.0470649	total: 3.21s	remaining: 151ms
    955:	learn: 24756.5570939	total: 3.21s	remaining: 148ms
    956:	learn: 24743.1743382	total: 3.21s	remaining: 144ms
    957:	learn: 24730.9553641	total: 3.22s	remaining: 141ms
    958:	learn: 24726.5364690	total: 3.22s	remaining: 138ms
    959:	learn: 24720.8543844	total: 3.22s	remaining: 134ms
    960:	learn: 24710.2078783	total: 3.23s	remaining: 131ms
    961:	learn: 24698.0604201	total: 3.23s	remaining: 128ms
    962:	learn: 24685.2125230	total: 3.24s	remaining: 124ms
    963:	learn: 24676.4012447	total: 3.24s	remaining: 121ms
    964:	learn: 24671.5981137	total: 3.25s	remaining: 118ms
    965:	learn: 24660.3519811	total: 3.25s	remaining: 114ms
    966:	learn: 24654.1178239	total: 3.25s	remaining: 111ms
    967:	learn: 24647.9111499	total: 3.26s	remaining: 108ms
    968:	learn: 24643.5486643	total: 3.26s	remaining: 104ms
    969:	learn: 24638.9770098	total: 3.27s	remaining: 101ms
    970:	learn: 24634.1237118	total: 3.27s	remaining: 97.7ms
    971:	learn: 24624.9299680	total: 3.27s	remaining: 94.4ms
    972:	learn: 24614.1734998	total: 3.28s	remaining: 91ms
    973:	learn: 24602.7994007	total: 3.28s	remaining: 87.6ms
    974:	learn: 24601.0745900	total: 3.29s	remaining: 84.3ms
    975:	learn: 24600.7191461	total: 3.29s	remaining: 80.9ms
    976:	learn: 24592.0116073	total: 3.29s	remaining: 77.5ms
    977:	learn: 24582.3011481	total: 3.3s	remaining: 74.2ms
    978:	learn: 24575.4010677	total: 3.3s	remaining: 70.9ms
    979:	learn: 24575.3169358	total: 3.31s	remaining: 67.5ms
    980:	learn: 24566.4777899	total: 3.31s	remaining: 64.1ms
    981:	learn: 24556.9972592	total: 3.31s	remaining: 60.7ms
    982:	learn: 24549.5926605	total: 3.32s	remaining: 57.4ms
    983:	learn: 24541.4082905	total: 3.32s	remaining: 54ms
    984:	learn: 24536.7968259	total: 3.32s	remaining: 50.6ms
    985:	learn: 24530.7276864	total: 3.33s	remaining: 47.3ms
    986:	learn: 24521.4097400	total: 3.33s	remaining: 43.9ms
    987:	learn: 24521.0198409	total: 3.34s	remaining: 40.5ms
    988:	learn: 24520.9015991	total: 3.34s	remaining: 37.1ms
    989:	learn: 24512.3114985	total: 3.35s	remaining: 33.8ms
    990:	learn: 24503.8823626	total: 3.35s	remaining: 30.4ms
    991:	learn: 24503.6610596	total: 3.35s	remaining: 27ms
    992:	learn: 24502.7457525	total: 3.36s	remaining: 23.7ms
    993:	learn: 24490.1166911	total: 3.36s	remaining: 20.3ms
    994:	learn: 24488.5023143	total: 3.36s	remaining: 16.9ms
    995:	learn: 24468.9731701	total: 3.37s	remaining: 13.5ms
    996:	learn: 24459.0497225	total: 3.37s	remaining: 10.1ms
    997:	learn: 24458.7284279	total: 3.38s	remaining: 6.76ms
    998:	learn: 24448.9981948	total: 3.38s	remaining: 3.38ms
    999:	learn: 24440.7422679	total: 3.38s	remaining: 0us
    




    0.38351169878113034




```python

```
