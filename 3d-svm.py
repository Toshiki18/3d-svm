import numpy as np
import pandas as pd

df0 = pd.read_csv('/Users/sample/train.csv', header = None)

df0.columns = ['1つ目', '2つ目', '3つ目', '4つ目', '5つ目','class label']
df0.head()

data = pd.read_csv("/Users/sample/mesh3.csv",  header = None)
data.columns = ['X', 'Y', 'Z', 'label']
data.head()

from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

y0 = df0.iloc[0:, 5].values
#1列目
X1  = np.copy(df0.iloc[0:, 0].values)
#2列目
X2 = np.copy(df0.iloc[0:, 1].values)
#3列目
X3  = np.copy(df0.iloc[0:, 2].values)
#4列目
X4  = np.copy(df0.iloc[0:, 3].values)
#5列目
X5   = np.copy(df0.iloc[0:, 4].values)

X0 = df0.iloc[0:, [1,2,3]].values

y1 = data.iloc[0:, 3].values

X = np.copy(data.iloc[0:, 0].values)

Y = np.copy(data.iloc[0:, 1].values)

Z = np.copy(data.iloc[0:, 2].values)

XYZ = data.iloc[0:, [0,1,2]].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X0, y0, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
#標準化
sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)

X_test_std  = sc.transform(X_test)

from sklearn.svm import SVC

svm = SVC(kernel = 'linear', C=10, random_state=0)

svm.fit(X_train_std,y_train)

y_pred = svm.predict(X_test_std)

y_pred2 = svm.predict(XYZ)

CL_lis = ['red','blue','green','grey','pink','magenta','salmon', 'blue', 'pink']

MK_lis = ['o','v','^','+','D','o', 'D', '^', 'D']

LB_lis = ['1','2','3', '4', '5','6','7','8', '9']

St = ""

X0_std = sc.transform(X0)


def render_frame(angle):
    global X0_std
    global XYZ
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, angle)
    plt.close()
    #ax.set_xlim(-2.0, 2.5)
    ax.set_ylim(-2.5, 2.5)
    #ax.set_zlim(-3.5, 3.5)
    ax.set_xlabel("V1-V3/5 pair density (normalized)")
    ax.set_ylabel("V1-V6 pair density (normalized)")
    ax.set_zlabel("V1-V4 pair density (normalized)")

    
    
    for idx in range(len(np.unique(y0))):
        ax.scatter(X0_std[y0==idx+1, 0], X0_std[y0==idx+1, 1],  X0_std[y0==idx+1, 2],
                   color=CL_lis[idx], marker=MK_lis[idx], label=LB_lis[idx], linestyle='None', s = 70)
        
    ax.scatter(XYZ[y_pred2 == 1, 0], XYZ[y_pred2 == 1, 1], XYZ[y_pred2 == 1, 2], 
              color = 'red', marker = 'o', label = "1(scatter)", linestyle = 'None', s = 20)

    ax.scatter(XYZ[y_pred2 == 2, 0], XYZ[y_pred2 == 2, 1], XYZ[y_pred2 == 2, 2],
               color = 'blue', marker = 'v', label = "2(scatter)", linestyle = 'None', s = 20)

    ax.scatter(XYZ[y_pred2 == 3, 0], XYZ[y_pred2 == 3, 1], XYZ[y_pred2 == 3, 2],
               color = 'green', marker = '^', label = "3(scatter)", linestyle = 'None', s = 20)

    ax.scatter(XYZ[y_pred2 == 4, 0], XYZ[y_pred2 == 4, 1], XYZ[y_pred2 == 4, 2],
               color = 'grey', marker = '+', label = "4(scatter)", linestyle = 'None', s = 40)

    ax.scatter(XYZ[y_pred2 == 5, 0], XYZ[y_pred2 == 5, 1], XYZ[y_pred2 == 5, 2],
               color = 'pink', marker = 'D', label = "5(scatter)", linestyle = 'None', s = 20)

    ax.scatter(XYZ[y_pred2 == 6, 0], XYZ[y_pred2 == 6, 1], XYZ[y_pred2 == 6, 2],
               color = 'magenta', marker = 'o', label = "6(scatter)", linestyle = 'None', s = 20)

    ax.scatter(XYZ[y_pred2 == 7, 0], XYZ[y_pred2 == 7, 1], XYZ[y_pred2 == 7, 2],
               color = 'salmon', marker = 'D', label = "7(scatter)", linestyle = 'None', s = 20)

    ax.scatter(XYZ[y_pred2 == 8, 0], XYZ[y_pred2 == 8, 1], XYZ[y_pred2 == 8, 2],
               color = 'blue', marker = '^', label = "8(scatter)", linestyle = 'None', s = 20)
    
    buf = BytesIO()
    fig.savefig(buf, bbox_inches='tight', pad_inches=0.0)
    return Image.open(buf)

render_frame(60)

images = [render_frame(angle) for angle in range(360)]
images[0].save('3d-svm-T8.gif', save_all=True, append_images=images[1:], duration=100, loop=0)