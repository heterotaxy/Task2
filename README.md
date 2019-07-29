# Task2

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def Press(event):
    global x1,y1,Dragflag,clikIm,backIm,dft_shift,waveIm
    if (event.xdata is None) or (event.ydata is None):
        return
    X = int(round(event.xdata))
    Y = int(round(event.ydata))
    x1 = X
    y1 = Y
    x1 = np.abs(X-cols//2)
    y1 = np.abs(Y-raws//2)
    Dragflag = True
    waveIm = F_re(x1,y1,raws,cols)
    backIm = makeIm(x1,y1,backIm,dft_shift,clikIm)
    clikIm = whereClik(X,Y,clikIm)
    im1.set_data(backIm)
    im2.set_data(clikIm)
    im3.set_data(waveIm)
    plt.draw()


def Drag(event):
    global x2,y2,Dragflag,clikIm,backIm,dft_shift,waveIm
    if Dragflag == False:
        return
    if (event.xdata is None) or (event.ydata is None):
        return
    X = int(round(event.xdata))
    Y = int(round(event.ydata))
    x2 = X
    y2 = Y
    x2 = np.abs(X-cols//2)
    y2 = np.abs(Y-raws//2)
    waveIm = F_re(x2,y2,raws,cols)
    backIm = makeIm(x2,y2,backIm,dft_shift,clikIm)
    clikIm = whereClik(X,Y,clikIm)
    im3.set_data(waveIm)
    im1.set_data(backIm)
    im2.set_data(clikIm)
    plt.draw()

def Release(event):
    global Dragflag
    Dragflag = False

def F_re(x,y,raws,cols):
    A = np.zeros(shape = (raws,cols))
    A[y][x] = 1
    A = cv2.idft(A)
    return A

def whereClik(x,y,image):
    A = image
    A[y,x] = 1
    return A

def makeIm(x,y,Back,Spec,Click):
    if(Click[y][x] == 1):
        return Back
    A = np.zeros(Spec.shape)
    A[:,:,0] = Click
    A[:,:,1] = Click
    A[y,x] = 1
    Copy = Spec*A
    Copy = np.fft.ifftshift(Copy)
    Copy = cv2.idft(Copy)
    Copy = cv2.magnitude(Copy[:,:,0],Copy[:,:,1])
    Back = Copy
    return Back

img = cv2.imread('images.JPEG')
img = cv2.resize(img,(50,50))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

x1 = 0
y1 = 0
x2 = 1
y2 = 1
Dragflag = False

raws,cols = gray.shape

backIm = np.zeros(shape = (raws,cols))
clikIm = np.zeros(shape = (raws,cols))
waveIm = np.zeros(shape = (raws,cols))

dft = cv2.dft(np.float32(gray),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])
magIm = 20*np.log(magnitude_spectrum)

waveIm = F_re(x1,y1,raws,cols)
backIm = makeIm(x1,y1,backIm,dft_shift,clikIm)
clikImIm = whereClik(x1+cols//2,y1+raws//2,clikIm)

waveIm = F_re(x2,y2,raws,cols)
backIm = makeIm(x2,y2,backIm,dft_shift,clikIm)
clikImIm = whereClik(x2+cols//2,y2+raws//2,clikIm)

plt.close('all')
plt.figure(figsize=(8,4))

plt.subplot(231),plt.imshow(gray, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(232),plt.imshow(magIm, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.subplot(234)
im1=plt.imshow(backIm, cmap = 'gray',vmin = 200000,  vmax  =600000)
plt.title('BACK IMAGE'),plt.xticks([]), plt.yticks([])

plt.subplot(235)
im2=plt.imshow(clikIm, cmap = 'gray')
plt.title('CLICK ON HERE'),plt.xticks([]), plt.yticks([])

plt.subplot(236)
im3=plt.imshow(waveIm, cmap = 'gray')
plt.title('WAVE IMAGE'),plt.xticks([]), plt.yticks([])

plt.connect('button_press_event', Press)
plt.connect('motion_notify_event',Drag)
plt.connect('button_release_event', Release)

plt.show()

```

参照したページ
クリックイベントを参照したページ
（https://qiita.com/ceptree/items/c547116bda4a5db11596）


np.fft.fft2やcv2.dftの参照ページ
（http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html）

![実行結果gif](https://raw.github.com/wiki/heterotaxy/Task2/image/Task.mov.gif)