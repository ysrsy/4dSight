import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

vr2d = np.load('vr2d.npy')
vr3d = np.load('vr3d.npy')
calibration = np.array([[100,0,960],[0,100,540],[0,0,1]]).astype(np.float32)
distortion = np.zeros((4,1))

img1 = cv.imread("img1.png")
img2 = cv.imread("img2.png")
img3 = cv.imread("img3.png")
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
gray3 = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)


success, rotationVector1, translationVector1 = cv.solvePnP(vr3d,vr2d,calibration,distortion)
feature_params = dict( maxCorners = 100,qualityLevel = 0.3,minDistance = 7,blockSize = 7 )
  
# Parameters for lucas kanade optical flow
lk_params = dict( winSize = (15, 15),maxLevel = 2,criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,10, 0.03))

p0 = vr2d

# calculate optical flow
p2, st, err = cv.calcOpticalFlowPyrLK(gray1,gray2,p0, None,**lk_params)
p3, st, err = cv.calcOpticalFlowPyrLK(gray1,gray3,p0, None,**lk_params)

success, rotationVector2, translationVector2 = cv.solvePnP(vr3d,p2,calibration,distortion)
success, rotationVector3, translationVector3 = cv.solvePnP(vr3d,p3,calibration,distortion)

# to plot the trajectry of cameras I only used translation values

points2 = np.zeros((3,3)) 
points2[0] = translationVector1.T
points2[1] = translationVector2.T
points2[2] = translationVector3.T
points2 = points2.T 


fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot(points2[0], points2[1], points2[2], marker = 'x')
ax.scatter(*points2.T[0], color = 'red')


plt.show()

"""
// AMA OKUDUĞUM KADARIYLA OPTICAL FLOW İLE DİREKT O 20 NOKTANIN 2. YA DA 3. RESİMDEKİ YERLERİNİ TESPİT EDEBİLDİĞİMİZİ SÖYLÜYORLARDI(LUCAS KANADE METODU İLE)
// (TABİ 20 NOKTANIN HEPSİ 2. YADA 3. RESİMDE OLMAYABİLİRDİ) FAKAT VERİLEN İNPUTLAR İMAGE 2 VE İMAGE 3 TE BULUNUYOR
// DAHA SONRA İMAGE 1 E UYGULADIĞIM GİBİ DİĞER İMAGELARADA OPTİCAL FLOWDAN TENİ BULDUĞUM 2D POİNTLERİ İLE PNPN UYGULADIM
//ROTATİON VE TRANSLATİONLARI BULDUM DAHA SONRA BU TRANSLATIONLARI SIRASI İLE BASIP TRAJECTROY YAPAYA ÇALIŞTIM 
"""



""" ALTTA KALAN KISMIN HATALI OLDUĞUNU DÜŞÜNDÜĞÜM İÇİN VAZGEÇTİM"""


"""
// BURDA SIFT UYGULAYARAK KEYPOINT BULUP O KEYPOINTLERİ MATCH EDİP,
// MATCH OLANLAR IÇINDE VERİLEN LİSTEDEKİ 20 POINTI ARAYIP ONLARA GÖRE KAMERANIN LOCATIONINI BULUP TRAJECTORY ÇİZDİRMEYİ DÜŞÜNMÜŞTÜM
// FAKAT DOĞRU BİR YAKLAŞIM OLMADIĞINI ANLADIM
// BULDUĞUMUZ NOKTALARIN NE KADAR YAKIN OLDUĞUNU ANLAMAK IÇIN BİR NOKTAYA BAKIP YAKIN OLANLARI VE ARADAKİ KÜÇÜK FARKI BULMAYA ÇALIŞTIM ANLAMSIZ GELEBİLİR IF TE YAZDIĞIM YER 



sift = cv.xfeatures2d.SIFT_create(5000) # initialize SIFT 


# DETECT AND DRAW KEYPOINTS
# sift.detect() returns a list of keypoints
# keypoint is a standard class of opencv (not just SIFT-related)
keyPoints1 ,descriptors1 = sift.detectAndCompute(gray1,None)
keyPoints2 ,descriptors2 = sift.detectAndCompute(gray2,None)
keyPoints3 ,descriptors3 = sift.detectAndCompute(gray3,None)




bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1,descriptors2)






for mat in matches:
    # Get the matching keypoints for each of the images
    img1_idx = mat.queryIdx
    

    # x - columns
    # y - rows
    # Get the coordinates
    (x1, y1) = keyPoints1[img1_idx].pt
    if(x1<861 and x1>859):
        print(x1)
    #for i in range(0,20):
    #    if vr2d[i][0][0]==x1 or vr2d[i][0][1]==y1:
    #        print(x1,y1)

    

img_1=cv.drawKeypoints(gray1,keyPoints1, img1) # mae new image with keypoints drawn
plt.imshow(img_1)
plt.show()"""

