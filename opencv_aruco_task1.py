import cv2
import numpy as np
import cv2.aruco as aruco
import imutils
import math


width= 1200
hieght = 750
def arucoDetect(img):
    gray = cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)
    key = getattr(aruco , f'DICT_5X5_250')
    dict = aruco.Dictionary_get(key)
    arucoparam = aruco.DetectorParameters_create()
    (corners,ids,rejected) = aruco.detectMarkers(img ,dict ,parameters = arucoparam)
    return corners , ids , rejected

def aruco_corners(img):
    (c,i,r) = arucoDetect(img)
    if len(c)>0:
        i = i.flatten()
        for(markercorner , markerid) in zip(c ,i):
            corner = markercorner.reshape((4,2))
            (tl,tr,br,bl)=corner
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))
        return tl,tr,br,bl

def distance(x1 ,y1 ,x2, y2):
    dist =int(math.sqrt((x2-x1)**2 + (y2-y1)**2))
    return dist


def crop(img):
    tl,tr,br,bl=aruco_corners(img)
    arr = [tl,tr,br,bl]
    xmax = arr[0][0]
    xmin = arr[0][1]
    ymax = arr[1][0]
    ymin = arr[1][1]
    for i in arr:
        if i[0]>xmax:
            xmax =i[0]
        if i[0]<xmin:
            xmin =i[0]
        if i[1]>ymax:
            xmax =i[1]
        if i[1]<ymin:
            xmax =i[1]
    fin =img[ymin:ymax,xmin:xmax]
    return fin



def arucoSet(img):
    tl, tr, br, bl = aruco_corners(img)
    slope = (tl[1] - tr[1])/(tl[0] - tr[0])
    theta = math.degrees(math.atan(slope))
    rot = imutils.rotate_bound(img, -theta)
    cimg = crop(rot)
    return cimg


listAr1 = [cv2.imread("LMAO.jpg") , cv2.imread("HaHa.jpg") ,cv2.imread("Ha.jpg") ,cv2.imread("XD.jpg")]
listar2 =[]

def namearuco(img):
    (c, i, r) = arucoDetect(img)
    if(i==1):
        a1 = img
        listar2.append('a1')

    if(i==2):
        a2 = img
        listar2.append('a2')

    if(i==3):
        a3 = img
        listar2.append('a3')

    if(i==4):
        a4 = img
        listar2.append('a4')


#shape and colour detction
img = cv2.imread('CVtask.png')
#img1 = cv2.resize(img, (1200 , 750))
grayim = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(grayim, 240, 255, cv2.THRESH_BINARY)
contours, xyz = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


def findColor(value):
    if ((value[0]==0) and (value[1]==0) and (value[2]==0)):
        return 3
    if ((value[0]==210) and (value[1]==222) and (value[2]==228)):
        return 4
    if ((value[0]==9) and (value[1]==127) and (value[2]==239)):
        return 2
    if ((value[0]==79) and (value[1]==209) and (value[2]==146)):
        return 1
    else:
        return None

for contour in contours:
    approx = cv2.approxPolyDP(contour , .01*cv2.arcLength(contour ,True) ,True)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    n = approx.ravel()


    if len(approx)== 4:
        x, y ,w ,d = cv2.boundingRect(approx)
        aspectRatio = float(w)/float(d)


        if (aspectRatio >=0.99)and(aspectRatio<=1.20):
            cv2.drawContours(img, [contour], 0 , (0, 255, 0), 3)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            angle = math.degrees(math.atan((approx[2][0][1] - approx[1][0][1]) / (approx[2][0][0] - approx[1][0][0])))
            dx = int(math.sqrt(((approx[0][0][1] - approx[1][0][1]) ** 2) + ((approx[0][0][0] - approx[1][0][0]) ** 2)))
            dy = int(math.sqrt(((approx[2][0][1] - approx[1][0][1]) ** 2) + ((approx[2][0][0] - approx[1][0][0]) ** 2)))

            col = img[int(y+(d/2)), int(x+(w/2))]
            print(col)

            ###checking the color at the center of the countour with the colorall over into the squares
            if findColor(col)==1:
                cv2.drawContours(img, [contour], 0, (0, 0, 0), -1)
                aru = cv2.resize(arucoSet(cv2.imread('[[1]].jpg')), (dx, dy))
                aru = imutils.rotate_bound(aru, angle + 90)
                img[y:y + d - 2, x:x + w - 2] = img[y:y + d - 2, x:x + w - 2] + aru

            if findColor(col)==2:
                cv2.drawContours(img, [contour], 0, (0, 0, 0), -1)
                aru = cv2.resize(arucoSet(cv2.imread('[[2]].jpg')), (dx, dy))
                aru = imutils.rotate_bound(aru, angle + 90)
                img[y:y + d-3, x:x + w-3] = img[y:y + d-3, x:x + w-3] + aru
                
            elif findColor(col)==3:
                cv2.drawContours(img,[contour],0,(0,0,0),-1)
                aru = cv2.resize(arucoSet(cv2.imread('[[3]].jpg')),(dx,dy))
                aru = imutils.rotate_bound(aru,angle+90)
                img[y:y+d-2,x:x+w-2] = img[y:y+d-2,x:x+w-2] + aru

            if findColor(col)==4:
                cv2.drawContours(img, [contour], 0, (0, 0, 0), -1)
                aru = cv2.resize(arucoSet(cv2.imread('[[4]].jpg')), (dx, dy))
                aru = imutils.rotate_bound(aru, angle + 90)
                img[y:y + d - 2, x:x + w - 2] = img[y:y + d - 2, x:x + w - 2] + aru





# print(type(output))
cv2.imshow( 'final'  , img )

if cv2.waitKey(0) & 0xFF == ord('q'):
   cv2.destroyAllWindows()


#dict ={ '1': 'LMAO.jpg'}



