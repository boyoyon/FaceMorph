# 顔が映った2枚の画像をモーフィングする。

import cv2, os, sys
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

ESC = 27

# 背景もモーフィングする場合はこちら
DEF_TRIANGLES = 'def_triangles2.txt'
NUM_DIVS = 20

# 顔だけをモーフィングする場合はこちら
#DEF_TRIANGLES = 'def_triangles.txt'

# 顔の特徴点で三角形で構成した、頂点インデックスのリストを読み込む
def get_triangles_list():

    triangles_list = []

    with open(os.path.join(os.path.dirname(__file__), DEF_TRIANGLES), mode='r') as f:
        lines = f.read().split('\n')
        for line in lines:
            data = line.split(' ')
            if len(data) == 3:
                triangles_list.append((int(data[0]), int(data[1]), int(data[2])))
    
    return triangles_list

# mediapipeでfacemeshを取得する
def get_facemesh(image):

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        image = image

        height, width = image.shape[:2]

        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        point2d = []

        # face meshが取得されたら
        if results.multi_face_landmarks:

            # 1個目だけを処理する
            face_landmarks = results.multi_face_landmarks[0]

            for landmark in face_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                point2d.append((x, y))

            # 画像の周囲にkeypointを追加する

            for i in range(NUM_DIVS+1):
                x = int(i * (width - 1) / NUM_DIVS)
                point2d.append((x, 0))
                point2d.append((x, height - 1))

            for i in range(1, NUM_DIVS):
                y = int(i * (height - 1) / NUM_DIVS)
                point2d.append((0, y))
                point2d.append((width - 1, y))

        return point2d

# アフィン変換行列を求め、パッチ画像にアフィン変換を適用する
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # 2つの三角形座標を与えてアフィン変換行列を求める
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # パッチ画像にアフィン変換を適用する
    H, W = src.shape[:2]
    if H == 0 or W == 0:
        return None

    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst

# 1つの三角形対に対してモーフィングを実行する
def morphTriangle(img1, img2, img, imgA, imgB, t1, t2, alpha) :

    # 補間した三角形を求める
    t = []
    for i in range(len(t1)):
        x = ( 1 - alpha ) * t1[i][0] + alpha * t2[i][0]
        y = ( 1 - alpha ) * t1[i][1] + alpha * t2[i][1]
        t.append((x,y))

    t1 = np.array(t1)
    t2 = np.array(t2)
    t  = np.array(t)

    # 各三角形の外接矩形を求める

    r1 = cv2.boundingRect(np.float32([t1]))
    left1   = r1[0]
    top1    = r1[1]
    right1  = r1[0] + r1[2]
    bottom1 = r1[1] + r1[3]

    r2 = cv2.boundingRect(np.float32([t2]))
    left2   = r2[0]
    top2    = r2[1]
    right2  = r2[0] + r2[2]
    bottom2 = r2[1] + r2[3]

    r = cv2.boundingRect(np.float32([t]))
    left    = r[0]
    top     = r[1]
    right   = r[0] + r[2]
    bottom  = r[1] + r[3]
    width   = r[2]
    height  = r[3]

    # 外接矩形の左上を原点とした三角形頂点座標を求める
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(3):
        tRect.append(((t[i][0] - left), (t[i][1] - top)))
        t1Rect.append(((t1[i][0] - left1), (t1[i][1] - top1)))
        t2Rect.append(((t2[i][0] - left2), (t2[i][1] - top2)))

    # 三角形を塗りつぶしてマスクを作成する
    mask = np.zeros((height, width, 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # 外接矩形を画像として抽出する
    img1Rect = img1[top1:bottom1, left1:right1]
    img2Rect = img2[top2:bottom2, left2:right2]

    size = (width, height)
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    if warpImage1 is not None and warpImage2 is not None:
   
        if warpImage1.shape[0] == 0 or warpImage1.shape[1] == 0:
            return

        if warpImage2.shape[0] == 0 or warpImage2.shape[1] == 0:
            return

        if top == bottom or left == right:
            return

        if img.shape[0] == 0 or img.shape[1] == 0:
            return

        # 外接矩形パッチにαブレンディングを適用
        imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

        try:
            # 外接矩形パッチにαブレンディングした三角形領域をコピーする
            img[top:bottom, left:right] = img[top:bottom, left:right] * ( 1 - mask ) + imgRect * mask
        except ValueError:
            print('img.shape:', img.shape, top, bottom, left, right)

        try:
            # 外接矩形パッチにαブレンディング前の三角形領域をコピーする
            imgA[top:bottom, left:right] = imgA[top:bottom, left:right] * ( 1 - mask ) + warpImage1 * mask
            imgB[top:bottom, left:right] = imgB[top:bottom, left:right] * ( 1 - mask ) + warpImage2 * mask
        except ValueError:
            print('imgA.shape:', imgA.shape, top, bottom, left, right)
            print('imgB.shape:', imgB.shape, top, bottom, left, right)

# 画像サイズをスケーリングした後でfacemeshを取得すると、取得できない場合があったので
# facemesh取得後に画像サイズをスケーリングし、facemesh座標をシフトすることにした

def scale(img1, facemesh1, img2, facemesh2):

    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    HEIGHT = np.max((height1, height2))
    WIDTH = np.max((width1, width2))

    out1 = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

    startY = (HEIGHT - height1) // 2
    endY = startY + height1
    startX = (WIDTH - width1) // 2
    endX = startX + width1

    out1[startY:endY, startX:endX] = img1

    shifted_mesh1 = []
    for mesh in facemesh1:
        x = mesh[0] + startX
        y = mesh[1] + startY
        shifted_mesh1.append((x, y))

    out2 = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

    startY = (HEIGHT - height2) // 2
    endY = startY + height2
    startX = (WIDTH - width2) // 2
    endX = startX + width2

    out2[startY:endY, startX:endX] = img2

    shifted_mesh2 = []
    for mesh in facemesh2:
        x = mesh[0] + startX
        y = mesh[1] + startY
        shifted_mesh2.append((x, y))

    return out1, shifted_mesh1, out2, shifted_mesh2

def callback(x):
    pass # do nothing

def main():

    argv = sys.argv
    argc = len(argv)

    if argc < 2:
        print('%s morphs between image1 and image2' % argv[0])
        print('%s <image1> <image2>' % argv[0])
        quit()

    image_path1 = argv[1]

    if argc > 2:
        image_path2 = argv[2]
    else:
        image_path2 = image_path1

    dirname1, base1 = os.path.split(image_path1)
    filename1, _ = os.path.splitext(base1)

    dirname2, base2 = os.path.split(image_path2)
    filename2, _ = os.path.splitext(base2)

    triangles_list = get_triangles_list()

    alpha = 0
    prev_alpha = -1

    # Read images
    img1 = cv2.imread(image_path1)

    if argc > 2:
        img2 = cv2.imread(image_path2)
    else:
        img2 = cv2.flip(img1, 1)

    mesh_path1 = os.path.join(dirname1, '%s_mesh2d.npy' % filename1)
    if os.path.isfile(mesh_path1):
        facemesh1 = np.load(mesh_path1)
    else:
        facemesh1 = get_facemesh(img1)

    mesh_path2 = os.path.join(dirname2, '%s_mesh2d.npy' % filename2)
    if os.path.isfile(mesh_path2):
        facemesh2 = np.load(mesh_path2)
    else:
        facemesh2 = get_facemesh(img2)

    img1, facemesh1, img2, facemesh2 = scale(img1, facemesh1, img2, facemesh2)

    # Convert Mat to float data type
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    # Allocate space for final output
    imgMorph = np.zeros(img1.shape, dtype = img1.dtype)
    imgMorphA = np.zeros(img1.shape, dtype = img1.dtype)
    imgMorphB = np.zeros(img1.shape, dtype = img1.dtype)

    # create a window
    cv2.namedWindow("Morphed Face", cv2.WINDOW_NORMAL)
    #cv2.namedWindow("Morphed Face")

    # create trackbar
    cv2.createTrackbar("alpha", "Morphed Face", 0, 100, callback)
    cv2.setTrackbarPos("alpha", "Morphed Face", 50) 

    print('Hit ESC-key to terminate')
    print('Hit s-key to get screenshot')

    while True:

        percent = cv2.getTrackbarPos("alpha", "Morphed Face") 
        alpha = percent / 100

        if prev_alpha != alpha:

            imgMorph = np.zeros(img1.shape, dtype = img1.dtype)
            imgMorphA = np.zeros(img1.shape, dtype = img1.dtype)
            imgMorphB = np.zeros(img1.shape, dtype = img1.dtype)

            for triangle in triangles_list:

                idx0 = triangle[0]
                idx1 = triangle[1]
                idx2 = triangle[2]

                t1 = [facemesh1[idx0], facemesh1[idx1], facemesh1[idx2]]
                t2 = [facemesh2[idx0], facemesh2[idx1], facemesh2[idx2]]

                # Morph one triangle at a time.
                morphTriangle(img1, img2, imgMorph, imgMorphA, imgMorphB, t1, t2, alpha)

            # Display Result
            cv2.imshow("Morphed Face", np.uint8(imgMorph))
            cv2.imshow("Morphed FaceA", np.uint8(imgMorphA))
            cv2.imshow("Morphed FaceB", np.uint8(imgMorphB))
            prev_alpha = alpha

        key = cv2.waitKey(10)

        if key == ord('s') or key == ord('S'):
            cv2.imwrite('Morph_%s_%s_%03d.png' % (filename1, filename2, percent), np.uint8(imgMorph))
            cv2.imwrite('MorphA_%s_%s_%03d.png' % (filename1, filename2, percent), np.uint8(imgMorphA))
            cv2.imwrite('MorphB_%s_%s_%03d.png' % (filename1, filename2, percent), np.uint8(imgMorphB))

        if key == ESC:
            break;

        if cv2.getWindowProperty('Morphed Face', cv2.WND_PROP_VISIBLE) <= 0:
            break

        if cv2.getWindowProperty('Morphed FaceA', cv2.WND_PROP_VISIBLE) <= 0:
            break

        if cv2.getWindowProperty('Morphed FaceB', cv2.WND_PROP_VISIBLE) <= 0:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__' :
    main()
