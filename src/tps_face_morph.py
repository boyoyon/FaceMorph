import cv2, os, sys
os.environ['TF_ENABLE_ONEDNN_OPTS']='0' # 警告の抑制
import mediapipe as mp
import numpy as np
import skimage.transform

DECIMATION = 5 # facemeshが多すぎるとうまくいかなかったので間引く

# Mediapipeの準備
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# 顔のランドマークを抽出する関数
def get_landmarks(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    
    landmarks = []
    for i, pt in enumerate(results.multi_face_landmarks[0].landmark):
        if i % DECIMATION == 0: # facemeshを間引く
            landmarks.append((int(pt.x * image.shape[1]), int(pt.y * image.shape[0])))
    return np.array(landmarks)

def main():

    argv = sys.argv
    argc = len(argv)

    print('%s warps image using Thin Plate Spline' % argv[0])
    print('[usage] %s <image1> <image2> <alpha(0-100)>' % argv[0])

    if argc < 3:
        quit()

    ALPHA = 50
    if argc > 3:
        ALPHA = int(argv[3])

    alpha = ALPHA / 100

    # 画像の読み込み
    img1 = cv2.imread(argv[1])
    H1, W1 = img1.shape[:2]
    img2 = cv2.imread(argv[2])
    H2, W2 = img2.shape[:2]
    img1 = cv2.resize(img1, ((W1+W2)//2, (H1+H2)//2))
    img2 = cv2.resize(img2, ((W1+W2)//2, (H1+H2)//2))
    
    # ランドマークを取得
    landmarks1 = get_landmarks(img1)
    landmarks2 = get_landmarks(img2)

    blended_landmarks = landmarks1 * alpha + landmarks2 * (1.0 - alpha)
    blended_landmarks = blended_landmarks.astype(np.int32)

    tps = skimage.transform.ThinPlateSplineTransform()
    
    tps.estimate(blended_landmarks, landmarks1)
    #tps.estimate(landmarks2, landmarks1)
    img1 = img1.astype(np.float32) / 255.0
    warped1 = skimage.transform.warp(img1, tps)

    tps.estimate(blended_landmarks, landmarks2)
    img2 = img2.astype(np.float32) / 255.0
    warped2 = skimage.transform.warp(img2, tps)

    morphed = warped1 * alpha + warped2 * (1.0 - alpha)
    
    # 結果の表示
    cv2.imshow('Image 1', warped1)
    cv2.imshow('Image 2', warped2)
    cv2.imshow('morphed', morphed)
    key = cv2.waitKey(0)
    if key == ord('s') or key == ord('S'):
        warped1 = np.clip(warped1 * 255, 0, 255)
        warped1 = warped1.astype(np.uint8)
        cv2.imwrite('warped1.png', warped1)
        print('save warped1.png')

        warped2 = np.clip(warped2 * 255, 0, 255)
        warped2 = warped2.astype(np.uint8)
        cv2.imwrite('warped2.png', warped2)
        print('save warped2.png')

        morphed = np.clip(morphed * 255, 0, 255)
        morphed = morphed.astype(np.uint8)
        cv2.imwrite('morphed.png', morphed)
        print('save morphed.png')

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
