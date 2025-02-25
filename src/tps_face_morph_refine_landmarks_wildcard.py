import cv2, glob, itertools, os, sys
os.environ['TF_ENABLE_ONEDNN_OPTS']='0' # 警告の抑制
import mediapipe as mp
import numpy as np
import skimage.transform

alpha = 0.5

OUTPUT_DIR = 'morphed'

DECIMATION = 5 # facemeshが多すぎるとうまくいかなかったので間引く

# refine_landmarks =True の設定で増える虹彩の中心、上下左右の landmark の index
indices = [468, 469, 470, 471, 472, 473, 474, 475, 476]

# Mediapipeの準備
mp_face_mesh = mp.solutions.face_mesh
# refine_landmarks 指定あり
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# 顔のランドマークを抽出する関数
def get_landmarks(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    
    landmarks = []
    for i, pt in enumerate(results.multi_face_landmarks[0].landmark):
        if i % DECIMATION == 0 or i in indices:
            landmarks.append((pt.x, pt.y))
    return np.array(landmarks)

def main():

    argv = sys.argv
    argc = len(argv)

    print('%s warps image using Thin Plate Spline' % argv[0])
    print('[usage] %s <wildcard for images>' % argv[0])

    if argc < 2:
        quit()

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    paths = glob.glob(argv[1])
    nrData = len(paths)

    facemeshs = []
    for i, path in enumerate(paths):
        print('pre-processing %d/%d' % (i+1, nrData))
        img = cv2.imread(path)
        facemeshs.append(get_landmarks(img))

    pairs = itertools.combinations(np.arange(nrData), 2)

    for pair in pairs:

        idxA = pair[0]
        idxB = pair[1]

        print('morphing %s and %s' % (paths[idxA], paths[idxB]))

        # 画像の読み込み
        imgA = cv2.imread(paths[idxA])
        HA, WA = imgA.shape[:2]
        imgB = cv2.imread(paths[idxB])
        HB, WB = imgB.shape[:2]

        H = (HA + HB) // 2
        W = (WA + WB) // 2

        imgA = cv2.resize(imgA, (W, H))
        imgB = cv2.resize(imgB, (W, H))
        
        # ランドマークを取得
        landmarksA = facemeshs[idxA].copy()
        landmarksA[:,0] *= W
        landmarksA[:,1] *= H
        landmarksA = landmarksA.astype(np.int32)

        landmarksB = facemeshs[idxB].copy()
        landmarksB[:,0] *= W
        landmarksB[:,1] *= H
        landmarksB = landmarksB.astype(np.int32)
    
        blended_landmarks = landmarksA * alpha + landmarksB * (1.0 - alpha)
        blended_landmarks = blended_landmarks.astype(np.int32)
    
        tps = skimage.transform.ThinPlateSplineTransform()
        
        tps.estimate(blended_landmarks, landmarksA)
        imgA = imgA.astype(np.float32) / 255.0
        warpedA = skimage.transform.warp(imgA, tps)
    
        tps.estimate(blended_landmarks, landmarksB)
        imgB = imgB.astype(np.float32) / 255.0
        warpedB = skimage.transform.warp(imgB, tps)
    
        morphed = warpedA * alpha + warpedB * (1.0 - alpha)
        morphed = np.clip(morphed * 255, 0, 255)
        morphed = morphed.astype(np.uint8)

        baseA = os.path.basename(paths[idxA])
        filenameA = os.path.splitext(baseA)[0]
        baseB = os.path.basename(paths[idxB])
        filenameB = os.path.splitext(baseB)[0]

        dst_path = os.path.join(OUTPUT_DIR, '%s_%s.png' % (filenameA, filenameB))
        cv2.imwrite(dst_path, morphed)
        print('save %s' % dst_path)

if __name__ == '__main__':
    main()
