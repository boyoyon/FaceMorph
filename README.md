<html lang="ja">
    <head>
        <meta charset="utf-8" />
    </head>
    <body>
        <h1><center>Face Morph</center></h1>
        <h2>なにものか？</h2>
        <p>
            Mediapipe Facemeshを使ってFaceMorphを書き直したものです。<br>
            <br>
            Face Morph Using OpenCV ? C++ / Python <br>
            <a href="https://learnopencv.com/face-morph-using-opencv-cpp-python">https://learnopencv.com/face-morph-using-opencv-cpp-python</a><br>
            <a href="https://raw.githubusercontent.com/spmallick/learnopencv/master/FaceMorph/faceMorph.py">https://raw.githubusercontent.com/spmallick/learnopencv/master/FaceMorph/faceMorph.py</a><br>
            <img src="images/FaceMorph.png">
        </p>
        <h2>環境構築方法</h2>
        <p>
            pip install mediapipe
        </p>
        <h2>使い方</h2>
        <p>
            python FaceMorph.py (顔画像1) (顔画像2)<br>
            ・Trackbarの移動：ブレンド比を変更する<br>
            ・sキー押下：モーフィング結果の画像を保存する<br>
            ・ESCキー押下：プログラムを終了する<br>
            <img src="images/Usage.svg">
            <br>
        </p>
        <h2>使用例</h2>
        <img src="images/example.svg">
    </body>
</html>
