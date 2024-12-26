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
                        <img src="images/Usage.svg">
            <table border="1">
                <tr><th>操作</th><th>機能</th></tr>
                <tr><td>Trackbarの移動</td><td>ブレンド比を変更してモーフィングする</td></tr>
                <tr><td>sキー押下</td><td>モーフィング結果の画像を保存する</td></tr>
                <tr><td>ESCキー押下</td><td>プログラムを終了する</td></tr>
            </table>
            python FaceMorph.py (顔画像1)<br>
            画像を1枚だけ指定した場合は、左右反転した画像とモーフィングします。<br>
            <img src="images/FaceMorph2.png"><br>
        </p>
        <h2>使用例</h2>
        <img src="images/example.svg">
    </body>
</html>
