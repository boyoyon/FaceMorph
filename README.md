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
            <img src="images/FaceMorph.gif">
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
        <h2>その他</h2>
        <h3>Thin Plate Spline</h3>
        <p>
            <a href="https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model">Thin-Plate-Spline-Motion-Model</a><br>
            が、かなりリアルだったので、Thin Plate Splineを使ってFaceMorphを作ってみた。<br>
            <img src="images/tps_face_morph.png"><br>
            ・scikit-imageが0.24より古い場合はupgradeする。<br>
            　pip install --upgrade scikit-image<br>
            ・python src\tps_face_morph.py (画像ファイル1) (画像ファイル2)<br>
            　sキー押下でモーフィングした画像を保存する。<br>
            ・facemeshが多いとうまく変換してくれなかったので、機械的に間引いている。<br>
            　大事なキーポイントを選んで間引くことで多少改善するかもしれない(未確認)。<br>
            <br>
            mediapipe facemesh にパラメータ refine_landmarks=True を指定すると虹彩の中心、上下左右のlandmarkが追加される。<br>
            その他のlandmarkの位置も若干変わる。<br>
            <img src="images/landmarks.png"><br>
            refine_landmarks=Trueを指定＋虹彩の中心、上下左右が間引かれな対応を実施した。<br>
            ・虹彩のぶれ<br>
            ・あご輪郭のぶれ<br>
            が少しだけ改善する<br>
            python src/tps_face_morph_refine_landmarks.py (画像1) (画像2)<br>
            <img src="images/before_after_refine_landmarks.png">
        </p>
    </body>
</html>
