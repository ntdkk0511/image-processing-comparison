import cv2
import os
import time

def show_image(title, img, max_width=800, max_height=600):
    """
    画像をウィンドウサイズに収まるようにリサイズして表示する関数
    """
    #img.shape→（高さ，幅，色）高さと幅のみ使うため，[:2]
    h, w = img.shape[:2]

    #縮小するために掛ける0.●●といった数値を決める
    scale_w = max_width / w
    scale_h = max_height / h
    #最小値をとる（1.0は拡大防止）
    scale = min(scale_w, scale_h, 1.0)  

    #縮小率をかけて整数に変換
    new_w = int(w * scale)
    new_h = int(h * scale)

    #OpenCVの引数は（幅，高さ）
    resized_img = cv2.resize(img, (new_w, new_h))

    cv2.imshow(title, resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def to_grayscale(img):
    """
    BGR画像をグレースケール画像に変換する関数
    """
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return gray

def detect_edge_canny(gray):
    """
    グレースケール画像からエッジ検出（Canny）
    """
    
    canny = cv2.Canny(gray,100,200)
    return canny

def detect_edge_sobel(gray):
    """
    グレースケール画像からエッジ検出（Sobel）
    """
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(sobel_x, sobel_y)
    sobel = cv2.convertScaleAbs(magnitude)
    return sobel


def main():
    image_path = os.path.join("images", "sample.png")

    img = cv2.imread(image_path)

    if img is None:
        print("画像が読み込めませんでした")
        return
    
    #原画像
    show_image("Original Image", img)

    #グレースケール
    gray = to_grayscale(img)
    show_image("Grayscale Image",gray)

    #Cannyエッジ検出（時間計測）
    start = time.time() #現在の時間をfloat型の秒単位で取得
    canny = detect_edge_canny(gray)
    canny_time = time.time() - start #現在時間からstart時の差を計測

    #Sobelエッジ検出（時間計測）
    start = time.time()
    sobel = detect_edge_sobel(gray)
    sobel_time = time.time() - start
    
    #結果表示
    print(f"Sobel time: {sobel_time:.6f} sec")
    print(f"Canny time: {canny_time:.6f} sec")
    
    #show_imageを計測に含まない！ユーザのキー入力待ち等を含むとアルゴリズムの処理時間ではない
    show_image("Canny Edge",canny)
    show_image("Sobel Edge",sobel)



#このファイルを直接実行したときだけmain()を動かす文→大規模化に繋がる
if __name__ == "__main__":
    main()
