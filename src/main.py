import cv2
import os

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

def detect_edge(gray):
    """
    グレースケール画像からエッジ検出
    """
    edge = cv2.Canny(gray,100,200)
    return edge



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

    #エッジ
    edge = detect_edge(gray)
    show_image("Edge",edge)

#このファイルを直接実行したときだけmain()を動かす文→大規模化に繋がる
if __name__ == "__main__":
    main()
