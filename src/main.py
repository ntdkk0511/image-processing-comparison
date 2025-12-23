import cv2
from pathlib import Path

# プロジェクト直下を基準に画像パスを作る
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = BASE_DIR / "images" / "sample.png"

img = cv2.imread(str(IMAGE_PATH))

if img is None:
    print("画像が読み込めませんでした")
else:
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
