import cv2
import numpy as np
import math
from tensorflow.keras.applications import ResNet50
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import preprocess_input

#2つの画像ファイルの類似性を調べるツール(白背景推奨)
def move_object_to_center(image):
    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # バイナリーマスクを作成（白背景を無視）
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # 重心を計算
    moments = cv2.moments(binary)
    center_x = int(moments['m10'] / moments['m00'])
    center_y = int(moments['m01'] / moments['m00'])

    # 画像の中心を計算
    height, width = image.shape[:2]
    image_center_x = width // 2
    image_center_y = height // 2

    # 移動量を計算
    dx = image_center_x - center_x
    dy = image_center_y - center_y

    # 移動行列を作成
    M = np.float32([[1, 0, dx], [0, 1, dy]])

    # 画像を移動
    translated_image = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    #cv2.imshow('center_img', translated_image)
    #cv2.waitKey(0)
    return translated_image


# 少しずつ回転の場合
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),  borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated


def find_best_homography(img1, img2):
    # 画像をグレースケールに変換
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SIFT検出器を作成
    sift = cv2.SIFT_create()

    # SIFTを使用して特徴点と記述子を検出
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    # デフォルトのパラメータでBFMatcherを使用
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # 距離でマッチをソート
    matches = sorted(matches, key=lambda x: x.distance)

    num_good_matches = 100  # 使用する良好なマッチの数
    good_matches = matches[:num_good_matches]

    # 良好なマッチの位置を抽出
    src_pts = np.zeros((len(good_matches), 2), dtype=np.float32)
    dst_pts = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        src_pts[i, :] = kp1[match.queryIdx].pt
        dst_pts[i, :] = kp2[match.trainIdx].pt

    # ホモグラフィを見つける
    if len(src_pts) >= 4 and len(dst_pts) >= 4:  # ホモグラフィを計算するための十分な点があるか確認
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)
        return H
    else:
        return None
    

#裏返し不可版
def align_and_overlay_images(img1, img2):

    # 回転などのバリエーション
    variations = [
        img2,
        cv2.flip(img2, 1),  # 水平反転
        cv2.flip(img2, 0),  # 垂直反転
        cv2.flip(img2, -1),  # 水平垂直反転
        cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE),  # 時計回りに90度回転
        cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE),  # 反時計回りに90度回転
        cv2.rotate(img2, cv2.ROTATE_180)  # 180度回転
    ]

    best_aligned_img = None
    min_difference = float('inf')

    # ホモグラフィを使用してimg2をimg1に合わせる
    for angle in range(0, 360, 15): 
        for var_img in variations:
        
            rotated_img2 = rotate_image(var_img, angle)
            H = find_best_homography(img1, rotated_img2)
            if H is not None:
                #H = H.astype(np.float32)  # データ型をfloat32に変換
                height, width, channels = img1.shape
                aligned_img = cv2.warpPerspective(rotated_img2, H, (width, height),borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

                difference = cv2.norm(img1, aligned_img)
                if difference < min_difference:
                    min_difference = difference
                    best_aligned_img = aligned_img

    #cv2.imshow('aligned_img', aligned_img)
    #cv2.waitKey(0)

    if best_aligned_img is not None:
        overlay_img = img1.copy()
        alpha = 0.5
        cv2.addWeighted(best_aligned_img, alpha, overlay_img, 1 - alpha, 0, overlay_img)
        return best_aligned_img
    else:
        raise ValueError("画像をアライメントできませんでした")

# マスクを作成（白い部分を無視するため）
def create_mask(image, color=(255, 255, 255)):
    # 指定された色をマスク
    mask = cv2.inRange(image, color, color)
    return mask

# 類似性の計算
def compute_similarity(img1, img2, mask1=None, mask2=None):
    # ResNet50モデルをロード
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    def extract_features(image, mask=None):
        # 画像をリサイズ
        image = cv2.resize(image, (224, 224))
        if mask is not None:
            mask = cv2.resize(mask, (224, 224))
            image[mask == 255] = 0
        # 画像を前処理
        image = preprocess_input(np.expand_dims(image, axis=0))
        # 特徴量を抽出
        features = model.predict(image)
        return features

    features1 = extract_features(img1, mask1)
    features2 = extract_features(img2, mask2)

    # コサイン類似度を計算
    similarity = cosine_similarity(features1, features2)
    return similarity[0][0]

# 元画像のグレースケース化
def highlight_differences(img1, img2, color=(0, 0, 255)):
    # img2をimg1にアライメント
    difference = cv2.absdiff(img1, img2)

    # 画像の差分を計算
    gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img1_colored = cv2.cvtColor(gray_img1, cv2.COLOR_GRAY2BGR)

    gray_img1_colored[mask != 0] = color

    return gray_img1_colored

#####

# 画像を読み込む
print("比較する画像ファイルのパスを入力してください。")
img1_path = input("ファイル1：")
img2_path = input("ファイル2：")

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

if img1 is None:
    raise FileNotFoundError(f"画像1を読み込めませんでした: {img1_path}")
if img2 is None:
    raise FileNotFoundError(f"画像2を読み込めませんでした: {img2_path}")

#位置を真ん中にする
img1_c = move_object_to_center(img1)
img2_c = move_object_to_center(img2)

# マスクを作成（白い部分を無視するため）
mask = create_mask(img1_c, color=(255, 255, 255))

# 画像のアライメント
aligned_img = align_and_overlay_images(img1_c, img2_c)

# 画像のリサイズ
original_height, original_width = img1_c.shape[:2]
re_img1 = cv2.resize(aligned_img, (0, 0), fx=0.5, fy=0.5)

# 1枚目と同じ位置や向きに調整した2枚目の画像ファイルを表示
cv2.imshow('Highlighted Differences', re_img1)
cv2.waitKey(0)

# マスクを使用して空白部分を削除
img1_masked = cv2.bitwise_and(img1_c, img1_c, mask=cv2.bitwise_not(mask))
aligned_img_masked = cv2.bitwise_and(aligned_img, aligned_img, mask=cv2.bitwise_not(mask))

#指定した画像からマスク部分を抜いたクロマキー画像(比較する画像)を表示
#cv2.imshow('Highlighted Differences', img1_masked)
#cv2.waitKey(0)
#cv2.imshow('Highlighted Differences', aligned_img_masked)
#cv2.waitKey(0)

# 類似性の計算
similarity = compute_similarity(img1_masked, aligned_img_masked)

par = "{:.1%}".format(similarity)
pars = "{:.5%}".format(similarity)
#print(similarity)
#print(par)

# 判定の閾値を設定
threshold = 1

# 結果
if similarity >= threshold:
    print("結果：全く同じです。(類似率 " + pars +")")
elif similarity > 0.99:
     print("結果：ほぼ同じ画像です。(類似率 " + par +")")
     if (par == "100.0%") & (similarity < 1):
         print("(詳細：" + pars +")")
else:
    print("結果：類似率 " + par + "です。")

# 違いを強調
highlighted_img = highlight_differences(img1_c, aligned_img)

# 画像のリサイズ
original_height, original_width = img1_c.shape[:2]
resized_img = cv2.resize(highlighted_img, (0, 0), fx=0.5, fy=0.5)

# 結果を表示
cv2.imshow('Highlighted Differences', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()