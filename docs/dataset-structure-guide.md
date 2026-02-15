# AI Labeller 資料夾結構指南

## 1. 只有影像，沒有標註（新專案）

### A) 扁平資料夾（最快開始）

```text
my_images/
  photo1.jpg
  photo2.jpg
  photo3.png
```

載入後會自動建立：

```text
my_images/
  photo1.jpg
  photo2.jpg
  photo3.png
  labels/
    train/
```

### B) YOLO 單一 split

```text
my_dataset/
  images/
    train/
      photo1.jpg
      photo2.jpg
```

載入後會自動建立：

```text
my_dataset/
  images/
    train/
      photo1.jpg
      photo2.jpg
  labels/
    train/
```

### C) YOLO 多 split（標準）

```text
my_dataset/
  images/
    train/
    val/
    test/
```

載入後會自動建立：

```text
my_dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

## 2. 已有部分標註（繼續標註）

```text
my_dataset/
  images/
    train/
      photo1.jpg
      photo2.jpg
    val/
      photo1.jpg
  labels/
    train/
      photo1.txt
      photo2.txt
    val/
      photo1.txt
```

對應規則：

```text
images/train/photo1.jpg -> labels/train/photo1.txt
images/val/photo1.jpg   -> labels/val/photo1.txt
```

YOLO 標註格式（每行一個物件）：

```text
class_id cx cy w h
```

## 3. 載入方式

1. 啟動程式
2. 選擇來源（影像資料夾 / 載入 YOLO 模型 / 載入 RF-DETR 模型）
3. 選擇資料夾
4. 程式會自動判斷：
- YOLO 結構：`images/{train,val,test}` + `labels/{train,val,test}`
- 扁平結構：`folder/*.jpg|png|jpeg`

若找不到可用影像，程式會顯示「Folder Diagnosis」診斷視窗。

## 4. 常見問題

- Q: 只有 `train` 可以嗎？
- A: 可以，程式支援只有 `train` 的專案。

- Q: `labels/` 一定要先建立嗎？
- A: 不用，程式會在需要時自動建立。

- Q: 支援哪些影像格式？
- A: `.jpg`, `.jpeg`, `.png`

## 5. 建議

- 小型或快速測試：扁平資料夾
- 正式訓練流程：YOLO 多 split（train/val/test）
- 檔名建議使用英文、數字、底線，避免特殊字元
