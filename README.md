# 野生動物辨識系統(僅有山羌)

* 模型以YOLOv8為基礎訓練，並以Flask框架架設應用服務
* 使用以ultralytics為基礎的docker image部署服務
* 上傳影片後會回傳zip檔，裡面包含辨識出的動物及數量的csv檔，以及標示動物的加框影片
