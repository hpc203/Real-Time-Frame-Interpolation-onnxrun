本套部署程序对应的paper是旷世研究院在ECCV2022发表的一篇文章《Real-Time Intermediate Flow Estimation for Video Frame Interpolation》，
模型简称rife，onnx模型文件只有8.79M，模型已经很小了，
训练源码在 https://github.com/megvii-research/ECCV2022-RIFE，有3.9k个star。
而谷歌在ECCV2022发布的做视频帧插值的文章《FILM: Frame Interpolation for Large Motion》，
模型简称film，训练源码在https://github.com/google-research/frame-interpolation，有2.6k个star
它的onnx模型文件却有133M，相比于旷世发布的rife，大很多了。因此旷世发布的rife更适合在工业界落地应用
