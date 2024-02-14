import cv2
import numpy as np
import onnxruntime
import os

class RIFE:
    def __init__(self, modelpath):
        # Initialize model
        self.onnx_session = onnxruntime.InferenceSession(modelpath)
        self.input1_name = self.onnx_session.get_inputs()[0].name
        self.input2_name = self.onnx_session.get_inputs()[1].name
        ####动态输入高宽

    def prepare_input(self, image):
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = input_image.shape

        align = 32
        if h % align != 0 or w % align != 0:
            ph = ((h - 1) // align + 1) * align
            pw = ((w - 1) // align + 1) * align

            pad_img = np.zeros(shape=(ph, pw, 3))
            pad_img[:h, :w, :] = input_image

            input_image = pad_img

        input_image = input_image.astype(np.float32) / 255.0
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        return input_image

    def interpolate(self, image1, image2):
        h, w, _ = image1.shape
        input1_image = self.prepare_input(image1)
        input2_image = self.prepare_input(image2)

        # Perform inference on the image
        result = self.onnx_session.run(None, {self.input1_name: input1_image, self.input2_name: input2_image})

        # Post process:squeeze, RGB->BGR, Transpose, uint8 cast
        mid_img = np.squeeze(result[0])
        mid_img = mid_img.transpose(1, 2, 0)
        mid_img = np.clip(mid_img * 255, 0, 255)
        mid_img = (mid_img + 0.5).astype(np.uint8)
        mid_img = mid_img[..., ::-1]
        mid_img = mid_img[:h, :w, ...]
        return mid_img
    
    def interpolate_4x(self, image1, image2):
        mid_img = self.interpolate(image1, image2)

        first_half = self.interpolate(image1, mid_img)
        second_half = self.interpolate(mid_img, image2)
        return [first_half, mid_img, second_half]

if __name__ == '__main__':
    use_video = 0
    mynet = RIFE('RIFE_HDv3.onnx')

    if use_video==0:
        inputs_imgpath = ["testimgs/frame07.png", "testimgs/frame08.png", "testimgs/frame09.png", "testimgs/frame10.png", "testimgs/frame11.png", "testimgs/frame12.png", "testimgs/frame13.png", "testimgs/frame14.png"]
        # dirpath = "imgs"  ###也可以输入图片文件夹
        # inputs_imgpath = os.listdir(dirpath)
        imgnum = len(inputs_imgpath)
        if imgnum<2:
            exit("input must be at least two or more images")
        

        for i in range(0, imgnum-1):
            srcimg1 = cv2.imread(inputs_imgpath[i])
            srcimg2 = cv2.imread(inputs_imgpath[i+1])
            mid_img = mynet.interpolate(srcimg1, srcimg2)

            cv2.imwrite(os.path.join("imgs_results", 'mid'+str(i)+'.jpg'), mid_img)

            # cv2.namedWindow('srcimg1', 0)
            # cv2.imshow('srcimg1', srcimg1)
            # cv2.namedWindow('mid_img', 0)
            # cv2.imshow('mid_img', mid_img)
            # cv2.namedWindow('srcimg2', 0)
            # cv2.imshow('srcimg2', srcimg2)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    else:
        videopath = "test.mp4"
        cap = cv2.VideoCapture(videopath)
        cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
        video_writer = cv2.VideoWriter(
            filename='output.mp4',
            fourcc=fourcc,
            fps=cap_fps,
            frameSize=(cap_width, cap_height),
        )

        # 创建实时显示的帧图像
        # n_output = 1
        # output_buffer = np.zeros((cap_height * (n_output + 2), cap_width, 3))
        # output_buffer = output_buffer.astype(np.uint8)

        images = []
        while True:
            # Capture read
            ret, frame = cap.read()
            if not ret:
                break

            images.append(frame)
            if len(images) < 2:
                continue
            elif len(images) > 2:
                images = images[1:]

            # inference
            img1, img2 = images
            mid_img = mynet.interpolate(img1, img2)

            # output_buffer[:cap_height, :cap_width, :] = images[0]
            # output_buffer[cap_height * 1:cap_height * 2, :cap_width, :] = mid_img
            # output_buffer[cap_height * 2:cap_height * 3, :cap_width, :] = images[1]
            # cv2.imshow('Deep learning frame_interpolate use onnxruntime', output_buffer)
            # key = cv2.waitKey(1)
            # if key == 27:  # ESC
            #     break

            # save results
            if video_writer is not None:
                video_writer.write(images[0])
                video_writer.write(mid_img)

        if video_writer:
            video_writer.release()
        if cap:
            cap.release()
        cv2.destroyAllWindows()