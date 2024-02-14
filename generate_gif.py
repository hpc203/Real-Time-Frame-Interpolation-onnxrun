import imageio
import numpy as np
import itertools

if __name__=='__main__':
    inputs_imgpath = ["testimgs/frame07.png", "testimgs/frame08.png", "testimgs/frame09.png", "testimgs/frame10.png", "testimgs/frame11.png", "testimgs/frame12.png", "testimgs/frame13.png", "testimgs/frame14.png"]
    mids_imgpath = ["imgs_results/mid0.jpg", "imgs_results/mid1.jpg", "imgs_results/mid2.jpg", "imgs_results/mid3.jpg", "imgs_results/mid4.jpg", "imgs_results/mid5.jpg", "imgs_results/mid6.jpg"]
    
    img_paths = list(itertools.chain.from_iterable(zip(inputs_imgpath, mids_imgpath))) ###交错合并列表
    img_paths.append(inputs_imgpath[-1])
    # print(img_paths)

    gif_images = []
    for path in inputs_imgpath:
        gif_images.append(imageio.imread(path))
    imageio.mimsave("source.gif",gif_images,fps=1)

    gif_images = []
    for path in img_paths:
        gif_images.append(imageio.imread(path))
    imageio.mimsave("result.gif",gif_images,fps=1)


    gif_1 = imageio.get_reader("source.gif")
    gif_2 = imageio.get_reader("result.gif")

    combined_gif = imageio.get_writer('combined.gif')
    ###gif_1有8帧, gif_2有15帧，合并后的动态图只有前8帧了
    for frame1, frame2 in zip(gif_1, gif_2):
        combined_gif.append_data(np.hstack((frame1, frame2))) ###水平方向平行显示
    
    gif_1.close()
    gif_2.close()    
    combined_gif.close()