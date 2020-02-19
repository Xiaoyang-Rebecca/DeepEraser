# DeepEraser
An object-oriented "eraser" for image and video. Is able to remove the pixels with designated type.

Input Video

<img src="https://github.com/Xiaoyang-Rebecca/DeepEraser/blob/master/demos/clip1.gif" width="40%">.

Output Video  (Bordered, Erased)

<img src="https://github.com/Xiaoyang-Rebecca/DeepEraser/blob/master/demos/clip1_borded.gif" width="40%">.
<img src="https://github.com/Xiaoyang-Rebecca/DeepEraser/blob/master/demos/clip1_erased.gif" width="40%">.

# Option1
python deep_eraser.py \
-t 'sports ball' \
--video demo/input/clip1.mp4 \
-o demo/input/clip1_out_sports_ball

# Option2
python deep_eraser.py \
-t 'sports ball' \
--video examples/astros/clip1.mp4 \
-o examples/astros/clip1_out_sports_ball


The pipeline is inspired from  [Mask RCNN](https://github.com/matterport/Mask_RCNN) and [Generative Inpainting Netork](https://github.com/JiahuiYu/generative_inpainting)



