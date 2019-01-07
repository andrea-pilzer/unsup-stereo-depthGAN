# unsup-stereo-depthGAN
The code for "Unsupervised Adversarial Depth Estimation using Cycled Generative Networks" in 3DV2018  
Paper link: https://arxiv.org/pdf/1807.10915.pdf  
By Andrea Pilzer, Dan Xu, Mihai Puscas, Elisa Ricci, Nicu Sebe

# Content

This code was developed with Tensorflow 1.5 and Python2, we run experiments on a HPC server with Python 3.6 and Tensorflow 1.10.

1. Training and testing

Training
'''
python main.py --dataset kitti --filenames_file utils/filenames/eigen_train_files_png.txt \
--data_path /path/to/KITTI/ --do_stereo --train_branch b2a
'''

Testing
'''
python main.py --mode test --dataset kitti --filenames_file utils/filenames/eigen_test_files_png.txt \
--data_path /path/to/KITTI/ --do_stereo --checkpoint_path my_model/model-5000
'''

Evaluation
'''
python utils/evaluate_kitti.py --split kitti --predicted_disp_path my_model/disparities.npy \
--gt_path ~/data/KITTI/
'''

2. Datasets

Please refer to the very well written dataset section of [Monodepth](https://github.com/mrharicot/monodepth/blob/master/readme.md)

3. Trained model

[Google Drive](https://drive.google.com/drive/folders/1dWffc6XSyvxRO_89_jicT-cqJjHbd2-c?usp=sharing)

4. Citation

'''
@inproceedings{pilzer2018unsupervised,
  title={Unsupervised Adversarial Depth Estimation using Cycled Generative Networks},
  author={Pilzer, Andrea and Xu, Dan and Puscas, Mihai and Ricci, Elisa and Sebe, Nicu},
  booktitle={2018 International Conference on 3D Vision (3DV)},
  pages={587--595},
  year={2018},
  organization={IEEE}
}
'''


