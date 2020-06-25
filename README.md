# Unsupervised Adversarial Depth Estimation using Cycled Generative Networks
The code for "Unsupervised Adversarial Depth Estimation using Cycled Generative Networks" in 3DV2018  
Paper link: https://arxiv.org/pdf/1807.10915.pdf  
By Andrea Pilzer, Dan Xu, Mihai Puscas, Elisa Ricci, Nicu Sebe

<p align="center">
  <img src="framework.jpg" width="800"/>
</p>

# Content

The experiments are performed on a HPC server with Python 3.6 and Tensorflow 1.10.

## 1. Training and testing

Training
```shell
python main.py --dataset kitti --filenames_file utils/filenames/eigen_train_files_png.txt \
--data_path /path/to/KITTI/ --do_stereo --train_branch b2a
```

Testing
```shell
python main.py --mode test --dataset kitti --filenames_file utils/filenames/eigen_test_files_png.txt \
--data_path /path/to/KITTI/ --do_stereo --checkpoint_path my_model/model-5000
```
**Please note that there is NO extension after the checkpoint name**

Evaluation
```shell
python utils/evaluate_kitti.py --split kitti --predicted_disp_path my_model/disparities.npy \
--gt_path ~/data/KITTI/
```

## 2. Datasets

We used the KITTI dataset in our experiments. Please refer to a very well written dataset description section of [Monodepth](https://github.com/mrharicot/monodepth/blob/master/readme.md) for data preparation.

## 3. Trained model

The pretrained model can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1dWffc6XSyvxRO_89_jicT-cqJjHbd2-c?usp=sharing).

## 4. Citation
Please condiser citing our paper if you find the code is useful for your projects:
<pre>
@inproceedings{pilzer2018unsupervised,
  title={Unsupervised Adversarial Depth Estimation using Cycled Generative Networks},
  author={Pilzer, Andrea and Xu, Dan and Puscas, Mihai and Ricci, Elisa and Sebe, Nicu},
  booktitle={2018 International Conference on 3D Vision (3DV)},
  pages={587--595},
  year={2018},
  organization={IEEE}
}

@article{pilzer2019progressive,
  title={Progressive Fusion for Unsupervised Binocular Depth Estimation using Cycled Networks},
  author={Pilzer, Andrea and Lathuili{\`e}re, St{\'e}phane and Xu, Dan and Puscas, Mihai Marian and Ricci, Elisa and Sebe, Nicu},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year={2019},
  publisher={IEEE}
}

</pre>


