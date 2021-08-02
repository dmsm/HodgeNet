## HodgeNet | [Webpage](https://people.csail.mit.edu/smirnov/hodgenet/) | [Paper](https://dl.acm.org/doi/abs/10.1145/3450626.3459797) | [Video](https://youtu.be/juP0PHxvnx8)

<img src="https://people.csail.mit.edu/smirnov/hodgenet/im.png" width="75%" alt="HodgeNet" />

**HodgeNet: Learning Spectral Geometry on Triangle Meshes**<br>
Dmitriy Smirnov, Justin Solomon<br>
[SIGGRAPH 2021](https://s2021.siggraph.org/)

### Set-up
To install the neecssary dependencies, run:
```
conda env create -f environment.yml
conda activate hodgenet
```

### Training
To train the segmentation model, first download the [Shape COSEG dataset](http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/ssd.htm). Then, run:
```
python train_segmentation.py --out out_dir --mesh_path path_to_meshes --seg_path path_to_segs
```

To train the classification model, first download the SHREC 2011 dataset:
```
wget -O shrec.tar.gz https://www.dropbox.com/s/4z4v1x30jsy0uoh/shrec.tar.gz?dl=0
tar -xvf shrec.tar.gz -C data
```
Then, run:
```
python train_classification.py --out out_dir
```

To train the dihedral angle stress test model, run:
```
python train_origami.py --out out_dir
```

To monitor the training, launch a TensorBoard instance with `--logdir out_dir`

To finetune a model, add the flag `--fine_tune` to the above training commands.

### BibTeX
```
@article{smirnov2021hodgenet,
  title={{HodgeNet}: Learning Spectral Geometry on Triangle Meshes},
  author={Smirnov, Dmitriy and Solomon, Justin},
  year={2021},
  journal={SIGGRAPH}
}
```
