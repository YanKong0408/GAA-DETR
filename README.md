# GAA-DETR
**MICCAI-2025 Early Accept**: Query-Level Alignment for End-to-end Lesion Detection with Human Gaze

![intro](image\fig_intro.jpg)

Our work introduces a novel detection framework, **GAA-DETR (Gaze-Aligned Attention Detection Transformer)**, which integrates clinical gaze data to enhance lesion detection accuracy. Inspired by how clinicians search for lesions during diagnosis, our method aligns model attention with gaze patterns, enabling detection models to "see" like doctors. 

We also contribute the first large-scale **Medical Lesion Detection Gaze Dataset**, which includes 1,669 open-sourced gaze annotations.

---

## Dataset
![dataset](image\fig_dataset.jpg)
### Gaze Data
Our gaze data were collected using [MicEye-v2.0](https://github.com/YanKong0408/MicEye-v2.0), a tool that records radiologists' eye movements during bounding box annotations. The processed dataset, including gaze heatmaps, is available for download:

**[Download Gaze Data](https://pan.baidu.com/s/1Rz3KSEU_uKzbMi7VJUeHHg?pwd=gaze)** (Password: `gaze`)

### Image Data
The image data used in our work is sourced from the following public datasets:

- **Breast Dataset**: Mammograms with annotations for malignant and benign lesions. The datasets used are:
  
  **INbreast**: A full-field digital mammographic database. It can be obtained by sending an email request to `medicalresearch@inescporto.pt`. Please refer to the following publication for more details:
    ```bibtex
    @article{moreira2012inbreast,
      title={INbreast: toward a full-field digital mammographic database},
      author={Moreira, Igor C and Amaral, Inês and Domingues, Inês and Cardoso, Ana and Cardoso, Maria J and Cardoso, Jaime S},
      journal={Academic radiology},
      volume={19},
      number={2},
      pages={236--248},
      year={2012}
    }
    ```
  **CBIS-DDSM**: A curated breast imaging subset of the Digital Database for Screening Mammography. This dataset is available for download from [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/cbis-ddsm/). You can use the `code/down_load_DDSM.py` script to automate the download process. For more information, please refer to:
    ```bibtex
    @article{lee2017curated,
      title={A curated mammography data set for use in computer-aided detection and diagnosis research},
      author={Lee, Ryan S and Gimenez, Fernando and Hoogi, Assaf and Miyake, Kristin K and Gorovoy, Marc and Rubin, Daniel L},
      journal={Scientific data},
      volume={4},
      number={1},
      pages={1--9},
      year={2017}
    }
    ```

- **ComparisonDetector Dataset**: Cervical images annotated for lesion detection. This dataset can be downloaded from [GitHub - ComparisonDetector](https://github.com/kuku-sichuan/ComparisonDetector). For more details, please refer to:
  ```bibtex
  @article{liang2021comparison,
    title={Comparison detector for cervical cell/clumps detection in the limited data scenario},
    author={Liang, Y and Tang, Z and Yan, M and Chen, J and Liu, Q and Xiang, Y},
    journal={Neurocomputing},
    volume={437},
    pages={195--205},
    year={2021}
  }
  ```
Please ensure to **cite the original datasets** when using our dataset in your research.

---

## Code

### Usage

#### Install

```sh
# Clone this repo
git clone https://github.com/YanKong0408/GAA-DETR.git
cd code/GAA-DETR

# Install Pytorch and torchvision
# Tested with 'python=3.7.3,pytorch=1.9.0,cuda=11.1'. Other versions might work.
conda install -c pytorch pytorch torchvision

# Install other required packages
pip install -r requirements.txt
```

### Model
Our code is based on [DETR](https://github.com/facebookresearch/detr).

Data can be obtained from the json file split resulting from the above step and should be organized into the coco dataset format.Gaze heatmaps should be placed in the same folder as the medical images and named as XXX_heatmap.jpg/png.
```
Your_Fold/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json  

Image_Fold/ 
  └── XXX1.jpg
  └── XXX1_heatmap.jpg
  └── XXX2.jpg
  └── XXX2_heatmap.jpg
```

Train
``` sh
python main.py 
    --output_dir  /path/to/your/output/dir 
    --coco_path /path/to/your/data/dir
    --resume path/to/your/pre-train/model.pth
```

Inference
``` sh
python mian.py \
    --output_dir  /path/to/your/output/dir \
    -c config/Gaze-DINO/Gaze_DINO_swin.py \
    --options batch_size=1 \
    --coco_path /path/to/your/data/dir
    --resume path/to/your/pre-train/model.pth
    --eval
```

<!-- More experiment results are shown in [our paper](https://arxiv.org/pdf/2405.09463). -->

<!-- More test will come soon.

## Citation
Use this bibtex to cite this repository:
```
@misc{kong2024gazedetr,
      title={Gaze-DETR: Using Expert Gaze to Reduce False Positives in Vulvovaginal Candidiasis Screening}, 
      author={Yan Kong and Sheng Wang and Jiangdong Cai and Zihao Zhao and Zhenrong Shen and Yonghao Li and Manman Fei and Qian Wang},
      year={2024},
      eprint={2405.09463},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
``` -->
