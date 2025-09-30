# No Concept Left Behind
[![arXiv](https://img.shields.io/badge/arXiv-2509.23457-<COLOR>.svg)](https://arxiv.org/abs/2509.23457)

The source code of [No Concept Left Behind: Test-Time Optimization for Compositional Text-to-Image Generation](https://arxiv.org/abs/2509.23457).
 

<p align="center">
 <img src="https://raw.githubusercontent.com/AmirMansurian/NoConceptLeftBehind/refs/heads/main/figures/Diagram.png"  width="600" height="600"/>
</p>

### Requirements

```bash
conda env create -f environment.yml
conda activate NoConceptLeftBehind
```

### How to Run?
For running the method simply use:
```bash
python main_image_generation_enhancement.py --output_dir outputs/
```
Please specify the '--output_dir' for saving the generated images.

### Experimental Results
Quantitative results on **T2I CompBench**:
| **Method** | **VQA** | **CLIP (L)** | **Captioning (L)** | **GPT4-o** |
|------------|:-------:|:------------:|:------------------:|:----------:|
| FLUX       | 0.865   | 0.272        | 0.687              | 0.717      |
| MILS       | 0.925   | 0.287        | 0.694              | 0.744      |
| **Our**    | **0.955** | **0.295**  | **0.701**          | **0.810**  |

Quantitative results on **DrawBench**:
| **Method** | **VQA** | **CLIP (L)** | **Captioning (L)** | **GPT4-o** |
|------------|:-------:|:------------:|:------------------:|:----------:|
| FLUX       | 0.620   | 0.279        | 0.645              | 0.719      |
| MILS       | 0.665   | 0.299        | 0.671              | 0.765      |
| **Our**    | **0.715** | **0.304**  | **0.677**          | **0.827**  |


Qalitative result samples:
<p align="left">
 <img src="https://raw.githubusercontent.com/AmirMansurian/NoConceptLeftBehind/refs/heads/main/figures/outputs.png"  width="800" height="400"/>
</p>

Per-category comparison results:
Qalitative result samples:
<p align="left">
 <img src="https://raw.githubusercontent.com/AmirMansurian/NoConceptLeftBehind/refs/heads/main/figures/category.png"  width="800" height="400"/>
</p>
 
 ## Citation
If you use this repository for your research or wish to refer to our method, please use the following BibTeX entry:
```bibtex
@article{sameti2025conceptleftbehind,
      title={No Concept Left Behind: Test-Time Optimization for Compositional Text-to-Image Generation}, 
      author={Mohammad Hossein Sameti and Amir M. Mansourian and Arash Marioriyad and Soheil Fadaee Oshyani and Mohammad Hossein Rohban and Mahdieh Soleymani Baghshah},
      journal={arXiv preprint arXiv:2509.23457},
      year={2025}  
}
```

### Acknowledgement
This codebase is heavily borrowed from [LLMs can see and hear without any training](https://github.com/facebookresearch/MILS). Thanks for their excellent work.

