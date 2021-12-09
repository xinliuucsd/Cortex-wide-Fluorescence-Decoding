# Cortex-wide Fluorescence Decoding
This repository hosts the codes for decoding of cortex-wide brain activity from local recordings of neural potentials.
![Model Schematic](https://user-images.githubusercontent.com/45840789/145486921-7911acbb-feb2-4566-a96e-a5f6f9a508b5.jpg)

## Publication
Liu, X., Ren, C., Huang, Z., Wilson, M., Kim, J.H., Lu, Y., Ramezani, M., Komiyama, T. and Kuzum, D., 2021. Decoding of cortex-wide brain activity from local recordings of neural potentials. _Journal of Neural Engineering_, _18_(6), p.066009.
- Link to the published paper: [Here](https://iopscience.iop.org/article/10.1088/1741-2552/ac33e7/meta)
- Link to the bioRxiv preprint: [Here](https://www.biorxiv.org/content/10.1101/2021.10.14.464468v1)
## Dependencies
Running the code requires below dependencies:
```
Pytorch v1.1
NumPy
SciPy
h5py
```

## Usage Guide
1. Download the dataset and store it in the "Data" folder.
2. Run "ECoG_Ca_decoding_example.py" to train the model for decoding tasks.
3. The decoding results and the models will be saved in the "saved_results" folder.
## The Wide-field Imaging and LFP Data
Link to the preprocessed Ca and LFP data: [Click here](https://drive.google.com/drive/folders/1OawL10NtduQSsP6Z8OMFPFtrz5fmwyOe?usp=sharing)
