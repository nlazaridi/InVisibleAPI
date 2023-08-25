# InVisible Scene Captioning API

## Installing the Conda Environment

To use this code, you need to install the required dependencies using Conda. Here's how to create a new Conda environment and install the dependencies:

1. Clone this repository to your local machine.
2. Open a terminal or command prompt and navigate to the root directory of the cloned repository.
3. Create a new Conda environment using the following command:
```console
conda env create -f invisivle_env.yml
```
4. Activate the new Conda environment using the following command:
```console
conda activate flask_invisible
```

## Usage

To use this code, first activate the Conda environment as described above. Then, you can run the code using the following command:
```console
python main.py
```
To upload your photos, open http://127.0.0.1:5000 on your browser.

## Finetuning the Model for More Classes

If you want to extend the model to predict more classes, follow these steps:

1. Add the folder containing images of the new class to the `dataset` folder.
2. Add the name of the new class to the `categories_places_365_enhanced.txt` file.
3. Open `train_placesCNN.py` and locate the `num_classes` argument. Change its value to the total number of classes including the new one.
4. Run the following command to start the finetuning process:
```console
python train_placesCNN.py --num_classes=X
```

Replace `X` with the actual number of classes after adding the new one.

Remember to ensure that your new images are labeled correctly and that the dataset is appropriately balanced to achieve the best results.


