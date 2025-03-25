
## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/Live-traffic-classify.git
    ```
2. Navigate to the project directory:
    ```sh
    cd traffic-sign-recognition
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. Prepare the dataset:
    - Download the GTSRB dataset from [here](https://benchmark.ini.rub.de/index.html).
    - Place the dataset in the `GTSRB/` directory.

2. Run data preparation scripts:
    ```sh
    python data_preparation/getData.py
    python data_preparation/getFile.py
    ```

3. Perform image enhancement:
    ```sh
    python data_augmentation/enhancing_image_RRDB.py
    python data_augmentation/processing_image.py
    ```

4. Train the models:
    - Use the Jupyter notebooks in the `base/` directory to train baseline models.
    - Use the `9_Fusion_model.ipynb` notebook to train the fusion model.

5. Evaluate the models:
    - Use the `Evaluation.ipynb` notebook to evaluate the performance of the models.

6. Run the Streamlit app:
    ```sh
    streamlit run streamlit.py
    ```

## Results
- The results of the experiments and model evaluations can be found in the `Evaluation.ipynb` notebook.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.