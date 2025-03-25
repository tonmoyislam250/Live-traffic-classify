
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
    - Download the GTSRB dataset from [here](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html).
    - Additional links:
      - [Final Trainned Images](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip)
      - [Final Test Images](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip)
      - [Final Test GT](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip)
    - Place the dataset in the `GTSRB/` directory.

2. For simple enhancement, use the files in the `simple_enhance` notebooks:
    - There are two models: Resnet50 and CNN.
3. For RRDM enhancement, use the `RRDM_enhance` folder to create and test models:
    - Refer to the [ESRGAN repository](https://github.com/xinntao/ESRGAN).
4. Run the Streamlit app with just Images:
    ```sh
    streamlit run streamlit.py
    ```
5. Run the streamlit app with `Live Video Detection and Classification`
    ```sh
    streamlit run video.py
    ```