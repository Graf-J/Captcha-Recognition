# Captcha-Recognition

## Data

<ul>
    <li>Hugging-Face Data: https://huggingface.co/datasets/hammer888/captcha-data</li>
    <li>Kaggle Data: https://www.kaggle.com/datasets/akashguna/large-captcha-dataset</li>
</ul>

## Setup
Install dependencies:
```bash
uv sync
```

Pull the data using the Python scripts inside the `scripts/` folder

Copy `v2.pth` into folder `weights/crnn/`

Copy `test_file_list_crnn.txt` into folder `notebooks/crnn`. You probably want to change the folder path names stored inside `test_file_list_crnn.txt` to match the names of your local folderstructure.

Copy `test_file_list_convtrans.txt` into folder `notebooks/convtrans`. You probably want to change the folder path names stored inside `test_file_list_convtrans.txt` to match the names of your local folderstructure.