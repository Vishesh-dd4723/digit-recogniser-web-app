# Digit Recogniser Web App

This project is a web application for recognizing handwritten digits using machine learning. It provides a simple interface where users can train, test the model. It also stores the model weights so that the next time, they can be loaded directly. Just, upload a image of digit (dims:28x28) and get predictions from the trained model.

## Installation 📥

`Application` requires Python3 to run effectively. If you don't have Python installed, you can download it from [here](https://www.python.org/downloads/).

You can also use `Docker` to deploy it in your local.

After you finished installing Python, you can install by following the steps below:

```bash
git clone https://github.com/Vishesh-dd4723/digit-recogniser-web-app.git
cd digit-recogniser-web-app

# Without Docker
# Install Virtual environment library
pip install virtualenv

# Creating a seperate environment with name "myenv"
python -m venv myenv

# Activating the virtual env
myenv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run the main.py file
python "./app/main.py"


#With Docker
# Building an image
docker build -t <image-name> .

# Deploying the image
docker run --rm --name <container-name> -d -p 8080:8080 <image-name>
```


## Usage 🛠️

1. After running the `main.py`, the server will start.
1. Open `http://localhost:8080/docs` in your browser
1. You will see the swagger UI of the supported endpoints and schemas.
1. Initialize the model by hitting ` POST http://localhost:8080/digitRecogniser/initialize`
1. Then you can choose the file to be used from training or simply load (if weights exist).
1. Then you can use any image to predict.


## Contributing 🤝

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Some of the things to be worked on: 
1. At the moment, the app only works on pre-present `csv` files for training. So create a mechanism to upload and perform CRUD operations on files.
2. The app consists of only back-end. Front-end is need to be created.