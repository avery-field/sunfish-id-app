## 🐟 Sunfish Identification Web App 🐟

The Sunfish family (Centrarchidae) includes species of fish beloved by freshwater anglers, like largemouth bass, bluegill, and crappie. Some of these species can be tricky to tell apart, so I built a web app that lets you upload a photo of your catch and identify the species using machine learning and computer vision! 🎣

💡 How I built it:
- Collected my own dataset of all 14 sunfish species using web scraping from iNaturalist.
- Fine-tuned a ResNet18 model for image classification.
- Built a Flask web app and deployed it on Heroku (note: memory limits reduce accuracy in the live app).

This project was a fun way to combine one of my hobbies with hands-on experience in ML and computer vision.

# Explore the dataset:
https://www.kaggle.com/datasets/averyfield98/sunfish-image-dataset/data

# See how I trained the model here:
https://www.kaggle.com/code/averyfield98/sunfish-species-identification-fine-tune-resnet18

## Installation
Clone this repository and navigate to the project directory:


```
git clone https://github.com/avery-field/sunfish-id-app.git
cd sunfish-id-app
```

Create a virtual environment and install the required packages:


```
python -m venv venv
source venv/bin/activate  # on Windows, use "venv\Scripts\activate"
pip install -r requirements.txt
```


## Usage
Run the app locally using Flask:

```
python -m flask run
```
Navigate to http://localhost:5000/ in your web browser to access the app.
