# Smart Crop Recommendation 🚜🌱

A modern web application that leverages machine learning to recommend the best crop to grow based on your soil and climate conditions. Includes interactive visualizations and model evaluation insights to help you understand the decision process.

--------------------------------------------------------

## 🚀 Features
- **Step-by-step input:** User-friendly wizard for entering soil and climate parameters
- **Accurate crop prediction:** Suggests the most suitable crop (with Telugu translation)
- **Model insights:** Visualizes model performance (confusion matrix, ROC, precision-recall, and more)
- **Data exploration:** View summary statistics and feature distributions
- **Beautiful UI:** Clean, responsive, and mobile-friendly design

--------------------------------------------------------

## 🌐 Live Demo
https://smart-crop-recommendation-vkrk.onrender.com
--------------------------------------------------------

## 🛠️ Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. **Clone this repository:**
   ```powershell
   git clone <your-repo-url>
   cd smart-crop-recommendation
   ```
2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Run the app:**
   ```powershell
   python app.py
   ```
4. **Open your browser:** Go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

--------------------------------------------------------

## 🗂️ Project Structure
```
app.py                  # Main Flask app
model.py                # Model training, prediction, and plotting
Crop_recommendation.csv # Dataset
saved_models/           # Trained models and encoders
static/plots/           # Generated plots for model insights
templates/              # HTML templates
```

--------------------------------------------------------

## 📊 Example Visualizations
| Feature Distribution | Confusion Matrix | ROC Curve |
|---------------------|-----------------|-----------|
| ![Feature](static/plots/feature_distribution.png) | ![Confusion](static/plots/confusion_matrix.png) | ![ROC](static/plots/roc_curve.png) |

--------------------------------------------------------

## ☁️ Deployment
You can deploy this app for free on platforms like [Render](https://render.com/), [PythonAnywhere](https://www.pythonanywhere.com/), or [Railway](https://railway.app/). Just upload your code, set up your environment, and go live!

- Add a `Procfile` for Render/Heroku:
  ```
  web: gunicorn app:app
  ```
- Make sure your `requirements.txt` is up to date.

--------------------------------------------------------

## 📚 Technologies Used
- Python, Flask
- scikit-learn, pandas, numpy
- matplotlib, seaborn
- Bootstrap 5 (UI)

--------------------------------------------------------

## 👨‍💻 Author
**Praneeth Kalyan Gurramolla**  
[GitHub](https://github.com/218r1a7230)  

--------------------------------------------------------

## 📄 License
This project is for educational and demonstration purposes.

--------------------------------------------------------

📞 Contact
👨‍💻 Developer: Praneeth Kalyan Gurramolla
📧 Email: 218r1a7230.cmrec@gmail.com
📌 GitHub: https://github.com/218r1a7230

--------------------------------------------------------

🚀 Enjoy using the Smart Crop Recommendation System! 🌾✨
