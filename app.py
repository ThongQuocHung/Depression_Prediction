import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file, redirect, url_for
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder

app = Flask(__name__)

# Tạo thư mục cho model và data
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('exports', exist_ok=True)

# Định nghĩa các giá trị cho các trường dữ liệu danh mục
gender_values = ['Male', 'Female', 'Other']
yes_no_values = ['Yes', 'No']
professional_status = ['Student', 'Working Professional']
satisfaction_levels = ['1', '2', '3', '4', '5']
pressure_levels = ['1', '2', '3', '4', '5']
city_options = ['Ho Chi Minh', 'Hanoi', 'Da Nang', 'Other']  # Cập nhật theo dữ liệu mẫu
profession_options = ['Student', 'Engineer', 'Teacher', 'Developer', 'Other']  # Cập nhật theo dữ liệu mẫu
dietary_habits = ['Good', 'Average', 'Poor']  # Cập nhật theo dữ liệu mẫu
degree_options = ['High School', 'Bachelor', 'Master', 'PhD', 'Other']
financial_stress_levels = ['1', '2', '3', '4', '5']  # Thêm mức độ stress tài chính


# Tải model và preprocessor nếu có
def load_model():
    try:
        # Tải model đã được lưu
        with open('models/stacking_clf.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        with open('models/encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        print("Model loaded successfully")
        return model, preprocessor, encoder
    except FileNotFoundError:
        print("Model files not found. Please train the model first.")
        return None, None, None
    
def save_to_history(data):
    """
    Lưu dữ liệu dự đoán vào file lịch sử
    
    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame chứa thông tin về người dùng và kết quả dự đoán
    """
    import pandas as pd
    import os
    from datetime import datetime
    
    # Thêm timestamp
    data['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Đường dẫn file lịch sử
    history_file = 'data/prediction_history.xlsx'
    
    # Kiểm tra xem file history đã tồn tại chưa
    if os.path.exists(history_file):
        # Nếu đã có, đọc file và thêm dữ liệu mới
        history_df = pd.read_excel(history_file)
        history_df = pd.concat([history_df, data], ignore_index=True)
    else:
        # Nếu chưa có, tạo mới từ dữ liệu hiện tại
        history_df = data
    
    # Lưu lại file
    history_df.to_excel(history_file, index=False)
    
    print(f"Đã lưu vào lịch sử: {history_file}")


# Chuyển đổi dữ liệu đầu vào thành DataFrame
def prepare_input_data(form_data):
    data = {
        'Name': [form_data.get('name', '')],
        'Age': [int(form_data.get('age', 0))],
        'Gender': [form_data.get('gender', '')],
        'City': [form_data.get('city', '')],
        'Profession': [form_data.get('profession', '')],
        'Working Professional or Student': [form_data.get('work_status', '')],
        'Academic Pressure': [form_data.get('academic_pressure', '')],
        'Work Pressure': ['3'],
        'CGPA': [float(form_data.get('cgpa', 0))],
        'Study Satisfaction': [form_data.get('study_satisfaction', '')],
        'Job Satisfaction': ['3'],
        'Work/Study Hours': [int(form_data.get('work_study_hours', 0))],
        'Have you ever had suicidal thoughts ?': [form_data.get('suicidal_thoughts', '')],
        'Financial Stress': [form_data.get('financial_stress', '')],
        'Family History of Mental Illness': [form_data.get('family_history', '')],
        'Sleep Duration': [int(form_data.get('sleep_duration', 0))],
        'Dietary Habits': [form_data.get('dietary_habits', '')],
        'Degree': [form_data.get('degree', '')],

    }
    return pd.DataFrame(data)

# Trang chủ
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', 
                           gender_values=gender_values,
                           yes_no_values=yes_no_values,
                           professional_status=professional_status,
                           satisfaction_levels=satisfaction_levels,
                           pressure_levels=pressure_levels,
                           city_options=city_options,
                           profession_options=profession_options,
                           dietary_habits=dietary_habits,
                           degree_options=degree_options,
                           financial_stress_levels=financial_stress_levels)  # Cập nhật tham số

# Xử lý dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    model, preprocessor, encoder = load_model()
    if model is None or preprocessor is None or encoder is None:
        return render_template('error.html', message="Model chưa được tải. Vui lòng kiểm tra thư mục models.")
    
    try:
        # Lấy dữ liệu từ form
        input_data = prepare_input_data(request.form)
  
        # Lưu dữ liệu gốc
        original_data = input_data.copy()
        
        # Áp dụng target encoding
        input_data[['City_encoded', 'Profession_encoded']] = encoder.transform(input_data[['City', 'Profession']])
        
        # Tiền xử lý dữ liệu
        input_preprocessed = preprocessor.transform(input_data)
        
        
        # Dự đoán
        prediction = model.predict(input_preprocessed)
        prediction_proba = model.predict_proba(input_preprocessed)
        depression_probability = prediction_proba[0][1] * 100  # Xác suất trầm cảm (%)
        prediction_label = "Depression" if prediction[0] == 1 else "No Depression"
        
        # Lưu dự đoán vào DataFrame
        original_data['Depression_Prediction'] = prediction_label
        original_data['Depression_Probability'] = f"{depression_probability:.2f}%"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exports/prediction_{timestamp}.xlsx"
        original_data.to_excel(filename, index=False)
            
        # Lưu vào lịch sử
        save_to_history(original_data)
        
        return render_template('result.html', 
                               prediction=prediction_label,
                               probability=f"{depression_probability:.2f}%", 
                               data=request.form,
                               filename=filename)
    except Exception as e:
        return render_template('error.html', message=f"Lỗi khi dự đoán: {str(e)}")

# Route để tải xuống file Excel
@app.route('/download/<path:filename>', methods=['GET'])
def download(filename):
    return send_file(filename, as_attachment=True)

# Route để xem lịch sử
@app.route('/history', methods=['GET'])
def history():
    history_file = 'data/prediction_history.xlsx'
    if os.path.exists(history_file):
        history_df = pd.read_excel(history_file)
        records = history_df.to_dict('records')
        return render_template('history.html', records=records)
    else:
        return render_template('history.html', records=[])

# Route để tải xuống lịch sử dưới dạng Excel
@app.route('/download_history', methods=['GET'])
def download_history():
    history_file = 'data/prediction_history.xlsx'
    if os.path.exists(history_file):
        return send_file(history_file, as_attachment=True)
    else:
        return redirect(url_for('history'))

# Mã để tạo model mẫu nếu chưa có model
@app.route('/create_sample_model', methods=['GET'])
def create_sample_model():
    # Đây chỉ là mô hình mẫu đơn giản để demo
    # Trong thực tế, bạn nên thay thế bằng mô hình đã huấn luyện từ notebook
    
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Tạo preprocessor đơn giản
    # Cập nhật numeric_features theo mẫu dữ liệu
    numeric_features = ['Age', 'CGPA', 'Work/Study Hours', 'Sleep Duration']  # Thêm Sleep Duration vào numeric
    categorical_features = ['Gender', 'City', 'Profession', 'Degree', 'Working Professional or Student', 
                       'Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction',
                       'Have you ever had suicidal thoughts ?', 'Dietary Habits', 'Financial Stress',
                       'Family History of Mental Illness']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('drop', 'drop', ['Name'])  # Only drop 'Name'
        ]
    )   
    
    # Tạo mô hình mẫu
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    meta_model = LogisticRegression()
    
    stacking_clf = StackingClassifier(
        estimators=[('rf', rf)],
        final_estimator=meta_model,
        passthrough=False
    )
    
    # Tạo mẫu dữ liệu để fit preprocessor và encoder theo định dạng mẫu
    sample_data = pd.DataFrame({
    'Name': ['Sample Person'],
    'Age': [25],
    'CGPA': [8.5],
    'Work/Study Hours': [8],
    'Sleep Duration': [7],  # Đổi thành số nguyên
    'Gender': ['Male'],
    'City': ['Ho Chi Minh'],
    'Profession': ['Student'],
    'Degree': ['Bachelor'],
    'Working Professional or Student': ['Student'],
    'Academic Pressure': ['3'],
    'Work Pressure': ['3'],
    'Study Satisfaction': ['3'],
    'Job Satisfaction': ['3'],
    'Have you ever had suicidal thoughts ?': ['No'],
    'Dietary Habits': ['Good'],
    'Financial Stress': ['3'],
    'Family History of Mental Illness': ['No']
})
    
    # Tạo encoder mẫu
    encoder = TargetEncoder(cols=['City', 'Profession'])
    # Giả lập fit với mẫu dữ liệu và target
    sample_target = np.array([0])  # Giả sử không có trầm cảm để fit encoder
    encoder.fit(sample_data[['City', 'Profession']], sample_target)
    
    sample_data[['City_encoded', 'Profession_encoded']] = encoder.transform(sample_data[['City', 'Profession']])
    
    # Fit preprocessor với mẫu dữ liệu
    preprocessor.fit(sample_data)
    
    # Fit model với dữ liệu đã được tiền xử lý
    X_processed = preprocessor.transform(sample_data)
    stacking_clf.fit(X_processed, sample_target)
    
    # Lưu model và preprocessor
    with open('models/stacking_clf.pkl', 'wb') as f:
        pickle.dump(stacking_clf, f)
    with open('models/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    with open('models/encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    
    return "Đã tạo mẫu model để demo. Trong thực tế, bạn nên thay thế bằng model đã train từ notebook."

if __name__ == '__main__':
    app.run(debug=True)