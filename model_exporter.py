import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from category_encoders import TargetEncoder

def export_model_from_notebook(stacking_clf, preprocessor, encoder):
    """
    Hàm để xuất model từ notebook
    
    Parameters:
    -----------
    stacking_clf : Mô hình stacking đã được huấn luyện
    preprocessor : Bộ tiền xử lý dữ liệu
    encoder : TargetEncoder đã được fit
    """
    
    # Tạo thư mục models nếu chưa tồn tại
    os.makedirs('models', exist_ok=True)
    
    # Lưu stacking classifier
    with open('models/stacking_clf.pkl', 'wb') as f:
        pickle.dump(stacking_clf, f)
    
    # Lưu preprocessor
    with open('models/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # Lưu encoder
    with open('models/encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    
    print("Mô hình đã được xuất thành công!")
    print("Các file xuất ra:")
    print("- models/stacking_clf.pkl")
    print("- models/preprocessor.pkl")
    print("- models/encoder.pkl")
    print("\nHãy đảm bảo copy các file này vào thư mục 'models' của ứng dụng Flask.")

if __name__ == "__main__":
    # Kiểm tra xem model đã tồn tại chưa
    if (os.path.exists('models/stacking_clf.pkl') and 
        os.path.exists('models/preprocessor.pkl') and 
        os.path.exists('models/encoder.pkl')):
        print("Mô hình đã tồn tại!")
        
        # Tải và kiểm tra mô hình
        try:
            with open('models/stacking_clf.pkl', 'rb') as f:
                model = pickle.load(f)
                
            print(f"Tải thành công mô hình: {type(model).__name__}")
            
            # Kiểm tra loại mô hình
            if isinstance(model, StackingClassifier):
                print("✅ Mô hình Stacking Classifier hợp lệ.")
                
                # Hiển thị thông tin mô hình
                print("\nCấu trúc mô hình:")
                print(f"- Số lượng mô hình cơ sở: {len(model.estimators_)}")
                print(f"- Mô hình meta: {type(model.final_estimator_).__name__}")
                
                # Hiển thị mô hình cơ sở
                print("\nCác mô hình cơ sở:")
                for name, estimator in model.estimators:
                    print(f"- {name}: {type(estimator).__name__}")
            else:
                print("⚠️ Mô hình không phải là Stacking Classifier.")
        except Exception as e:
            print(f"❌ Lỗi khi tải mô hình: {str(e)}")
    else:
        print("Chưa tìm thấy mô hình. Vui lòng chạy hàm export_model_from_notebook() từ notebook.")
        print("\nVí dụ cách sử dụng trong notebook:")
        print("from export_model import export_model_from_notebook")
        print("export_model_from_notebook(stacking_clf, preprocessor, encoder)")