'''
คู่มือการใช้งานสำหรับแอปพลิเคชันตรวจจับวัตถุ:

1. **ติดตั้ง Flask **:
   - ใช้คำสั่ง `pip install Flask` เพื่อติดตั้งไลบรารีที่จำเป็น

2. **สร้างโฟลเดอร์ `uploads`**: 
   - ตรวจสอบว่าโฟลเดอร์นี้มีอยู่แล้วในโปรเจ็กต์ หากไม่ให้สร้างใหม่ เพื่อให้โปรแกรมสามารถบันทึกภาพที่ผู้ใช้ส่งมาได้

3. **เก็บโมเดลในโฟลเดอร์ `models`**:
   - ให้แน่ใจว่าไฟล์โมเดล (เช่น `cnn_model.h5`, `vgg16_model.h5`, เป็นต้น) อยู่ในที่อยู่ที่กำหนดในโค้ด

4. **รันแอปพลิเคชัน**:
   - ใช้คำสั่ง `python your_script.py` เพื่อรันแอปพลิเคชัน Flask

5. **เข้าถึงเว็บแอปพลิเคชัน**:
   - เปิดเว็บเบราว์เซอร์ไปที่ `http://127.0.0.1:5000/` เพื่อดูหน้าแรกของแอปพลิเคชัน

6. **อัปโหลดภาพ**:
   - ในหน้าแรกของแอปพลิเคชันจะมีแบบฟอร์มให้เลือกภาพที่ต้องการทดสอบและเลือกโมเดลที่ต้องการใช้งาน (เช่น CNN, VGG16, ResNet50, InceptionV3, MobileNetV2)

7. **ดูผลลัพธ์**:
   - หลังจากอัปโหลดภาพ ระบบจะทำการพยากรณ์ และแสดงผลลัพธ์ว่าภาพที่ส่งเข้ามานั้นเป็น Yims Cafe Glass หรือไม่

'''

from flask import Flask, request, jsonify, render_template  # นำเข้าโมดูล Flask ที่จำเป็น
from tensorflow.keras.models import load_model  # นำเข้าโมดูลสำหรับโหลดโมเดล
from tensorflow.keras.preprocessing import image  # นำเข้าโมดูลสำหรับการประมวลผลภาพ
import numpy as np  # นำเข้า NumPy สำหรับการจัดการอาเรย์
import os  # นำเข้าโมดูลสำหรับการจัดการไฟล์และโฟลเดอร์

# สร้างแอป Flask
app = Flask(__name__)

# โฟลเดอร์ที่ใช้เก็บภาพที่อัปโหลด
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# โหลดโมเดลทั้งหมดที่ใช้ในการตรวจจับ
models = {
    'cnn': load_model('model/cnn_model.h5'),
    'vgg16': load_model('model/vgg16_model.h5'),
    'resnet50': load_model('model/resnet50_model.h5'),
    'inceptionv3': load_model('model/inceptionv3_model.h5'),
    'mobilenetv2': load_model('model/mobilenetv2_model.h5')
}

# ฟังก์ชันสำหรับเตรียมภาพก่อนทำการพยากรณ์
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # โหลดภาพและปรับขนาด
    img_array = image.img_to_array(img)  # แปลงภาพเป็นอาเรย์
    img_array = np.expand_dims(img_array, axis=0)  # เพิ่มมิติใหม่สำหรับ batch
    img_array /= 255.0  # ทำการ normalize ค่าพิกเซลให้เป็นช่วง [0, 1]
    return img_array  # ส่งกลับอาเรย์ภาพที่เตรียมพร้อม

@app.route('/')
def index():
    # แสดงหน้าแรก
    return render_template('index.html')

# API สำหรับจัดการการอัปโหลดภาพและการพยากรณ์
@app.route('/predict', methods=['POST'])
def predict():
    # ตรวจสอบว่ามีไฟล์และชื่อโมเดลถูกส่งมาหรือไม่
    if 'file' not in request.files or 'model_name' not in request.form:
        return jsonify({"error": "No file or model name provided"}), 400

    # รับไฟล์ที่อัปโหลดและชื่อโมเดล
    file = request.files['file']
    model_name = request.form['model_name']

    # บันทึกไฟล์ลงในโฟลเดอร์ uploads
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # เตรียมภาพสำหรับการพยากรณ์
    img = prepare_image(filepath)

    # ตรวจสอบว่าโมเดลมีอยู่หรือไม่
    if model_name not in models:
        return jsonify({"error": "Model not found"}), 400

    # รับโมเดลและทำการพยากรณ์
    model = models[model_name]
    prediction = model.predict(img)

    # แปลผลลัพธ์
    if prediction[0] > 0.5:
        result = "Dog detected"  # ถ้าผลลัพธ์มากกว่า 0.5 แสดงว่าเป็น Dog
    else :
        result = "Cat detected"  # ถ้าผลลัพธ์น้อยกว่าหรือเท่ากับ 0.5 แสดงว่าพบ Cat

    # ส่งผลลัพธ์กลับไปยังผู้ใช้
    return jsonify({"result": result, "model_used": model_name})

if __name__ == '__main__':
    # สร้างโฟลเดอร์ uploads หากยังไม่มี
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)  # รันแอปในโหมด debug
