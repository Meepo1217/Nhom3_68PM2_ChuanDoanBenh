"""
Module: features.py
Mục đích: Chứa các hàm xử lý EDA (Khám phá dữ liệu), làm sạch dữ liệu thô,
xoay ngang ma trận đặc trưng, mã hóa nhãn và chia tập Train/Test cho hệ thống
chẩn đoán bệnh sơ bộ qua triệu chứng.
Quy tắc đặt tên: Tên biến rõ ràng, tường minh, không viết tắt.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd


def load_all_raw_datasets(
    raw_directory_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Đọc 4 tệp dữ liệu thô từ thư mục data/raw/ mà không làm thay đổi file gốc."""
  # Đọc bảng dữ liệu chính chứa tên bệnh (Disease) và 17 cột triệu chứng (Symptom_1 -> Symptom_17)
  dataframe_raw_dataset = pd.read_csv(raw_directory_path / "dataset.csv")

  # Đọc bảng trọng số mức độ nghiêm trọng (weight) của từng triệu chứng
  dataframe_raw_severity = pd.read_csv(
      raw_directory_path / "Symptom-severity.csv"
  )

  # Đọc bảng mô tả thông tin chi tiết của 41 loại bệnh
  dataframe_raw_description = pd.read_csv(
      raw_directory_path / "symptom_Description.csv"
  )

  # Đọc bảng khuyến nghị xử lý / phòng ngừa (4 bước) cho từng loại bệnh
  dataframe_raw_precaution = pd.read_csv(
      raw_directory_path / "symptom_precaution.csv"
  )

  # Trả về cả 4 bảng dữ liệu dạng DataFrame
  return (
      dataframe_raw_dataset,
      dataframe_raw_severity,
      dataframe_raw_description,
      dataframe_raw_precaution,
  )


def generate_exploratory_summary(
    dataframe_raw_dataset: pd.DataFrame, dataframe_raw_severity: pd.DataFrame
) -> dict:
  """Thống kê các chỉ số khám phá dữ liệu (EDA) trên bảng thô trước khi làm sạch."""
  # Tạo danh sách tên 17 cột triệu chứng từ Symptom_1 đến Symptom_17 (range(1, 18) dừng trước 18)
  symptom_column_names = [f"Symptom_{index}" for index in range(1, 18)]

  # Duyệt qua từng cột triệu chứng, bỏ giá trị NaN (.dropna()), lấy giá trị duy nhất (.unique())
  # và gom vào tập hợp (set) để đếm số chuỗi triệu chứng gốc trước khi xóa khoảng trắng
  raw_unique_symptom_set = {
      symptom_text
      for column_name in symptom_column_names
      for symptom_text in dataframe_raw_dataset[column_name].dropna().unique()
  }

  # Xóa khoảng trắng thừa ở đầu và cuối từng chuỗi triệu chứng bằng .strip()
  stripped_unique_symptom_set = {
      symptom_text.strip() for symptom_text in raw_unique_symptom_set
  }

  # Lấy tập hợp tên triệu chứng chuẩn hóa bên bảng Symptom-severity.csv để đối chiếu
  severity_symptom_set = set(dataframe_raw_severity["Symptom"].str.strip())

  # Tìm các tên triệu chứng xuất hiện trong dataset.csv nhưng chưa khớp tên với bảng trọng số
  unmatched_symptom_names = sorted(
      list(stripped_unique_symptom_set - severity_symptom_set)
  )

  # Đếm số lượng triệu chứng thực tế (không phải NaN) của từng bệnh nhân theo chiều ngang (axis=1)
  symptom_count_per_patient = (
      dataframe_raw_dataset[symptom_column_names].notnull().sum(axis=1)
  )

  # Đóng gói toàn bộ kết quả thống kê vào từ điển (dictionary) rõ nghĩa
  exploratory_summary_dictionary = {
      "total_raw_rows": int(dataframe_raw_dataset.shape[0]),  # Tổng số dòng gốc
      "total_raw_columns": int(
          dataframe_raw_dataset.shape[1]
      ),  # Tổng số cột gốc
      "total_unique_diseases": int(
          dataframe_raw_dataset["Disease"].nunique()
      ),  # Số lượng bệnh duy nhất
      "total_duplicate_rows": int(
          dataframe_raw_dataset.duplicated().sum()
      ),  # Số dòng trùng lặp hoàn toàn
      "total_unique_records": int(
          len(dataframe_raw_dataset.drop_duplicates())
      ),  # Số dòng độc nhất còn lại
      "min_symptoms_per_patient": int(
          symptom_count_per_patient.min()
      ),  # Ít triệu chứng nhất / 1 bệnh nhân
      "max_symptoms_per_patient": int(
          symptom_count_per_patient.max()
      ),  # Nhiều triệu chứng nhất / 1 bệnh nhân
      "average_symptoms_per_patient": round(
          float(symptom_count_per_patient.mean()), 2
      ),  # Trung bình triệu chứng / bệnh nhân
      "raw_symptom_string_count": len(
          raw_unique_symptom_set
      ),  # Số chuỗi triệu chứng gốc
      "stripped_symptom_string_count": len(
          stripped_unique_symptom_set
      ),  # Số chuỗi sau .strip()
      "unmatched_symptoms_with_severity": (
          unmatched_symptom_names
      ),  # Danh sách tên lệch với bảng trọng số
  }

  return exploratory_summary_dictionary


def clean_raw_symptom_dataset(
    dataframe_raw_dataset: pd.DataFrame,
) -> pd.DataFrame:
  """Chuẩn hóa khoảng trắng, sửa lỗi chính tả và xóa các bản ghi trùng lặp."""
  # Danh sách 17 cột triệu chứng cần làm sạch
  symptom_column_names = [f"Symptom_{index}" for index in range(1, 18)]

  # Từ điển sửa 3 tên triệu chứng bị lệch khoảng trắng / dấu gạch dưới so với bảng trọng số
  symptom_typo_dictionary = {
      "dischromic _patches": "dischromic_patches",
      "spotting_ urination": "spotting_urination",
      "foul_smell_of urine": "foul_smell_ofurine",
  }

  # Từ điển sửa 1 lỗi chính tả tên bệnh để đồng nhất với bảng mô tả bệnh
  disease_typo_dictionary = {
      "Dimorphic hemmorhoids(piles)": "Dimorphic hemorrhoids(piles)",
  }

  # Tạo bản sao (.copy()) để không làm hỏng dữ liệu gốc ban đầu
  dataframe_cleaned_text = dataframe_raw_dataset.copy()

  # Xóa khoảng trắng hai đầu ở cột Disease rồi thay thế tên bệnh bị sai chính tả
  dataframe_cleaned_text["Disease"] = (
      dataframe_cleaned_text["Disease"]
      .str.strip()
      .replace(disease_typo_dictionary)
  )

  # Duyệt qua từng cột Symptom_1 -> Symptom_17:
  # Nếu ô không phải NaN (pd.notna), cắt khoảng trắng (.strip()) và sửa lỗi chính tả nếu có trong map;
  # nếu là NaN thì giữ nguyên np.nan
  for column_name in symptom_column_names:
    dataframe_cleaned_text[column_name] = dataframe_cleaned_text[
        column_name
    ].apply(
        lambda cell_value: (
            symptom_typo_dictionary.get(cell_value.strip(), cell_value.strip())
            if pd.notna(cell_value)
            else np.nan
        )
    )

  # Xóa các dòng bị trùng lặp hoàn toàn sau khi chuẩn hóa chữ, sau đó đánh lại index từ 0
  dataframe_unique_records = dataframe_cleaned_text.drop_duplicates().reset_index(
      drop=True
  )

  return dataframe_unique_records


def build_severity_weight_dictionary(
    dataframe_raw_severity: pd.DataFrame,
) -> dict:
  """Làm sạch bảng Symptom-severity.csv và trả về từ điển {triệu_chứng: trọng_số}."""
  # Loại bỏ dòng rác 'prognosis' (tên cột bị lọt nhầm thành 1 hàng dữ liệu trong file gốc)
  dataframe_filtered_severity = dataframe_raw_severity[
      dataframe_raw_severity["Symptom"].str.strip() != "prognosis"
  ].copy()

  # Xóa khoảng trắng thừa ở cột tên triệu chứng
  dataframe_filtered_severity["Symptom"] = dataframe_filtered_severity[
      "Symptom"
  ].str.strip()

  # Gom nhóm theo tên triệu chứng và lấy giá trị weight lớn nhất (.max()) nếu bị trùng tên
  # (ví dụ 'fluid_overload' xuất hiện 2 lần với trọng số 6 và 4), rồi chuyển thành dict
  severity_weight_dictionary = (
      dataframe_filtered_severity.groupby("Symptom")["weight"].max().to_dict()
  )

  return severity_weight_dictionary


def create_symptom_feature_matrices(
    dataframe_unique_records: pd.DataFrame, severity_weight_dictionary: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Xoay ngang 17 cột triệu chứng thành ma trận nhị phân (0/1) và ma trận trọng số."""
  symptom_column_names = [f"Symptom_{index}" for index in range(1, 18)]

  # Gom tất cả tên triệu chứng khác NaN từ 17 cột, loại trùng và sắp xếp theo thứ tự bảng chữ cái
  sorted_unique_symptoms = sorted({
      symptom_text
      for column_name in symptom_column_names
      for symptom_text in dataframe_unique_records[column_name].dropna().unique()
  })

  # Khởi tạo ma trận nhị phân toàn số 0: mỗi hàng là 1 hồ sơ, mỗi cột là 1 triệu chứng
  matrix_binary_symptoms = pd.DataFrame(
      0, index=dataframe_unique_records.index, columns=sorted_unique_symptoms
  )

  # Duyệt qua từng cột triệu chứng và từng dòng: nếu có triệu chứng thì gán giá trị 1 tại đúng cột đó
  for column_name in symptom_column_names:
    for row_index, symptom_text in dataframe_unique_records[column_name].items():
      if pd.notna(symptom_text):
        matrix_binary_symptoms.loc[row_index, symptom_text] = 1

  # Tạo ma trận trọng số bằng cách nhân từng cột 0/1 với mức độ nghiêm trọng (mặc định = 1 nếu không tìm thấy)
  matrix_weighted_symptoms = matrix_binary_symptoms.copy()
  for symptom_column in matrix_weighted_symptoms.columns:
    severity_score = severity_weight_dictionary.get(symptom_column, 1)
    matrix_weighted_symptoms[symptom_column] = (
        matrix_weighted_symptoms[symptom_column] * severity_score
    )

  return matrix_binary_symptoms, matrix_weighted_symptoms


def split_and_save_processed_data(
    matrix_binary_symptoms: pd.DataFrame,
    matrix_weighted_symptoms: pd.DataFrame,
    series_disease_names: pd.Series,
    processed_directory_path: Path,
    reports_directory_path: Path,
    exploratory_summary_dictionary: dict,
) -> dict:
  """Mã hóa nhãn bệnh, chia Train/Test (có phân tầng) và lưu toàn bộ dữ liệu sạch bằng Pandas/NumPy thuần."""
  processed_directory_path.mkdir(parents=True, exist_ok=True)
  reports_directory_path.mkdir(parents=True, exist_ok=True)

  # 1. Mã hóa nhãn bệnh bằng Pandas Categorical (thay thế LabelEncoder)
  # Sắp xếp danh sách tên bệnh theo bảng chữ cái và gán mã số nguyên 0 -> 40
  sorted_disease_classes = np.array(sorted(series_disease_names.unique()))
  disease_to_index_map = {
      disease_name: index
      for index, disease_name in enumerate(sorted_disease_classes)
  }
  encoded_disease_labels = series_disease_names.map(
      disease_to_index_map
  ).to_numpy()

  # 2. Chia Train (80%) / Test (20%) có phân tầng (Stratified) bằng NumPy thuần
  random_generator = np.random.default_rng(seed=42)
  train_indices_list = []
  test_indices_list = []

  for class_index in range(len(sorted_disease_classes)):
    class_row_indices = np.where(encoded_disease_labels == class_index)[0]
    shuffled_indices = random_generator.permutation(class_row_indices)
    test_sample_size = max(1, int(round(len(shuffled_indices) * 0.2)))
    test_indices_list.extend(shuffled_indices[:test_sample_size])
    train_indices_list.extend(shuffled_indices[test_sample_size:])

  train_indices_array = np.array(sorted(train_indices_list))
  test_indices_array = np.array(sorted(test_indices_list))

  features_train_binary = matrix_binary_symptoms.iloc[
      train_indices_array
  ].reset_index(drop=True)
  features_test_binary = matrix_binary_symptoms.iloc[
      test_indices_array
  ].reset_index(drop=True)
  labels_train_encoded = encoded_disease_labels[train_indices_array]
  labels_test_encoded = encoded_disease_labels[test_indices_array]

  # 3. Lưu toàn bộ file dữ liệu sạch và nhật ký làm sạch
  matrix_binary_symptoms.to_csv(
      processed_directory_path / "matrix_binary_symptoms.csv", index=False
  )
  matrix_weighted_symptoms.to_csv(
      processed_directory_path / "matrix_weighted_symptoms.csv", index=False
  )
  features_train_binary.to_csv(
      processed_directory_path / "features_train_binary.csv", index=False
  )
  features_test_binary.to_csv(
      processed_directory_path / "features_test_binary.csv", index=False
  )
  np.save(
      processed_directory_path / "labels_train_encoded.npy", labels_train_encoded
  )
  np.save(
      processed_directory_path / "labels_test_encoded.npy", labels_test_encoded
  )
  np.save(
      processed_directory_path / "disease_class_names.npy",
      sorted_disease_classes,
  )

  cleaning_log_dictionary = {
      **exploratory_summary_dictionary,
      "final_feature_count": int(matrix_binary_symptoms.shape[1]),
      "train_sample_count": int(features_train_binary.shape[0]),
      "test_sample_count": int(features_test_binary.shape[0]),
  }

  with open(
      reports_directory_path / "cleaning_log.json", "w", encoding="utf-8"
  ) as log_file:
    json.dump(cleaning_log_dictionary, log_file, ensure_ascii=False, indent=2)

  return {
      "features_train_binary": features_train_binary,
      "features_test_binary": features_test_binary,
      "labels_train_encoded": labels_train_encoded,
      "labels_test_encoded": labels_test_encoded,
      "disease_class_names": sorted_disease_classes,
  }