import pandas as pd
from datetime import datetime

def calculate_minutes_difference(date1_str: str, date2_str: str) -> float:
    date_format = "%Y-%m-%d %H:%M:%S"
    date1 = datetime.strptime(date1_str, date_format)
    date2 = datetime.strptime(date2_str, date_format)
    time_difference = date2 - date1
    difference_in_minutes = time_difference.total_seconds() / 60
    return difference_in_minutes

data = pd.read_csv(r'priroda_udivitelna\data\registration.csv')  

results = []

for folder, group in data.groupby('name_folder'):
    group = group.sort_values(by='date_registration')
    current_registration = None
    current_class = None
    start_time = None
    end_time = None
    max_count = 0
    
    for i, row in group.iterrows():
        if current_registration is None:
            current_registration = row['registration_class']
            current_class = row['class_predict']
            start_time = row['date_registration']
            end_time = row['date_registration']
            max_count = row['count']
        else:
            time_diff = calculate_minutes_difference(end_time, row['date_registration'])
            if (time_diff <= 30 and row['class_predict'] == current_class) or pd.isna(row['count']):
                end_time = row['date_registration']
                max_count = max(max_count, row['count'])
            else:
                results.append({
                    'folder_name': folder,
                    'class': current_class,
                    'date_registration_start': start_time,
                    'date_registration_end': end_time,
                    'count': min(max_count, 5)
                })
                current_registration = row['registration_class']
                current_class = row['class_predict']
                start_time = row['date_registration']
                end_time = row['date_registration']
                max_count = row['count']

    if current_registration is not None:
        results.append({
            'folder_name': folder,
            'class': current_class,
            'date_registration_start': start_time,
            'date_registration_end': end_time,
            'count': min(max_count, 5)  # Ограничиваем max_count до 5
        })

result_df = pd.DataFrame(results)
result_df.to_csv(r'priroda_udivitelna\data\registration_results.csv', index=False)

print(result_df.head(50))