import pandas as pd
from datetime import datetime
from db import DB

def calculate_minutes_difference(date1: datetime, date2: datetime) -> float:
    time_difference = date2 - date1
    difference_in_minutes = time_difference.total_seconds() / 60
    return difference_in_minutes

def process_data(file_path: str, db_manager: DB):
    # Загрузка данных
    data = pd.read_csv(r'priroda_udivitelna\data\registration.csv', sep=',')  
    data.dropna(inplace=True)

    # Добавление изображений в базу данных
    for i, row in data.iterrows():
        db_manager.add_image(
            folder_name=row['name_folder'],
            image_name=row['name'],
            class_predict=row['class_predict'],
            registration_class=row['registration_class'],
            registration_date=row['date_registration'],
            count=row['count'],
            max_count=row['max_count']
        )

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
                time_diff = calculate_minutes_difference(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S"), datetime.strptime(row['date_registration'], "%Y-%m-%d %H:%M:%S"))
                if (time_diff <= 30 and row['class_predict'] == current_class) or pd.isna(row['count']):
                    end_time = row['date_registration']
                    max_count = max(max_count, row['count'])
                else:
                    registration_id = db_manager.add_registration(
                        folder_name=folder,
                        class_name=current_class,
                        date_registration_start=start_time,
                        date_registration_end=end_time,
                        max_count=min(int(max_count), 5)
                    )
                    # Обновление внешнего ключа в Image
                    images_to_update = db_manager.get_images_in_range(
                        folder_name=folder,
                        start_time=datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S"),
                        end_time=datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
                    )
                    for image in images_to_update:
                        db_manager.update_image_registration_id(image.id, registration_id)
                    
                    results.append({
                        'folder_name': folder,
                        'class': current_class,
                        'date_registration_start': start_time,
                        'date_registration_end': end_time,
                        'count': min(int(max_count), 5)
                    })
                    
                    current_registration = row['registration_class']
                    current_class = row['class_predict']
                    start_time = row['date_registration']
                    end_time = row['date_registration']
                    max_count = row['count']

        if current_registration is not None:
            registration_id = db_manager.add_registration(
                folder_name=folder,
                class_name=current_class,
                date_registration_start=start_time,
                date_registration_end=end_time,
                max_count=min(int(max_count), 5)
            )
            # Обновление внешнего ключа в Image
            images_to_update = db_manager.get_images_in_range(
                folder_name=folder,
                start_time=datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S"),
                end_time=datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            )
            for image in images_to_update:
                db_manager.update_image_registration_id(image.id, registration_id)
            
            results.append({
                'folder_name': folder,
                'class': current_class,
                'date_registration_start': start_time,
                'date_registration_end': end_time,
                'count': min(int(max_count), 5)
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(r'priroda_udivitelna\data\registration_results.csv', index=False)


if __name__ == "__main__":
    db_manager = DB('sqlite:///data/db.sqlite')
    process_data(r'priroda_udivitelna\data\registration.csv', db_manager)