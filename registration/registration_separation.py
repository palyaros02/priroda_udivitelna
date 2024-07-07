import pandas as pd
from datetime import datetime
from db import DB, Image

def calculate_minutes_difference(date1: datetime, date2: datetime) -> float:
    time_difference = date2 - date1
    difference_in_minutes = time_difference.total_seconds() / 60
    return difference_in_minutes

def process_data(db_manager: DB):
    # Извлечение данных из таблицы Image
    images = db_manager.session.query(Image).all()

    data = [{
        'folder_name': image.folder_name,
        'image_name': image.image_name,
        'class_predict': image.class_predict,
        'registration_class': image.registration_class,
        'registration_date': image.registration_date,
        'count': image.count
    } for image in images]

    df = pd.DataFrame(data)
    df['date_registration'] = pd.to_datetime(df['registration_date'])

    results = []

    for folder, group in df.groupby('folder_name'):
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
                    registration_id = db_manager.add_registration(
                        folder_name=folder,
                        class_name=current_class,
                        date_registration_start=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                        date_registration_end=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                        max_count=min(int(max_count), 5)
                    )
                    # Обновление внешнего ключа в Image
                    images_to_update = db_manager.get_images_in_range(
                        folder_name=folder,
                        start_time=start_time,
                        end_time=end_time
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
                date_registration_start=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                date_registration_end=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                max_count=min(int(max_count), 5)
            )
            # Обновление внешнего ключа в Image
            images_to_update = db_manager.get_images_in_range(
                folder_name=folder,
                start_time=start_time,
                end_time=end_time
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
    result_df.to_csv('priroda_udivitelna/data/registration_results.csv', index=False)

if __name__ == "__main__":
    db_manager = DB('sqlite:///data/db.sqlite')
    process_data(db_manager)