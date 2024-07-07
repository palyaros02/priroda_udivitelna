from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from datetime import datetime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()

class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True)
    registration_id = Column(Integer, ForeignKey('registrations.id'), nullable=True)
    folder_name = Column(String)
    image_name = Column(String)
    class_predict = Column(String)
    confidence = Column(Float)
    registration_class = Column(String) # fixed class_predict
    registration_date = Column(String)
    count = Column(Integer)
    max_count = Column(Integer)

    registration = relationship("Registration", back_populates="images", uselist=False, single_parent=True, cascade="all, delete-orphan", passive_deletes=True, foreign_keys=[registration_id])

class Registration(Base):
    __tablename__ = 'registrations'
    id = Column(Integer, primary_key=True)
    folder_name = Column(String)
    class_name = Column(String)
    date_registration_start = Column(String) # fixed class_predict
    date_registration_end = Column(String)
    count = Column(Integer)

    #it is parent in many-to-one relationship
    images = relationship("Image", back_populates="registration", cascade="all, delete-orphan", passive_deletes=True)

class DB:
    def __init__(self):
        self.engine = create_engine('sqlite:///data/db.sqlite',
                                    # echo=True
                                    )
        Base.metadata.create_all(self.engine)
        # Base.metadata.drop_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

    # fixme
    def add_image(self, folder_name, image_name, class_predict, confidence, registration_class, registration_date, count, max_count=-1):
        registration_date = datetime.strptime(registration_date, "%Y:%m:%d %H:%M:%S")
        registration_date = datetime.strftime(registration_date, "%Y-%m-%d %H:%M:%S")
        image = Image(folder_name=folder_name, image_name=image_name, class_predict=class_predict, registration_class=registration_class, registration_date=registration_date, count=count, max_count=max_count, confidence=confidence)
        self.session.add(image)
        self.session.commit()

    def add_registration(self, folder_name: str, class_name: str, date_registration_start: str, date_registration_end: str, max_count: int):
        date_format = "%Y-%m-%d %H:%M:%S"
        date_registration_start = datetime.strptime(date_registration_start, date_format)
        date_registration_end = datetime.strptime(date_registration_end, date_format)

        registration = Registration(folder_name=folder_name, class_name=class_name, date_registration_start=date_registration_start, date_registration_end=date_registration_end, count=max_count)
        self.session.add(registration)
        self.session.commit()
        return registration.id
    def update_image_registration_id(self, image_id: int, registration_id: int):
        image = self.session.query(Image).filter_by(id=image_id).first()
        if image:
            image.registration_id = registration_id
            self.session.commit()
    def get_images_in_range(self, folder_name: str, start_time: datetime, end_time: datetime):
        return self.session.query(Image).filter(
            Image.folder_name == folder_name,
            Image.registration_date >= start_time,
            Image.registration_date <= end_time
        ).all()


    def close(self):
        self.session.close()