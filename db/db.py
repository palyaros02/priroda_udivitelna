from sqlalchemy import create_engine, Column, Integer, String, Float

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True)
    folder_name = Column(String)
    image_name = Column(String)
    class_predict = Column(String)
    confidence = Column(Float)
    registration_class = Column(String) # fixed class_predict
    registration_date = Column(String)
    count = Column(Integer)


class DB:
    def __init__(self):
        self.engine = create_engine('sqlite:///data/db.sqlite',
                                    # echo=True
                                    )
        Base.metadata.create_all(self.engine)
        # Base.metadata.drop_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

    def add_image(self, folder_name, image_name, class_predict, confidence, registration_class, registration_date, count):
        image = Image(folder_name=folder_name, image_name=image_name, class_predict=class_predict, registration_class=registration_class, registration_date=registration_date, count=count, confidence=confidence)
        self.session.add(image)
        self.session.commit()

    def close(self):
        self.session.close()